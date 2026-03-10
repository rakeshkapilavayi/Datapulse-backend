from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import plotly.express as px
import logging
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import pickle
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
from data_processor import DataProcessor
from ml_engine import MLEngine
from insights_generator import InsightsGenerator
from visualization_engine import VisualizationEngine
from report_generator import build_docx, build_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datapulse_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ FIXED: was r"/api/*" which missed /clean/*, /visualizations/*, /outliers/* etc.
# Now covers ALL routes with r"/*"
CORS(app, resources={r"/*": {
    "origins": [
        "https://datapulsee.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', 104857600))  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# In-memory session storage
session_data = {}

# Initialize engines
data_processor = DataProcessor()
ml_engine      = MLEngine()
viz_engine     = VisualizationEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def track_cleaning_operation(session_id: str, operation_type: str, details: dict):
    """Append a cleaning action to the session log."""
    if session_id not in session_data:
        return
    session_data[session_id].setdefault('cleaning_operations', []).append({
        'type':      operation_type,
        'timestamp': datetime.now().isoformat(),
        **details
    })


def track_outlier_treatment(session_id: str, column: str, method: str, details: dict):
    """Append an outlier-treatment action to the session log."""
    if session_id not in session_data:
        return
    session_data[session_id].setdefault('report_outliers', []).append({
        'column':    column or 'all columns',
        'method':    method,
        'timestamp': datetime.now().isoformat(),
        **details
    })


def _build_report_data(session_id: str) -> dict:
    """
    Assemble everything stored in the session into the dict
    that report_generator.build_docx / build_pdf expects.
    """
    sess = session_data[session_id]
    df   = sess.get('cleaned_df')

    # Summary
    try:
        summary_raw = data_processor.get_summary(df) if df is not None else {}
    except Exception:
        summary_raw = {}

    missing_total = sum(
        c.get('missing', 0) for c in summary_raw.get('column_info', [])
    ) if summary_raw else '—'

    summary = {
        'total_rows':           summary_raw.get('total_rows', '—'),
        'total_columns':        summary_raw.get('total_columns', '—'),
        'numeric_columns':      summary_raw.get('numeric_columns', '—'),
        'categorical_columns':  summary_raw.get('categorical_columns', '—'),
        'missing_values_total': missing_total,
        'duplicate_rows':       summary_raw.get('duplicate_rows', '—'),
    }

    # Cleaning — merge auto-clean report + manual actions log
    cleaning = {}
    if sess.get('cleaning_report'):
        cleaning['auto_clean'] = sess['cleaning_report']
    if sess.get('cleaning_operations'):
        manual_actions = {}
        remove_dups    = False
        for op in sess['cleaning_operations']:
            if op['type'] == 'handle_missing':
                manual_actions[op.get('column', '?')] = op.get('method', '?')
            elif op['type'] == 'remove_duplicates':
                remove_dups = True
        if manual_actions or remove_dups:
            cleaning['manual'] = {
                'missing_actions':   manual_actions,
                'remove_duplicates': remove_dups,
            }

    # ML info
    ml_info    = {}
    model_data = sess.get('model')
    if model_data:
        ml_info = {
            'task_type':     model_data.get('task_type', '—'),
            'model_type':    model_data.get('model_type', '—'),
            'target_column': model_data.get('target_column', '—'),
            'test_size':     model_data.get('test_size', 0.2),
            'report':        sess.get('ml_report', {}),
        }

    # Prediction
    pred_raw   = sess.get('report_prediction', {})
    prediction = {}
    if pred_raw:
        prediction = {
            'input':  pred_raw.get('inputs') or pred_raw.get('input', {}),
            'result': pred_raw.get('result', {}),
        }

    return {
        'upload': {
            'filename':   sess.get('filename', '—'),
            'created_at': sess.get('created_at', '—'),
        },
        'summary':    summary,
        'cleaning':   cleaning,
        'outliers':   sess.get('report_outliers', []),
        'ml':         ml_info,
        'prediction': prediction,
        'insights':   sess.get('report_insights', {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':    'healthy',
        'timestamp': datetime.now().isoformat(),
        'version':   '2.0.0'
    })


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process primary dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed'}), 400

        session_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename   = secure_filename(file.filename)
        filepath   = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return jsonify({'error': f'Error loading file: {e}'}), 400

        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400
        if len(df.columns) == 0:
            return jsonify({'error': 'Dataset has no columns'}), 400

        summary = data_processor.get_summary(df)

        session_data[session_id] = {
            'original_df':         df.copy(),
            'cleaned_df':          df.copy(),
            'filename':            filename,
            'filepath':            filepath,
            'summary':             summary,
            'created_at':          datetime.now().isoformat(),
            'cleaning_operations': [],
            'report_outliers':     [],
        }

        logger.info(f"File uploaded: {filename}, Session: {session_id}, Shape: {df.shape}")

        return jsonify({
            'session_id': session_id,
            'filename':   filename,
            'summary':    summary,
            'preview':    df.head(10).to_dict('records')
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {e}'}), 500


@app.route('/api/upload_secondary', methods=['POST'])
def upload_secondary_dataset():
    """Upload a secondary dataset for merge/concat operations."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file       = request.files['file']
        session_id = request.form.get('session_id')

        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400

        filename = secure_filename(file.filename)

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or Excel'}), 400

        session_data[session_id]['secondary_df'] = df
        logger.info(f"Secondary dataset uploaded for session {session_id}: {filename}")

        return jsonify({
            'success': True,
            'filename': filename,
            'rows':     int(df.shape[0]),
            'columns':  int(df.shape[1])
        })

    except Exception as e:
        logger.error(f"Secondary upload error: {e}")
        return jsonify({'error': f'Upload failed: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/summary/<session_id>', methods=['GET'])
def get_summary(session_id):
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        df      = session_data[session_id]['cleaned_df']
        summary = data_processor.get_summary(df)
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return jsonify({'error': f'Failed to get summary: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/clean/manual', methods=['POST'])
def manual_clean():
    """Apply manual cleaning and track every action for the report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id        = data.get('session_id')
        missing_actions   = data.get('missing_actions', {})
        remove_duplicates = data.get('remove_duplicates', False)

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df         = session_data[session_id]['cleaned_df']
        cleaned_df = data_processor.manual_cleaning(df, missing_actions, remove_duplicates)
        session_data[session_id]['cleaned_df'] = cleaned_df

        for col, method in missing_actions.items():
            rows_affected = int(df[col].isna().sum()) if col in df.columns else 0
            track_cleaning_operation(session_id, 'handle_missing', {
                'column':        col,
                'method':        method,
                'rows_affected': rows_affected,
            })

        if remove_duplicates:
            dup_count = int(df.duplicated().sum())
            track_cleaning_operation(session_id, 'remove_duplicates', {
                'rows_removed': dup_count,
            })

        logger.info(f"Manual cleaning applied for session {session_id}")

        return jsonify({
            'success': True,
            'summary': data_processor.get_summary(cleaned_df),
            'preview': cleaned_df.head(10).to_dict('records')
        })

    except Exception as e:
        logger.error(f"Manual cleaning error: {e}")
        return jsonify({'error': f'Manual cleaning failed: {e}'}), 500


@app.route('/api/clean/auto', methods=['POST'])
def auto_clean():
    """Apply automated cleaning."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df                 = session_data[session_id]['cleaned_df']
        cleaned_df, report = data_processor.auto_clean(df)

        session_data[session_id]['cleaned_df']      = cleaned_df
        session_data[session_id]['cleaning_report'] = report

        logger.info(f"Auto cleaning applied for session {session_id}")

        return jsonify({
            'success': True,
            'report':  report,
            'summary': data_processor.get_summary(cleaned_df),
            'preview': cleaned_df.head(10).to_dict('records')
        })

    except Exception as e:
        logger.error(f"Auto cleaning error: {e}")
        return jsonify({'error': f'Auto cleaning failed: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/outliers/<session_id>', methods=['GET'])
def get_outliers(session_id):
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df           = session_data[session_id]['cleaned_df']
        outlier_info = data_processor.get_all_outliers(df)

        max_values = 1000
        for col, info in outlier_info.items():
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > max_values:
                    values = values.sample(n=max_values, random_state=42)
                info['values'] = values.tolist()

        return jsonify(outlier_info)

    except Exception as e:
        logger.error(f"Outlier detection error: {e}")
        return jsonify({'error': f'Failed to detect outliers: {e}'}), 500


@app.route('/api/outliers/treat', methods=['POST'])
def treat_outliers():
    """Treat outliers and track the action for the report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id')
        column     = data.get('column')
        method     = data.get('method', 'cap')

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df = session_data[session_id]['cleaned_df']

        if column:
            treated_df, report = data_processor.treat_outliers(df, column, method)
        else:
            treated_df, report = data_processor.treat_all_outliers(df, method)

        session_data[session_id]['cleaned_df'] = treated_df

        track_outlier_treatment(session_id, column or 'all columns', method, {
            'count': report.get('treated_count') or report.get('outliers_treated', '?'),
        })

        logger.info(f"Outliers treated for session {session_id}")

        return jsonify({
            'success': True,
            'report':  report,
            'summary': data_processor.get_summary(treated_df)
        })

    except Exception as e:
        logger.error(f"Outlier treatment error: {e}")
        return jsonify({'error': f'Failed to treat outliers: {e}'}), 500


@app.route('/api/outliers/<session_id>/boxplot/<column>', methods=['GET'])
def get_boxplot(session_id, column):
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df = session_data[session_id]['cleaned_df']

        if column not in df.columns:
            return jsonify({'error': f'Column {column} not found'}), 404
        if df[column].dtype not in ['float64', 'int64']:
            return jsonify({'error': f'Column {column} is not numerical'}), 400

        fig = px.box(df, y=column, title=f'Box Plot: {column}',
                     color_discrete_sequence=['#2563eb'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827', family='Inter, sans-serif'),
            yaxis=dict(gridcolor='#e5e7eb'),
            height=400
        )
        return jsonify({'figure': fig.to_json()})

    except Exception as e:
        logger.error(f"Box plot error: {e}")
        return jsonify({'error': f'Failed to create box plot: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/visualizations/<session_id>', methods=['GET'])
def get_visualizations(session_id):
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        df       = session_data[session_id]['cleaned_df']
        viz_type = request.args.get('type', 'all')
        return jsonify(viz_engine.create_visualizations(df, viz_type))
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': f'Failed to create visualizations: {e}'}), 500


@app.route('/api/visualizations/<session_id>/custom', methods=['POST'])
def create_custom_visualization(session_id):
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        data = request.get_json()
        df   = session_data[session_id]['cleaned_df']

        chart_config = {
            'type':    data.get('type'),
            'xAxis':   data.get('xAxis'),
            'yAxis':   data.get('yAxis'),
            'colorBy': data.get('colorBy')
        }

        figure = viz_engine.create_custom_chart(df, chart_config)
        if not figure:
            return jsonify({'error': 'Failed to create chart'}), 400
        return jsonify({'figure': figure})

    except Exception as e:
        logger.error(f"Custom visualization error: {e}")
        return jsonify({'error': f'Failed to create custom chart: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ML — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/ml/train', methods=['POST'])
def train_model():
    """Train a machine learning model and store config + report in session."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id    = data.get('session_id')
        target_column = data.get('target_column')
        task_type     = data.get('task_type')
        model_type    = data.get('model_type')
        test_size     = float(data.get('test_size', 0.2))
        tune_params   = data.get('tune_params', False)

        if not all([session_id, target_column, task_type, model_type]):
            return jsonify({'error': 'Missing required parameters'}), 400
        if not (0.1 <= test_size <= 0.9):
            return jsonify({'error': 'test_size must be between 0.1 and 0.9'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df = session_data[session_id]['cleaned_df']

        pipeline, report, cm, cm_fig, features, label_encoder = ml_engine.train_model(
            df=df,
            target_column=target_column,
            task_type=task_type,
            model_type=model_type,
            test_size=test_size,
            tune_params=tune_params
        )

        model_filename = f"model_{session_id}_{int(time.time())}.pkl"
        model_path     = os.path.join(app.config['MODELS_FOLDER'], model_filename)

        model_data = {
            'pipeline':      pipeline,
            'features':      features,
            'label_encoder': label_encoder,
            'task_type':     task_type,
            'model_type':    model_type,
            'target_column': target_column,
            'test_size':     test_size,
            'trained_at':    datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        session_data[session_id]['model']          = model_data
        session_data[session_id]['model_filename'] = model_filename
        session_data[session_id]['ml_report']      = report

        logger.info(f"Model trained for session {session_id}: {model_type}")

        return jsonify({
            'success':              True,
            'report':               report,
            'model_filename':       model_filename,
            'confusion_matrix_fig': cm_fig.to_json() if cm_fig else None,
            'features':             features
        }), 200

    except Exception as e:
        logger.error(f"Model training error: {e}")
        return jsonify({'error': f'Model training failed: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ML — PREDICT
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/ml/predict', methods=['POST'])
def predict():
    """Make a prediction and store the example in the session for the report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id')
        input_data = data.get('input_data')

        if not session_id or not input_data:
            return jsonify({'error': 'Session ID and input data are required'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        session    = session_data[session_id]
        model_data = session.get('model')
        if not model_data:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404

        features      = model_data['features']
        label_encoder = model_data.get('label_encoder')
        task_type     = model_data.get('task_type', 'classification')
        session_df    = session.get('cleaned_df')

        converted_input = {}
        for feature in features:
            value = input_data.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for feature: {feature}'}), 400

            if session_df is not None and feature in session_df.columns:
                dtype = session_df[feature].dtype
                try:
                    if dtype in ['int64', 'int32']:
                        converted_input[feature] = int(value)
                    elif dtype in ['float64', 'float32']:
                        converted_input[feature] = float(value)
                    else:
                        converted_input[feature] = str(value)
                except (ValueError, TypeError):
                    converted_input[feature] = value
            else:
                converted_input[feature] = value

        model_obj = model_data.get('pipeline') or model_data.get('model')
        result    = ml_engine.predict(model_obj, converted_input, features, label_encoder, task_type)

        session_data[session_id]['report_prediction'] = {
            'inputs':    converted_input,
            'result':    result,
            'timestamp': datetime.now().isoformat(),
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ML — SUITABLE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/ml/suitable-columns/<session_id>', methods=['GET'])
def get_suitable_columns(session_id):
    try:
        task_type = request.args.get('task_type', 'classification')

        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        summary     = data_processor.get_summary(session_data[session_id]['cleaned_df'])
        column_info = summary.get('column_info', [])
        suitable    = []

        if task_type == 'classification':
            for col in column_info:
                if (col['dtype'] in ['int64', 'float64'] and col['unique'] <= 20) \
                        or col['dtype'] in ['object', 'category']:
                    suitable.append({
                        'name':   col['name'],
                        'dtype':  col['dtype'],
                        'unique': col['unique']
                    })
        else:
            for col in column_info:
                if col['dtype'] in ['int64', 'float64']:
                    suitable.append({
                        'name':   col['name'],
                        'dtype':  col['dtype'],
                        'unique': col['unique']
                    })

        return jsonify({'suitable_columns': suitable}), 200

    except Exception as e:
        logger.error(f"Error getting suitable columns: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/insights/<session_id>', methods=['GET'])
def get_insights(session_id):
    try:
        insight_type = request.args.get('type', 'raw')
        logger.info(f"Insights request: session={session_id}, type={insight_type}")

        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df = session_data[session_id].get('cleaned_df')
        if df is None:
            return jsonify({'error': 'Dataset not found in session'}), 404

        try:
            summary = data_processor.get_summary(df)
        except Exception as e:
            return jsonify({'error': f'Failed to generate summary: {e}'}), 500

        cleaning_report = session_data[session_id].get('cleaning_report')
        ml_report       = session_data[session_id].get('ml_report')
        insights_engine = InsightsGenerator()

        if insight_type == 'raw':
            raw = insights_engine.generate_structured_insights(df, summary)
            session_data[session_id]['report_insights'] = raw
            return jsonify(raw), 200

        elif insight_type == 'enhanced':
            result = insights_engine.generate_enhanced_insights(
                df, summary, cleaning_report, ml_report
            )
            return jsonify({'insights': result}), 200

        elif insight_type == 'quick':
            result = insights_engine.generate_quick_summary(df, summary)
            return jsonify({'insights': result}), 200

        else:
            return jsonify({'error': 'Invalid insight type. Use: raw, enhanced, or quick'}), 400

    except Exception as e:
        logger.error(f"Insights error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD — Dataset & Model
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/download/<session_id>', methods=['GET'])
def download_data(session_id):
    """Download cleaned dataset as CSV."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df       = session_data[session_id]['cleaned_df']
        filename = session_data[session_id]['filename']

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"cleaned_{filename}"
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': f'Download failed: {e}'}), 500


@app.route('/api/download/model/<session_id>', methods=['GET'])
def download_model(session_id):
    """Download trained model as .pkl."""
    try:
        session = session_data.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        model_filename = session.get('model_filename')
        if not model_filename:
            return jsonify({'error': 'No model found'}), 404

        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404

        return send_file(
            model_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=model_filename
        )

    except Exception as e:
        logger.error(f"Download model error: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — Generate & Download (DOCX + PDF)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/report/save-context', methods=['POST'])
def save_report_context():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id')
        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        if 'insights' in data:
            session_data[session_id]['report_insights'] = data['insights']
        if 'prediction' in data:
            session_data[session_id]['report_prediction'] = data['prediction']

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"save-report-context error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/download/docx/<session_id>', methods=['GET'])
def download_report_docx(session_id):
    """Download the full session report as a Word (.docx) document."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        report_data = _build_report_data(session_id)
        docx_bytes  = build_docx(report_data)

        stem = session_data[session_id].get('filename', 'report').rsplit('.', 1)[0]
        return send_file(
            io.BytesIO(docx_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"DataPulse_Report_{stem}.docx"
        )

    except Exception as e:
        logger.error(f"DOCX report error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to generate DOCX report: {e}'}), 500


@app.route('/api/report/download/pdf/<session_id>', methods=['GET'])
def download_report_pdf(session_id):
    """Download the full session report as a PDF."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        report_data = _build_report_data(session_id)
        pdf_bytes   = build_pdf(report_data)

        stem = session_data[session_id].get('filename', 'report').rsplit('.', 1)[0]
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"DataPulse_Report_{stem}.pdf"
        )

    except Exception as e:
        logger.error(f"PDF report error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to generate PDF report: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/notebook/execute', methods=['POST'])
def execute_notebook_cell():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id')
        code       = data.get('code')

        if not session_id or not code:
            return jsonify({'error': 'Session ID and code are required'}), 400
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        df       = session_data[session_id]['cleaned_df'].copy()
        other_df = session_data[session_id].get('secondary_df')
        if other_df is not None:
            other_df = other_df.copy()

        result = data_processor.execute_code(code, df, other_df)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return jsonify({'error': f'Code execution failed: {e}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port  = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)
