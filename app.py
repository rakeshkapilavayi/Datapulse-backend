from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import logging
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import pickle
import json
import time
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Import your existing modules
from data_processor import DataProcessor
from ml_engine import MLEngine
from insights_generator import InsightsGenerator
from visualization_engine import VisualizationEngine

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
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration from environment variables
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', 104857600))  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# In-memory storage (use Redis in production)
session_data = {}

# Initialize engines
data_processor = DataProcessor()
ml_engine = MLEngine()
insights_generator = InsightsGenerator()
viz_engine = VisualizationEngine()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process dataset - UPDATED to store summary"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed'}), 400
        
        # Generate session ID
        session_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Load dataset
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return jsonify({'error': f'Error loading file: {str(e)}'}), 400
        
        # Validate dataset
        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400
        
        if len(df.columns) == 0:
            return jsonify({'error': 'Dataset has no columns'}), 400
        
        # Get summary
        summary = data_processor.get_summary(df)
        
        # Store in session - UPDATED to include summary
        session_data[session_id] = {
            'original_df': df.copy(),
            'cleaned_df': df.copy(),
            'filename': filename,
            'filepath': filepath,
            'summary': summary,  # ADDED: Store summary for later use
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"File uploaded successfully: {filename}, Session: {session_id}, Shape: {df.shape}")
        
        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'summary': summary,
            'preview': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/summary/<session_id>', methods=['GET'])
def get_summary(session_id):
    """Get dataset summary"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        summary = data_processor.get_summary(df)
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        return jsonify({'error': f'Failed to get summary: {str(e)}'}), 500


@app.route('/api/clean/manual', methods=['POST'])
def manual_clean():
    """Apply manual cleaning"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        missing_actions = data.get('missing_actions', {})
        remove_duplicates = data.get('remove_duplicates', False)
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        cleaned_df = data_processor.manual_cleaning(df, missing_actions, remove_duplicates)
        
        session_data[session_id]['cleaned_df'] = cleaned_df
        
        logger.info(f"Manual cleaning applied for session {session_id}")
        
        return jsonify({
            'success': True,
            'summary': data_processor.get_summary(cleaned_df),
            'preview': cleaned_df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Manual cleaning error: {str(e)}")
        return jsonify({'error': f'Manual cleaning failed: {str(e)}'}), 500


@app.route('/api/clean/auto', methods=['POST'])
def auto_clean():
    """Apply automated cleaning"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        cleaned_df, report = data_processor.auto_clean(df)
        
        session_data[session_id]['cleaned_df'] = cleaned_df
        session_data[session_id]['cleaning_report'] = report
        
        logger.info(f"Auto cleaning applied for session {session_id}")
        
        return jsonify({
            'success': True,
            'report': report,
            'summary': data_processor.get_summary(cleaned_df),
            'preview': cleaned_df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Auto cleaning error: {str(e)}")
        return jsonify({'error': f'Auto cleaning failed: {str(e)}'}), 500


@app.route('/api/visualizations/<session_id>', methods=['GET'])
def get_visualizations(session_id):
    """Get all visualizations"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        viz_type = request.args.get('type', 'all')
        
        visualizations = viz_engine.create_visualizations(df, viz_type)
        
        return jsonify(visualizations)
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({'error': f'Failed to create visualizations: {str(e)}'}), 500


@app.route('/api/outliers/<session_id>', methods=['GET'])
def get_outliers(session_id):
    """Get outlier information with column values for box plots"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        outlier_info = data_processor.get_all_outliers(df)
        
        # Add column values for box plots (sample if too large)
        max_values = 1000  # Limit values for performance
        for col, info in outlier_info.items():
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > max_values:
                    values = values.sample(n=max_values, random_state=42)
                info['values'] = values.tolist()
        
        return jsonify(outlier_info)
        
    except Exception as e:
        logger.error(f"Outlier detection error: {str(e)}")
        return jsonify({'error': f'Failed to detect outliers: {str(e)}'}), 500


@app.route('/api/outliers/treat', methods=['POST'])
def treat_outliers():
    """Treat outliers"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        column = data.get('column')
        method = data.get('method', 'cap')
        
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
        
        logger.info(f"Outliers treated for session {session_id}")
        
        return jsonify({
            'success': True,
            'report': report,
            'summary': data_processor.get_summary(treated_df)
        })
        
    except Exception as e:
        logger.error(f"Outlier treatment error: {str(e)}")
        return jsonify({'error': f'Failed to treat outliers: {str(e)}'}), 500


@app.route('/api/ml/train', methods=['POST'])
def train_model():
    """Train machine learning model"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        target_column = data.get('target_column')
        task_type = data.get('task_type')
        model_type = data.get('model_type')
        tune_params = data.get('tune_params', False)
        
        if not all([session_id, target_column, task_type, model_type]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        
        # Train model
        model, report, cm, cm_fig, features, label_encoder = ml_engine.train_model(
            df, target_column, task_type, model_type, tune_params
        )
        
        # Save model
        model_filename = f"{session_id}_{model_type}.pkl"
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': features,
                'label_encoder': label_encoder,
                'task_type': task_type
            }, f)
        
        # Store in session
        session_data[session_id]['model_path'] = model_path
        session_data[session_id]['ml_report'] = report
        
        # Convert confusion matrix figure to JSON if exists
        cm_json = cm_fig.to_json() if cm_fig else None
        
        logger.info(f"Model trained for session {session_id}: {model_type}")
        
        return jsonify({
            'success': True,
            'report': report,
            'confusion_matrix': cm_json,
            'model_filename': model_filename
        })
        
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500


@app.route('/api/ml/predict', methods=['POST'])
def predict():
    """Make predictions with trained model"""
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
        
        if 'model_path' not in session_data[session_id]:
            return jsonify({'error': 'No trained model found'}), 400
        
        # Load model
        with open(session_data[session_id]['model_path'], 'rb') as f:
            model_data = pickle.load(f)
        
        # Make prediction
        prediction = ml_engine.predict(
            model_data['model'],
            input_data,
            model_data['features'],
            model_data['label_encoder']
        )
        
        return jsonify({
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/insights/<session_id>', methods=['GET'])
def get_insights(session_id):
    """Generate insights for the dataset - FIXED with better error handling"""
    try:
        insight_type = request.args.get('type', 'raw')
        logger.info(f"Insights request: session={session_id}, type={insight_type}")
        
        # Validate session exists
        if session_id not in session_data:
            logger.error(f"Session not found: {session_id}")
            return jsonify({'error': 'Session not found'}), 404
        
        # Get cleaned dataframe
        df = session_data[session_id].get('cleaned_df')
        if df is None:
            logger.error(f"No cleaned_df found for session: {session_id}")
            return jsonify({'error': 'Dataset not found in session'}), 404
        
        logger.info(f"Dataset found: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # CRITICAL FIX: Generate summary dynamically instead of using stored one
        # The stored summary might not exist or be outdated
        try:
            summary = data_processor.get_summary(df)
            logger.info(f"Summary generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500
        
        # Get optional reports (may not exist)
        cleaning_report = session_data[session_id].get('cleaning_report')
        ml_report = session_data[session_id].get('ml_report')
        
        # Initialize insights generator
        insights_engine = InsightsGenerator()
        
        # Generate insights based on type
        if insight_type == 'raw':
            logger.info("Generating raw structured insights...")
            try:
                raw_insights = insights_engine.generate_structured_insights(df, summary)
                logger.info("Raw insights generated successfully")
                return jsonify(raw_insights), 200
            except Exception as e:
                logger.error(f"Error generating raw insights: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Failed to generate raw insights: {str(e)}'}), 500
            
        elif insight_type == 'enhanced':
            logger.info("Generating enhanced AI insights...")
            try:
                ai_insights = insights_engine.generate_enhanced_insights(
                    df, summary, cleaning_report, ml_report
                )
                logger.info("Enhanced insights generated successfully")
                return jsonify({'insights': ai_insights}), 200
            except Exception as e:
                logger.error(f"Error generating enhanced insights: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Failed to generate enhanced insights: {str(e)}'}), 500
            
        elif insight_type == 'quick':
            logger.info("Generating quick AI summary...")
            try:
                quick_summary = insights_engine.generate_quick_summary(df, summary)
                logger.info("Quick summary generated successfully")
                return jsonify({'insights': quick_summary}), 200
            except Exception as e:
                logger.error(f"Error generating quick summary: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Failed to generate quick summary: {str(e)}'}), 500
            
        else:
            logger.error(f"Invalid insight type: {insight_type}")
            return jsonify({'error': 'Invalid insight type. Use: raw, enhanced, or quick'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error in insights endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/download/<session_id>', methods=['GET'])
def download_data(session_id):
    """Download cleaned dataset"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        filename = session_data[session_id]['filename']
        
        # Create CSV in memory
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
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/download/model/<session_id>', methods=['GET'])
def download_model(session_id):
    """Download trained model"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        if 'model_path' not in session_data[session_id]:
            return jsonify({'error': 'No trained model found'}), 400
        
        model_path = session_data[session_id]['model_path']
        
        return send_file(
            model_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=os.path.basename(model_path)
        )
        
    except Exception as e:
        logger.error(f"Model download error: {str(e)}")
        return jsonify({'error': f'Model download failed: {str(e)}'}), 500


@app.route('/api/notebook/execute', methods=['POST'])
def execute_notebook_cell():
    """Execute notebook cell code with support for secondary dataset"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        code = data.get('code')
        
        if not session_id or not code:
            return jsonify({'error': 'Session ID and code are required'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df'].copy()
        
        # Get secondary dataset if available
        other_df = session_data[session_id].get('secondary_df')
        if other_df is not None:
            other_df = other_df.copy()
        
        # Execute code safely
        result = data_processor.execute_code(code, df, other_df)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        return jsonify({'error': f'Code execution failed: {str(e)}'}), 500


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/visualizations/<session_id>/custom', methods=['POST'])
def create_custom_visualization(session_id):
    """Create custom visualization based on user selection"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        data = request.get_json()
        df = session_data[session_id]['cleaned_df']
        
        # Pass the data directly - the visualization engine expects camelCase
        chart_config = {
            'type': data.get('type'),
            'xAxis': data.get('xAxis'),
            'yAxis': data.get('yAxis'),
            'colorBy': data.get('colorBy')
        }
        
        figure = viz_engine.create_custom_chart(df, chart_config)
        
        if not figure:
            return jsonify({'error': 'Failed to create chart'}), 400
        
        return jsonify({'figure': figure})
        
    except Exception as e:
        logger.error(f"Custom visualization error: {str(e)}")
        return jsonify({'error': f'Failed to create custom chart: {str(e)}'}), 500


@app.route('/api/outliers/<session_id>/boxplot/<column>', methods=['GET'])
def get_boxplot(session_id, column):
    """Get box plot for a specific column"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        
        if column not in df.columns:
            return jsonify({'error': f'Column {column} not found'}), 404
        
        if df[column].dtype not in ['float64', 'int64']:
            return jsonify({'error': f'Column {column} is not numerical'}), 400
        
        # Create box plot using plotly
        fig = px.box(
            df, 
            y=column, 
            title=f'Box Plot: {column}',
            color_discrete_sequence=['#2563eb']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827', family='Inter, sans-serif'),
            yaxis=dict(gridcolor='#e5e7eb'),
            height=400
        )
        
        return jsonify({'figure': fig.to_json()})
        
    except Exception as e:
        logger.error(f"Box plot error: {str(e)}")
        return jsonify({'error': f'Failed to create box plot: {str(e)}'}), 500


@app.route('/api/upload_secondary', methods=['POST'])
def upload_secondary_dataset():
    """Upload a secondary dataset for merge/concat operations"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id')
        
        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Save and load dataset
        filename = secure_filename(file.filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or Excel'}), 400
        
        # Store in session
        session_data[session_id]['secondary_df'] = df
        
        logger.info(f"Secondary dataset uploaded for session {session_id}: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1])
        })
        
    except Exception as e:
        logger.error(f"Secondary upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    

# Add this to your app.py Flask routes

@app.route('/api/ml/train', methods=['POST'])
def train_ml_model():
    """Train a machine learning model with custom parameters"""
    try:
        data = request.json
        session_id = data.get('session_id')
        task_type = data.get('task_type')
        model_type = data.get('model_type')
        target_column = data.get('target_column')
        test_size = data.get('test_size', 0.2)  # Default 20%
        tune_params = data.get('tune_params', False)
        
        if not all([session_id, task_type, model_type, target_column]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Validate test_size
        test_size = float(test_size)
        if not (0.1 <= test_size <= 0.9):
            return jsonify({'error': 'test_size must be between 0.1 and 0.9'}), 400
        
        # Get dataset
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = session_data[session_id]['cleaned_df']
        
        # Train model
        ml_engine = MLEngine()
        pipeline, report, cm, cm_fig, features, label_encoder = ml_engine.train_model(
            df=df,
            target_column=target_column,
            task_type=task_type,
            model_type=model_type,
            test_size=test_size,
            tune_params=tune_params
        )
        
        # Save model
        model_filename = f"model_{session_id}_{int(time.time())}.pkl"
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        
        model_data = {
            'pipeline': pipeline,
            'features': features,
            'label_encoder': label_encoder,
            'task_type': task_type,
            'model_type': model_type,
            'target_column': target_column
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Store in session
        session_data[session_id]['model'] = model_data
        session_data[session_id]['model_filename'] = model_filename
        
        # Prepare response
        response = {
            'report': report,
            'model_filename': model_filename,
            'confusion_matrix_fig': cm_fig.to_json() if cm_fig else None
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"ML training error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Also add an endpoint to get suitable columns for target selection
@app.route('/api/ml/suitable-columns/<session_id>', methods=['GET'])
def get_suitable_columns(session_id):
    """Get columns suitable for classification or regression"""
    try:
        task_type = request.args.get('task_type', 'classification')
        
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        summary = data_processor.get_summary(session_data[session_id]['cleaned_df'])
        
        column_info = summary.get('column_info', [])
        suitable_columns = []
        
        if task_type == 'classification':
            # For classification: categorical or numeric with few unique values
            for col in column_info:
                is_numeric_few_unique = (
                    col['dtype'] in ['int64', 'float64'] and 
                    col['unique'] <= 20
                )
                is_categorical = col['dtype'] in ['object', 'category']
                
                if is_numeric_few_unique or is_categorical:
                    suitable_columns.append({
                        'name': col['name'],
                        'dtype': col['dtype'],
                        'unique': col['unique']
                    })
        else:
            # For regression: numerical columns only
            for col in column_info:
                if col['dtype'] in ['int64', 'float64']:
                    suitable_columns.append({
                        'name': col['name'],
                        'dtype': col['dtype'],
                        'unique': col['unique']
                    })
        
        return jsonify({'suitable_columns': suitable_columns}), 200
        
    except Exception as e:
        logger.error(f"Error getting suitable columns: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)