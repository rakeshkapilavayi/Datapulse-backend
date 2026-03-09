"""
Alternative PDF Report Generator using ReportLab
Use this if docx2pdf doesn't work on your server
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate PDF reports using ReportLab"""
    
    def create_pdf_report(self, session_data, output_path):
        """Create a PDF report"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#4472C4'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading1_style = ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=colors.HexColor('#4472C4'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            heading2_style = ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#4472C4'),
                spaceAfter=10,
                spaceBefore=10
            )
            
            # Title Page
            story.append(Paragraph("Machine Learning Analysis Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Comprehensive Data Science Workflow Documentation", styles['Normal']))
            story.append(Spacer(1, 0.5*inch))
            
            info = f"""
            <para align=center>
            Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
            Session ID: {session_data.get('session_id', 'N/A')}<br/>
            Dataset: {session_data.get('filename', 'N/A')}
            </para>
            """
            story.append(Paragraph(info, styles['Normal']))
            story.append(PageBreak())
            
            # Dataset Overview
            story.append(Paragraph("1. Dataset Overview", heading1_style))
            summary = session_data.get('summary', {})
            
            # Basic stats - build as HTML string
            total_rows = summary.get('total_rows', 'N/A')
            total_cols = summary.get('total_columns', 'N/A')
            
            # Format rows with comma if numeric
            if isinstance(total_rows, (int, float)):
                rows_str = f"{int(total_rows):,}"
            else:
                rows_str = str(total_rows)
            
            # Format columns
            if isinstance(total_cols, (int, float)):
                cols_str = str(int(total_cols))
            else:
                cols_str = str(total_cols)
            
            dataset_info = f"""
            <b>Dataset Statistics:</b><br/>
            • Total Rows: {rows_str}<br/>
            • Total Columns: {cols_str}<br/>
            • Numeric Columns: {summary.get('numeric_columns', 'N/A')}<br/>
            • Categorical Columns: {summary.get('categorical_columns', 'N/A')}<br/>
            • Missing Values: {summary.get('missing_values_total', 'N/A')}<br/>
            • Duplicate Rows: {summary.get('duplicate_rows', 'N/A')}
            """
            story.append(Paragraph(dataset_info, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Column Information Table
            if summary.get('column_info'):
                story.append(Paragraph("Column Information", heading2_style))
                
                col_data = [['Column', 'Type', 'Non-Null', 'Unique', 'Missing']]
                for col in summary['column_info'][:20]:  # Limit to 20 columns
                    col_data.append([
                        col['name'][:30],  # Truncate long names
                        col['dtype'],
                        str(col['non_null']),
                        str(col['unique']),
                        str(col['missing'])
                    ])
                
                table = Table(col_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            
            story.append(PageBreak())
            
            # Data Cleaning
            story.append(Paragraph("2. Data Cleaning Operations", heading1_style))
            cleaning_ops = session_data.get('cleaning_operations', [])
            
            if cleaning_ops:
                for i, op in enumerate(cleaning_ops, 1):
                    op_type = op.get('type', 'Unknown')
                    if op_type == 'remove_duplicates':
                        text = f"{i}. Removed Duplicates: {op.get('rows_removed', 0)} rows"
                    elif op_type == 'handle_missing':
                        text = f"{i}. Handled Missing Values in '{op.get('column')}': {op.get('method', 'N/A')}"
                    elif op_type == 'drop_column':
                        text = f"{i}. Dropped Column: '{op.get('column')}'"
                    else:
                        text = f"{i}. {op_type}"
                    
                    story.append(Paragraph(text, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph("No cleaning operations performed.", styles['Normal']))
            
            story.append(PageBreak())
            
            # Model Training
            story.append(Paragraph("3. Model Training", heading1_style))
            model_config = session_data.get('model_config', {})
            
            if not model_config or not model_config.get('task_type'):
                story.append(Paragraph("No model training was performed.", styles['Normal']))
            else:
                task_type = model_config.get('task_type', 'N/A')
                test_size = model_config.get('test_size', 0.2)
                
                # Format task type
                if isinstance(task_type, str):
                    task_type_str = task_type.title()
                else:
                    task_type_str = str(task_type)
                
                # Format test size
                if isinstance(test_size, (int, float)):
                    test_size_str = f"{test_size * 100:.0f}%"
                else:
                    test_size_str = str(test_size)
                
                model_info = f"""
                <b>Configuration:</b><br/>
                • Task Type: {task_type_str}<br/>
                • Target Column: {model_config.get('target_column', 'N/A')}<br/>
                • Algorithm: {model_config.get('model_type', 'N/A')}<br/>
                • Test Size: {test_size_str}<br/>
                • Hyperparameter Tuning: {'Enabled' if model_config.get('tune_params') else 'Disabled'}
                """
                story.append(Paragraph(model_info, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Performance Metrics
            story.append(Paragraph("Performance Metrics", heading2_style))
            report = session_data.get('performance_report', {})
            task_type = model_config.get('task_type', 'classification')
            
            if task_type == 'classification':
                metrics = f"""
                <b>Classification Metrics:</b><br/>
                • Accuracy: {report.get('accuracy', 0) * 100:.2f}%<br/>
                • F1 Score: {report.get('F1_Score', 0):.4f}<br/>
                • Cross-Validation Score: {report.get('Cross_Validation_Score', 0):.4f}
                """
            else:
                r2 = report.get('R² Score', 0)
                metrics = f"""
                <b>Regression Metrics:</b><br/>
                • R² Score: {r2:.4f} ({r2 * 100:.1f}% variance explained)<br/>
                • RMSE: {(report.get('Mean Squared Error', 0) ** 0.5):.4f}<br/>
                • MAE: {report.get('Mean Absolute Error', 0):.4f}<br/>
                • Cross-Validation Score: {report.get('Cross_Validation_Score', 0):.4f}
                """
            
            story.append(Paragraph(metrics, styles['Normal']))
            
            story.append(PageBreak())
            
            # Example Prediction
            prediction = session_data.get('example_prediction')
            if prediction:
                story.append(Paragraph("4. Example Prediction", heading1_style))
                
                story.append(Paragraph("<b>Input Values:</b>", styles['Normal']))
                inputs = prediction.get('inputs', {})
                for feature, value in list(inputs.items())[:10]:  # Limit to 10 features
                    story.append(Paragraph(f"• {feature}: {value}", styles['Normal']))
                
                story.append(Spacer(1, 0.2*inch))
                result = prediction.get('result', {})
                pred_value = result.get('prediction', ['N/A'])[0]
                
                pred_text = f"<b>Predicted Value:</b> <font color='green' size=14>{pred_value}</font>"
                story.append(Paragraph(pred_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            raise