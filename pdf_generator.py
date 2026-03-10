"""
PDF Report Generator using ReportLab
Used by report_generator.py for PDF generation on the server
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
        """Create a PDF report and save to output_path"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # ── Custom styles ──────────────────────────────────────────────
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

            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                leading=16
            )

            # ── Title Page ─────────────────────────────────────────────────
            story.append(Paragraph("Machine Learning Analysis Report", title_style))
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph(
                "Comprehensive Data Science Workflow Documentation",
                ParagraphStyle('sub', parent=styles['Normal'], fontSize=13,
                               textColor=colors.HexColor('#595959'), alignment=TA_CENTER)
            ))
            story.append(Spacer(1, 0.5 * inch))

            info_text = (
                f"<para align=center>"
                f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
                f"Session ID: {session_data.get('session_id', 'N/A')}<br/>"
                f"Dataset: {session_data.get('filename', 'N/A')}"
                f"</para>"
            )
            story.append(Paragraph(info_text, styles['Normal']))
            story.append(PageBreak())

            # ── 1. Dataset Overview ────────────────────────────────────────
            story.append(Paragraph("1. Dataset Overview", heading1_style))
            summary = session_data.get('summary', {})

            total_rows = summary.get('total_rows', 'N/A')
            rows_str = f"{int(total_rows):,}" if isinstance(total_rows, (int, float)) else str(total_rows)

            total_cols = summary.get('total_columns', 'N/A')
            cols_str = str(int(total_cols)) if isinstance(total_cols, (int, float)) else str(total_cols)

            dataset_info = (
                f"<b>Dataset Statistics:</b><br/>"
                f"• Total Rows: {rows_str}<br/>"
                f"• Total Columns: {cols_str}<br/>"
                f"• Numeric Columns: {summary.get('numeric_columns', 'N/A')}<br/>"
                f"• Categorical Columns: {summary.get('categorical_columns', 'N/A')}<br/>"
                f"• Missing Values: {summary.get('missing_values_total', 'N/A')}<br/>"
                f"• Duplicate Rows: {summary.get('duplicate_rows', 'N/A')}"
            )
            story.append(Paragraph(dataset_info, body_style))
            story.append(Spacer(1, 0.2 * inch))

            # Column Information Table
            if summary.get('column_info'):
                story.append(Paragraph("Column Information", heading2_style))

                col_data = [['Column', 'Type', 'Non-Null', 'Unique', 'Missing']]
                for col in summary['column_info'][:20]:
                    col_data.append([
                        str(col['name'])[:30],
                        str(col['dtype']),
                        str(col['non_null']),
                        str(col['unique']),
                        str(col['missing'])
                    ])

                table = Table(col_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC'))
                ]))
                story.append(table)

            story.append(PageBreak())

            # ── 2. Data Cleaning ───────────────────────────────────────────
            story.append(Paragraph("2. Data Cleaning Operations", heading1_style))
            cleaning_ops = session_data.get('cleaning_operations', [])

            if cleaning_ops:
                for i, op in enumerate(cleaning_ops, 1):
                    op_type = op.get('type', 'Unknown')
                    if op_type == 'remove_duplicates':
                        text = f"{i}. <b>Removed Duplicates:</b> {op.get('rows_removed', 0)} rows removed"
                    elif op_type == 'handle_missing':
                        method_labels = {
                            'drop': 'Dropped rows with missing values',
                            'fill_mean': 'Filled with mean value',
                            'fill_median': 'Filled with median value',
                            'fill_mode': 'Filled with most frequent value',
                            'fill_forward': 'Forward filled values',
                            'fill_backward': 'Backward filled values',
                        }
                        method = op.get('method', 'N/A')
                        text = f"{i}. <b>Missing Values in '{op.get('column')}':</b> {method_labels.get(method, method)}"
                    elif op_type == 'drop_column':
                        text = f"{i}. <b>Dropped Column:</b> '{op.get('column')}'"
                    elif op_type == 'rename_column':
                        text = f"{i}. <b>Renamed Column:</b> '{op.get('old_name')}' → '{op.get('new_name')}'"
                    else:
                        text = f"{i}. {op_type}"

                    story.append(Paragraph(text, body_style))
            else:
                story.append(Paragraph("No cleaning operations were performed.", body_style))

            story.append(PageBreak())

            # ── 3. Outlier Treatment ───────────────────────────────────────
            story.append(Paragraph("3. Outlier Treatment", heading1_style))
            outliers = session_data.get('outlier_treatment', [])

            if outliers:
                for op in outliers:
                    column = op.get('column', 'Unknown')
                    method = op.get('method', 'Unknown')
                    found = op.get('outliers_found', 0)

                    if method == 'remove':
                        text = f"• <b>Column '{column}':</b> Removed {found} outliers"
                    elif method == 'cap':
                        text = f"• <b>Column '{column}':</b> Capped {found} outliers at IQR boundaries"
                    elif method == 'transform':
                        text = f"• <b>Column '{column}':</b> Applied {op.get('transform_type', 'log')} transformation"
                    else:
                        text = f"• <b>Column '{column}':</b> Method: {method}"

                    story.append(Paragraph(text, body_style))
            else:
                story.append(Paragraph("No outlier treatment was performed.", body_style))

            story.append(PageBreak())

            # ── 4. Model Training ──────────────────────────────────────────
            story.append(Paragraph("4. Model Training Configuration", heading1_style))
            model_config = session_data.get('model_config', {})

            if not model_config or not model_config.get('task_type'):
                story.append(Paragraph("No model training was performed.", body_style))
            else:
                task_type = model_config.get('task_type', 'N/A')
                task_str = task_type.title() if isinstance(task_type, str) else str(task_type)

                test_size = model_config.get('test_size', 0.2)
                test_str = f"{test_size * 100:.0f}%" if isinstance(test_size, (int, float)) else str(test_size)
                train_str = f"{(1 - test_size) * 100:.0f}%" if isinstance(test_size, (int, float)) else 'N/A'
                tuning = 'Enabled (GridSearchCV)' if model_config.get('tune_params') else 'Disabled (Default Parameters)'

                config_text = (
                    f"<b>Training Configuration:</b><br/>"
                    f"• Task Type: {task_str}<br/>"
                    f"• Target Column: {model_config.get('target_column', 'N/A')}<br/>"
                    f"• Algorithm: {model_config.get('model_type', 'N/A')}<br/>"
                    f"• Test Size: {test_str}<br/>"
                    f"• Train Size: {train_str}<br/>"
                    f"• Hyperparameter Tuning: {tuning}"
                )
                story.append(Paragraph(config_text, body_style))

            story.append(Spacer(1, 0.3 * inch))

            # ── 5. Performance Metrics ─────────────────────────────────────
            story.append(Paragraph("5. Model Performance Metrics", heading1_style))
            report = session_data.get('performance_report', {})
            task_type = (model_config or {}).get('task_type', 'classification')

            if not report:
                story.append(Paragraph("No performance metrics available.", body_style))
            elif task_type == 'classification':
                accuracy = report.get('accuracy', 0)
                f1 = report.get('F1_Score', 0)
                cv = report.get('Cross_Validation_Score', 0)

                metrics_text = (
                    f"<b>Classification Metrics:</b><br/>"
                    f"• Accuracy: {accuracy * 100:.2f}%<br/>"
                    f"• F1 Score: {f1:.4f}<br/>"
                    f"• Cross-Validation Score: {cv:.4f}"
                )
                if 'ROC_AUC' in report:
                    metrics_text += f"<br/>• ROC AUC Score: {report['ROC_AUC']:.4f}"
                story.append(Paragraph(metrics_text, body_style))

                # Per-class table
                class_rows = [(k, v) for k, v in report.items()
                              if isinstance(v, dict) and 'precision' in v
                              and k not in ['macro avg', 'weighted avg']]
                if class_rows:
                    story.append(Spacer(1, 0.2 * inch))
                    story.append(Paragraph("Per-Class Metrics", heading2_style))
                    tbl_data = [['Class', 'Precision', 'Recall', 'F1-Score']]
                    for k, v in class_rows:
                        tbl_data.append([
                            str(k),
                            f"{v.get('precision', 0):.4f}",
                            f"{v.get('recall', 0):.4f}",
                            f"{v.get('f1-score', 0):.4f}"
                        ])
                    tbl = Table(tbl_data)
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC'))
                    ]))
                    story.append(tbl)
            else:
                r2 = report.get('R² Score', 0)
                mse = report.get('Mean Squared Error', 0)
                rmse = mse ** 0.5
                mae = report.get('Mean Absolute Error', 0)
                cv = report.get('Cross_Validation_Score', 0)

                metrics_text = (
                    f"<b>Regression Metrics:</b><br/>"
                    f"• R² Score: {r2:.4f} ({r2 * 100:.1f}% variance explained)<br/>"
                    f"• Mean Squared Error: {mse:.4f}<br/>"
                    f"• Root Mean Squared Error: {rmse:.4f}<br/>"
                    f"• Mean Absolute Error: {mae:.4f}<br/>"
                    f"• Cross-Validation Score: {cv:.4f}"
                )
                story.append(Paragraph(metrics_text, body_style))

            story.append(PageBreak())

            # ── 6. Feature Importance ──────────────────────────────────────
            story.append(Paragraph("6. Feature Importance Analysis", heading1_style))
            feature_importance = report.get('Feature_Importance', []) if report else []

            if feature_importance:
                story.append(Paragraph("Top 10 Most Important Features:", body_style))
                fi_data = [['Rank', 'Feature', 'Importance Score']]
                for i, feat in enumerate(feature_importance[:10], 1):
                    fi_data.append([
                        f"#{i}",
                        feat.get('Feature', 'Unknown'),
                        f"{feat.get('Importance', 0):.6f}"
                    ])
                fi_table = Table(fi_data)
                fi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC'))
                ]))
                story.append(fi_table)
            else:
                story.append(Paragraph("Feature importance data not available for this model.", body_style))

            story.append(PageBreak())

            # ── 7. Example Prediction ──────────────────────────────────────
            story.append(Paragraph("7. Example Prediction", heading1_style))
            prediction = session_data.get('example_prediction')

            if not prediction:
                story.append(Paragraph("No example prediction was performed.", body_style))
            else:
                story.append(Paragraph("<b>Input Values:</b>", body_style))
                inputs = prediction.get('inputs', {})
                for feature, value in list(inputs.items())[:15]:
                    story.append(Paragraph(f"• {feature}: {value}", body_style))

                story.append(Spacer(1, 0.2 * inch))
                result = prediction.get('result', {})
                pred_value = result.get('prediction', ['N/A'])[0]

                pred_style = ParagraphStyle(
                    'pred', parent=styles['Normal'],
                    fontSize=14, textColor=colors.HexColor('#16A34A'), fontName='Helvetica-Bold'
                )
                story.append(Paragraph(f"Predicted Value: {pred_value}", pred_style))

                if 'probability' in result:
                    story.append(Spacer(1, 0.2 * inch))
                    story.append(Paragraph("<b>Class Probabilities:</b>", body_style))
                    for class_name, prob in result['probability'].items():
                        story.append(Paragraph(f"• {class_name}: {prob * 100:.2f}%", body_style))

            story.append(PageBreak())

            # ── 8. AI Insights ─────────────────────────────────────────────
            story.append(Paragraph("8. AI-Generated Insights", heading1_style))
            insights = session_data.get('insights', {})

            if not insights:
                story.append(Paragraph("No insights were generated for this analysis.", body_style))
            else:
                if insights.get('key_findings'):
                    story.append(Paragraph("<b>Key Findings:</b>", body_style))
                    for finding in insights['key_findings']:
                        story.append(Paragraph(f"• {finding}", body_style))

                if insights.get('recommendations'):
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(Paragraph("<b>Recommendations:</b>", body_style))
                    for rec in insights['recommendations']:
                        story.append(Paragraph(f"• {rec}", body_style))

                if insights.get('data_quality'):
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(Paragraph("<b>Data Quality Notes:</b>", body_style))
                    for note in insights['data_quality']:
                        story.append(Paragraph(f"• {note}", body_style))

            story.append(PageBreak())

            # ── Footer page ────────────────────────────────────────────────
            footer_style = ParagraphStyle(
                'footer', parent=styles['Normal'],
                fontSize=10, textColor=colors.HexColor('#808080'), alignment=TA_CENTER
            )
            story.append(Spacer(1, 2 * inch))
            story.append(Paragraph("End of Report", footer_style))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(
                f"Generated by DataPulse ML Platform • {datetime.now().year}",
                ParagraphStyle('footer2', parent=styles['Normal'],
                               fontSize=9, textColor=colors.HexColor('#AAAAAA'), alignment=TA_CENTER)
            ))

            # Build PDF
            doc.build(story)
            logger.info(f"PDF report created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            raise
