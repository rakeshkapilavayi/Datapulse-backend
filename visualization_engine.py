import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Generates all visualizations for EDA"""
    
    def create_visualizations(self, df, viz_type='all'):
        """Create visualizations based on type"""
        try:
            visualizations = {}
            
            if viz_type in ['all', 'distributions']:
                visualizations['distributions'] = self._create_distributions(df)
            
            if viz_type in ['all', 'correlations']:
                visualizations['correlations'] = self._create_correlations(df)
            
            if viz_type in ['all', 'outliers']:
                visualizations['outliers'] = self._create_outlier_plots(df)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization creation error: {str(e)}")
            raise
    
    def _create_distributions(self, df):
        """Create distribution plots - ONLY histograms for numerical, ONLY bar charts for categorical"""
        try:
            distributions = {
                'numerical': [],
                'categorical': []
            }
            
            # Numerical distributions - ONLY for numeric columns
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in num_cols:
                fig = self._create_numerical_distribution(df, col)
                if fig:
                    distributions['numerical'].append({
                        'column': col,
                        'figure': fig.to_json()
                    })
            
            # Categorical distributions - ONLY for categorical/object columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                # Only create bar chart if it has a reasonable number of unique values
                if df[col].nunique() <= 50:  # Limit to prevent overcrowded charts
                    fig = self._create_categorical_distribution(df, col)
                    if fig:
                        distributions['categorical'].append({
                            'column': col,
                            'figure': fig.to_json()
                        })
            
            return distributions
            
        except Exception as e:
            logger.error(f"Distribution creation error: {str(e)}")
            return {'numerical': [], 'categorical': []}
    
    def _create_numerical_distribution(self, df, col):
        """Create histogram ONLY for numerical columns"""
        try:
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name='Distribution',
                    marker_color='#636EFA',
                    opacity=0.7,
                    nbinsx=30
                )
            )
            
            # Calculate statistics for display
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            # Add vertical lines for mean and median with separate positioning
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"Mean: {mean_val:.2f}", 
                annotation_position="top"
            )
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="green", 
                annotation_text=f"Median: {median_val:.2f}", 
                annotation_position="bottom"
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                showlegend=False,
                title_text=f"📊 Distribution of {col}",
                title_font_size=16,
                xaxis_title=col,
                yaxis_title="Frequency",
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Numerical distribution error for {col}: {str(e)}")
            return None
    
    def _create_categorical_distribution(self, df, col):
        """Create bar chart ONLY for categorical columns"""
        try:
            # Get value counts (returns a Series)
            value_counts = df[col].value_counts()
            
            # Limit to top 20
            if len(value_counts) > 20:
                value_counts = value_counts.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            # Calculate percentages
            total = len(df[col])
            percentages = (value_counts / total * 100).round(2)
            
            # Convert index to list of strings for better display
            categories = [str(x) for x in value_counts.index.tolist()]
            counts = value_counts.values.tolist()
            pcts = percentages.values.tolist()
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=categories,
                y=counts,
                text=[f"{count} ({pct}%)" for count, pct in zip(counts, pcts)],
                textposition='auto',
                marker=dict(
                    color='#AB63FA',
                    line=dict(color='#8B4FD9', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ))
            
            # Update layout with explicit settings
            fig.update_layout(
                title=f"📊 Categorical Distribution: {col}{title_suffix}",
                xaxis=dict(
                    title=col,
                    type='category'
                ),
                yaxis=dict(
                    title="Count"
                ),
                height=500,
                showlegend=False,
                title_font_size=16,
                template='plotly_white',
                bargap=0.2
            )
            
            # Rotate x-axis labels if many categories
            if len(value_counts) > 10:
                fig.update_xaxes(tickangle=-45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Categorical distribution error for {col}: {str(e)}")
            return None
    
    def _create_correlations(self, df):
        """Create correlation visualizations"""
        try:
            correlations = {
                'heatmap': None,
                'scatter': None
            }
            
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(num_cols) >= 2:
                # Correlation heatmap
                corr_matrix = df[num_cols].corr()
                
                heatmap = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Correlation Heatmap',
                    labels=dict(color="Correlation")
                )
                heatmap.update_layout(height=600, width=800, template='plotly_white')
                correlations['heatmap'] = heatmap.to_json()
                
                # Scatter plot for highest correlation
                corr_matrix_upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                max_corr_idx = corr_matrix_upper.abs().stack().idxmax()
                col1, col2 = max_corr_idx
                corr_value = corr_matrix.loc[col1, col2]
                
                # Sample if too large
                max_rows = 5000
                sample_df = df[[col1, col2]].sample(n=min(max_rows, len(df)), random_state=42)
                
                scatter = px.scatter(
                    sample_df,
                    x=col1,
                    y=col2,
                    title=f'Correlation: {col1} vs {col2} (r = {corr_value:.2f})',
                    trendline="ols",
                    opacity=0.6,
                    color_discrete_sequence=['#EF553B']
                )
                scatter.update_layout(height=500, template='plotly_white')
                correlations['scatter'] = scatter.to_json()
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation visualization error: {str(e)}")
            return {'heatmap': None, 'scatter': None}
    
    def _create_outlier_plots(self, df):
        """Create box plots for outlier detection"""
        try:
            outlier_plots = []
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in num_cols:
                fig = px.box(
                    df,
                    y=col,
                    title=f'Box Plot: {col}',
                    color_discrete_sequence=['#00CC96']
                )
                fig.update_layout(height=400, template='plotly_white')
                
                outlier_plots.append({
                    'column': col,
                    'figure': fig.to_json()
                })
            
            return outlier_plots
            
        except Exception as e:
            logger.error(f"Outlier plot creation error: {str(e)}")
            return []
    
    def create_custom_chart(self, df, chart_config):
        """Create custom chart based on user configuration"""
        try:
            chart_type = chart_config.get('type')
            x_axis = chart_config.get('xAxis')
            y_axis = chart_config.get('yAxis')
            color_by = chart_config.get('colorBy')
            
            # Clean up empty strings to None
            if not x_axis or x_axis == '':
                x_axis = None
            if not y_axis or y_axis == '':
                y_axis = None
            if not color_by or color_by == '':
                color_by = None
            
            logger.info(f"Creating custom chart - Type: {chart_type}, X: {x_axis}, Y: {y_axis}, Color: {color_by}")
            
            fig = None
            
            if chart_type == 'pie' and x_axis and y_axis:
                # Pie chart: x_axis = labels (categorical), y_axis = values (numerical)
                fig = px.pie(df, names=x_axis, values=y_axis,
                            title=f"Pie Chart: {y_axis} by {x_axis}")
                fig.update_traces(textposition='inside', textinfo='percent+label')
            
            elif chart_type == 'bar':
                if x_axis and y_axis:
                    # Bar chart with both axes
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_by,
                                title=f"Bar Chart: {x_axis} vs {y_axis}")
                elif x_axis:
                    # Bar chart with just X (count)
                    fig = px.histogram(df, x=x_axis, color=color_by,
                                      title=f"Bar Chart: {x_axis}")
            
            elif chart_type == 'scatter' and x_axis and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                                title=f"Scatter Plot: {x_axis} vs {y_axis}")
            
            elif chart_type == 'line' and x_axis and y_axis:
                # Sort by x_axis for proper line chart
                df_sorted = df.sort_values(by=x_axis)
                fig = px.line(df_sorted, x=x_axis, y=y_axis, color=color_by,
                             title=f"Line Chart: {x_axis} vs {y_axis}")
            
            elif chart_type == 'box' and y_axis:
                fig = px.box(df, y=y_axis, x=x_axis, color=color_by,
                            title=f"Box Plot: {y_axis}" + (f" by {x_axis}" if x_axis else ""))
            
            elif chart_type == 'violin' and y_axis:
                fig = px.violin(df, y=y_axis, x=x_axis, color=color_by,
                               title=f"Violin Plot: {y_axis}" + (f" by {x_axis}" if x_axis else ""))
            
            elif chart_type == 'histogram' and x_axis:
                fig = px.histogram(df, x=x_axis, color=color_by,
                                  title=f"Histogram: {x_axis}")
            
            if fig:
                fig.update_layout(template='plotly_white', height=500)
                logger.info(f"Custom chart created successfully: {chart_type}")
                return fig.to_json()
            else:
                logger.warning(f"Could not create chart with config: {chart_config}")
                return None
            
        except Exception as e:
            logger.error(f"Custom chart creation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise