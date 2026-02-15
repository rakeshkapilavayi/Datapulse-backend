"""
Updated Insights Generator using the new google.genai package
"""
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
from scipy import stats
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini API from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Initialize client
gemini_client = None
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI insights will not be available.")
else:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI configured successfully with new SDK")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {str(e)}")


class InsightsGenerator:
    """Generates insights and recommendations from data"""
    
    def generate_statistical_insights(self, df):
        """Generate traditional statistical insights"""
        try:
            insights = []
            recommendations = []
            
            # Statistical summary
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                stats_summary = df[numeric_cols].describe()
                for col in numeric_cols:
                    mean = stats_summary.loc['mean', col]
                    std = stats_summary.loc['std', col]
                    insights.append(
                        f"The average value of {col} is {mean:.2f} with a standard deviation of {std:.2f}."
                    )
                    if std > mean * 0.5:
                        insights.append(
                            f"{col} shows high variability (std > 50% of mean), indicating potential outliers or diverse data points."
                        )
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                            insights.append(
                                f"Strong correlation ({corr_matrix.iloc[i, j]:.2f}) between {col1} and {col2}."
                            )
                            recommendations.append(
                                f"Consider investigating the relationship between {col1} and {col2} for potential multicollinearity."
                            )
            
            # Categorical insights
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                unique_count = df[col].nunique()
                if unique_count < 10:
                    top_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if top_value:
                        top_freq = df[col].value_counts().iloc[0]
                        insights.append(
                            f"{col} has {unique_count} unique categories, with '{top_value}' being the most frequent "
                            f"({top_freq} times, {top_freq/len(df)*100:.1f}%)."
                        )
            
            # Outlier detection
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    insights.append(f"{col} has {outliers} potential outliers (Z-score > 3).")
                    recommendations.append(f"Review and handle outliers in {col} to improve model performance.")
            
            # Data quality
            missing_total = df.isnull().sum().sum()
            if missing_total > 0:
                insights.append(
                    f"The dataset contains {missing_total} missing values across "
                    f"{len(df.columns[df.isnull().any()])} columns."
                )
                recommendations.append("Impute or drop missing values to ensure data completeness.")
            
            if not insights:
                insights.append("The dataset appears to be well-balanced with no immediate anomalies.")
            if not recommendations:
                recommendations.append("No specific recommendations at this time; consider further EDA or feature engineering.")
            
            return insights, recommendations
            
        except Exception as e:
            logger.error(f"Statistical insights error: {str(e)}")
            return ["Error generating insights"], ["Please check your data"]
    
    def generate_enhanced_insights(self, df, summary, cleaning_report=None, ml_report=None):
        """Generate enhanced insights using LLM - UPDATED to use new google.genai package"""
        try:
            # Get traditional insights first
            insights, recommendations = self.generate_statistical_insights(df)
            
            # Check if Gemini is configured
            if not gemini_client:
                logger.warning("Gemini API client not available. Returning traditional insights.")
                return self._format_fallback_insights(insights, recommendations, summary)
            
            # Build summary - FIXED: Handle missing summary data
            dataset_summary = f"""
Dataset Shape: {summary.get('shape', {}).get('rows', 0)} rows, {summary.get('shape', {}).get('columns', 0)} columns
Missing Values: {summary.get('missing_total', 0)} total
Duplicates: {summary.get('duplicates', 0)}
Memory Usage: {summary.get('memory_usage', 0):.2f} MB
"""
            
            # Add cleaning info
            cleaning_summary = ""
            if cleaning_report:
                cleaning_summary = f"""
Cleaning Applied:
- Handled missing values in {len(cleaning_report.get('missing_handled', {}))} columns
- Removed {cleaning_report.get('duplicates_removed', 0)} duplicates
- Capped outliers in {len(cleaning_report.get('outliers_capped', {}))} columns
"""
            
            # Add ML info
            ml_summary = ""
            if ml_report:
                if isinstance(ml_report, dict):
                    if 'accuracy' in str(ml_report).lower() or 'f1' in str(ml_report).lower():
                        ml_summary = "\nMachine Learning: Classification model trained"
                        if 'Cross_Validation_Score' in ml_report:
                            ml_summary += f" (CV Score: {ml_report['Cross_Validation_Score']:.2f})"
                    elif 'R² Score' in ml_report:
                        ml_summary = f"\nMachine Learning: Regression model trained (R² Score: {ml_report['R² Score']:.2f})"
            
            # Combine insights
            traditional_insights = "\n".join([f"- {insight}" for insight in insights])
            traditional_recommendations = "\n".join([f"- {rec}" for rec in recommendations])
            
            # Create prompt for LLM
            prompt = f"""You are a data analysis expert. Transform the following analysis into a comprehensive, professional report.

{dataset_summary}
{cleaning_summary}
{ml_summary}

CURRENT INSIGHTS:
{traditional_insights}

CURRENT RECOMMENDATIONS:
{traditional_recommendations}

YOUR TASK:
Create a well-structured, professional analysis report with these sections:

## 📊 Dataset Overview
Summarize the key characteristics and data quality.

## 🔍 Key Findings
Expand on the insights with deeper explanations and implications.

## 💡 Statistical Patterns
Explain the statistical patterns discovered and what they mean.

## ⚠️ Data Quality Assessment
Discuss data quality, cleaning performed, and any remaining concerns.

## 🎯 Strategic Recommendations
Provide specific, actionable next steps for:
- Further analysis
- Feature engineering
- Model improvement (if applicable)
- Data collection improvements

## 📈 Business Impact
Explain what these findings mean in practical terms and potential business value.

Requirements:
- Be specific with numbers and percentages
- Make insights actionable and clear
- Use professional but accessible language
- Focus on practical implications
- Keep it concise but comprehensive (aim for 500-800 words)
- Use emojis for visual appeal
- Do NOT mention that you are an AI or that this was generated by AI
"""
            
            # Generate content using new API
            logger.info("Generating enhanced insights with Gemini (new SDK)...")
            
            try:
                # Use the new API format
                response = gemini_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=2048,
                    )
                )
                
                # Extract text from response
                enhanced_insights = response.text
                
                if not enhanced_insights or enhanced_insights.strip() == '':
                    raise ValueError("Empty response from Gemini API")
                
                logger.info("Enhanced insights generated successfully")
                return enhanced_insights
                
            except Exception as api_error:
                logger.error(f"Gemini API error: {str(api_error)}")
                logger.error(f"Error type: {type(api_error).__name__}")
                # Fallback to traditional insights
                return self._format_fallback_insights(insights, recommendations, summary)
            
        except Exception as e:
            logger.error(f"Enhanced insights error: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            # Fallback to traditional insights
            insights, recommendations = self.generate_statistical_insights(df)
            return self._format_fallback_insights(insights, recommendations, summary)
    
    def generate_quick_summary(self, df, summary):
        """Generate quick summary - UPDATED to use new google.genai package"""
        try:
            # Check if Gemini is available
            if not gemini_client:
                logger.warning("Gemini API client not available. Using traditional summary.")
                insights, recommendations = self.generate_statistical_insights(df)
                return self._format_quick_fallback(insights, recommendations, summary)
            
            insights, recommendations = self.generate_statistical_insights(df)
            
            # Create a simpler prompt for quick summary
            prompt = f"""You are a data analyst. Create a brief, clear summary of this dataset analysis.

Dataset: {summary.get('shape', {}).get('rows', 0)} rows × {summary.get('shape', {}).get('columns', 0)} columns
Missing Values: {summary.get('missing_total', 0)}
Duplicates: {summary.get('duplicates', 0)}

Key Insights:
{chr(10).join([f'- {insight}' for insight in insights[:5]])}

Create a brief summary (200-300 words) covering:
1. Dataset overview
2. Top 3 most important findings
3. Top 3 actionable recommendations

Use clear, concise language with emojis. Do not mention you are an AI.
"""
            
            try:
                # Use the new API format
                response = gemini_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
                
                quick_summary = response.text
                
                if not quick_summary or quick_summary.strip() == '':
                    raise ValueError("Empty response from Gemini API")
                
                return quick_summary
                
            except Exception as api_error:
                logger.error(f"Gemini API error in quick summary: {str(api_error)}")
                # Fallback
                return self._format_quick_fallback(insights, recommendations, summary)
            
        except Exception as e:
            logger.error(f"Quick summary error: {str(e)}")
            insights, recommendations = self.generate_statistical_insights(df)
            return self._format_quick_fallback(insights, recommendations, summary)
    
    def _format_fallback_insights(self, insights, recommendations, summary):
        """Format fallback insights when LLM fails"""
        return f"""## 📊 Dataset Overview

**Shape:** {summary.get('shape', {}).get('rows', 0)} rows × {summary.get('shape', {}).get('columns', 0)} columns
**Missing Values:** {summary.get('missing_total', 0)}
**Duplicates:** {summary.get('duplicates', 0)}

## 🔍 Key Findings

{chr(10).join([f'- {insight}' for insight in insights])}

## 🎯 Recommendations

{chr(10).join([f'- {rec}' for rec in recommendations])}

---
*Note: Enhanced AI analysis is temporarily unavailable. Showing traditional statistical insights.*
"""
    
    def _format_quick_fallback(self, insights, recommendations, summary):
        """Format fallback for quick summary"""
        return f"""## 📊 Dataset Summary

**Size:** {summary.get('shape', {}).get('rows', 0)} rows × {summary.get('shape', {}).get('columns', 0)} columns
**Missing Values:** {summary.get('missing_total', 0)} total
**Duplicate Rows:** {summary.get('duplicates', 0)}
**Memory Usage:** {summary.get('memory_usage', 0):.2f} MB

## 🔍 Key Insights

{chr(10).join([f'{i+1}. {insight}' for i, insight in enumerate(insights[:5])])}

## 🎯 Top Recommendations

{chr(10).join([f'{i+1}. {rec}' for i, rec in enumerate(recommendations[:3])])}
"""
    
    def generate_structured_insights(self, df, summary):
        """Generate structured raw insights for frontend display"""
        try:
            insights = []
            recommendations = []
            
            # Calculate statistics
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Dataset statistics
            missing_total = df.isnull().sum().sum()
            missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] > 0 and df.shape[1] > 0 else 0
            duplicates = df.duplicated().sum()
            duplicate_pct = (duplicates / len(df)) * 100 if len(df) > 0 else 0
            quality_score = 100 - missing_pct - (duplicate_pct * 0.5)
            quality_label = 'Excellent' if quality_score > 90 else 'Good' if quality_score > 70 else 'Fair' if quality_score > 50 else 'Poor'
            
            statistics = {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'missing_total': int(missing_total),
                'missing_pct': float(missing_pct),
                'duplicates': int(duplicates),
                'duplicate_pct': float(duplicate_pct),
                'quality_score': float(quality_score),
                'quality_label': quality_label
            }
            
            # Numerical column insights
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                
                insights.append({
                    'text': f"Column '{col}' has an average value of {mean:.2f} with a standard deviation of {std:.2f}.",
                    'type': 'distribution',
                    'importance': 'medium'
                })
                
                # High variability detection
                if std > mean * 0.5:
                    insights.append({
                        'text': f"'{col}' shows high variability (std > 50% of mean), suggesting diverse data points or potential outliers.",
                        'type': 'distribution',
                        'importance': 'high'
                    })
                    recommendations.append({
                        'text': f"Investigate the distribution of '{col}' for potential outliers or data quality issues.",
                        'priority': 'high',
                        'category': 'quality'
                    })
                
                # Outlier detection
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    outliers_pct = (outliers / len(df)) * 100
                    insights.append({
                        'text': f"'{col}' contains {outliers} potential outliers ({outliers_pct:.1f}% of data) with Z-score > 3.",
                        'type': 'outlier',
                        'importance': 'high' if outliers_pct > 5 else 'medium'
                    })
                    recommendations.append({
                        'text': f"Consider capping or removing outliers in '{col}' before modeling.",
                        'priority': 'high' if outliers_pct > 5 else 'medium',
                        'category': 'cleaning'
                    })
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, corr_val))
                            
                            insights.append({
                                'text': f"Strong correlation ({corr_val:.2f}) detected between '{col1}' and '{col2}'.",
                                'type': 'correlation',
                                'importance': 'high' if abs(corr_val) > 0.85 else 'medium'
                            })
                
                if high_corr_pairs:
                    recommendations.append({
                        'text': f"Investigate multicollinearity between highly correlated features. Consider feature selection or PCA.",
                        'priority': 'medium',
                        'category': 'modeling'
                    })
            
            # Categorical insights
            for col in cat_cols:
                unique_count = df[col].nunique()
                
                if unique_count < 10:
                    top_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if top_value:
                        top_freq = df[col].value_counts().iloc[0]
                        top_pct = (top_freq / len(df)) * 100
                        
                        insights.append({
                            'text': f"'{col}' has {unique_count} unique categories. Most frequent: '{top_value}' ({top_pct:.1f}% of data).",
                            'type': 'distribution',
                            'importance': 'low'
                        })
                        
                        if top_pct > 70:
                            insights.append({
                                'text': f"'{col}' is highly imbalanced with '{top_value}' dominating {top_pct:.1f}% of the data.",
                                'type': 'quality',
                                'importance': 'high'
                            })
                            recommendations.append({
                                'text': f"Address class imbalance in '{col}' using resampling techniques or class weights.",
                                'priority': 'high',
                                'category': 'modeling'
                            })
                elif unique_count > 50:
                    insights.append({
                        'text': f"'{col}' has high cardinality ({unique_count} unique values). Consider encoding strategy.",
                        'type': 'quality',
                        'importance': 'medium'
                    })
                    recommendations.append({
                        'text': f"Use appropriate encoding for high-cardinality feature '{col}' (e.g., target encoding, frequency encoding).",
                        'priority': 'medium',
                        'category': 'modeling'
                    })
            
            # Data quality insights
            if missing_total > 0:
                missing_cols = df.columns[df.isnull().any()].tolist()
                insights.append({
                    'text': f"Dataset contains {missing_total} missing values across {len(missing_cols)} columns ({missing_pct:.1f}% of total data).",
                    'type': 'quality',
                    'importance': 'critical' if missing_pct > 10 else 'high' if missing_pct > 5 else 'medium'
                })
                recommendations.append({
                    'text': f"Handle missing values using appropriate imputation or removal strategies.",
                    'priority': 'critical' if missing_pct > 10 else 'high',
                    'category': 'cleaning'
                })
            
            if duplicates > 0:
                insights.append({
                    'text': f"Found {duplicates} duplicate rows ({duplicate_pct:.1f}% of dataset).",
                    'type': 'quality',
                    'importance': 'high' if duplicate_pct > 5 else 'medium'
                })
                recommendations.append({
                    'text': f"Remove duplicate rows to avoid data leakage and improve model performance.",
                    'priority': 'high' if duplicate_pct > 5 else 'medium',
                    'category': 'cleaning'
                })
            
            # General recommendations
            if not insights:
                insights.append({
                    'text': "Dataset appears to be well-structured with no immediate anomalies detected.",
                    'type': 'quality',
                    'importance': 'low'
                })
            
            if not recommendations:
                recommendations.append({
                    'text': "Data quality is good. Consider feature engineering and exploratory analysis for deeper insights.",
                    'priority': 'low',
                    'category': 'analysis'
                })
            
            return {
                'insights': insights,
                'recommendations': recommendations,
                'statistics': statistics
            }
            
        except Exception as e:
            logger.error(f"Structured insights error: {str(e)}")
            raise