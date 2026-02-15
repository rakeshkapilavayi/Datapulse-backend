import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data processing operations"""
    
    def get_summary(self, df):
        """Generate comprehensive dataset summary"""
        try:
            summary = {
                'shape': {
                    'rows': int(df.shape[0]),
                    'columns': int(df.shape[1])
                },
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
                'missing_total': int(df.isnull().sum().sum()),
                'duplicates': int(df.duplicated().sum()),
                'memory_usage': float(df.memory_usage(deep=True).sum() / (1024 ** 2)),  # MB
                'column_info': []
            }
            
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'unique': int(df[col].nunique()),
                    'missing_pct': float(df[col].isnull().sum() / len(df) * 100)
                }
                
                if df[col].dtype in ['float64', 'int64']:
                    col_info.update({
                        'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                        'median': float(df[col].median()) if not df[col].isnull().all() else None,
                        'std': float(df[col].std()) if not df[col].isnull().all() else None,
                        'min': float(df[col].min()) if not df[col].isnull().all() else None,
                        'q25': float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,  # 25th percentile
                        'q75': float(df[col].quantile(0.75)) if not df[col].isnull().all() else None,  # 75th percentile
                        'max': float(df[col].max()) if not df[col].isnull().all() else None
                    })
                elif df[col].dtype == 'object':
                    if df[col].nunique() < 50:
                        value_counts = df[col].value_counts().head(10).to_dict()
                        col_info['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
                
                summary['column_info'].append(col_info)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            raise
    
    def manual_cleaning(self, df, missing_actions, remove_duplicates):
        """Apply manual cleaning based on user specifications"""
        try:
            cleaned_df = df.copy()
            
            for col, action in missing_actions.items():
                if col not in cleaned_df.columns:
                    continue
                
                if action == "drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                elif action == "mean" and cleaned_df[col].dtype in ['float64', 'int64']:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                elif action == "median" and cleaned_df[col].dtype in ['float64', 'int64']:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                elif action == "mode":
                    if not cleaned_df[col].mode().empty:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                elif action == "forward_fill":
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                elif action == "backward_fill":
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
            
            if remove_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()
            
            logger.info(f"Manual cleaning applied. Shape: {cleaned_df.shape}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Manual cleaning error: {str(e)}")
            raise
    
    def auto_clean(self, df):
        """Perform automated cleaning"""
        try:
            cleaned_df = df.copy()
            report = {
                'missing_handled': {},
                'duplicates_removed': 0,
                'outliers_capped': {},
                'rows_before': len(df),
                'rows_after': 0
            }
            
            # Handle missing values
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() > 0:
                    if cleaned_df[col].dtype in ['float64', 'int64']:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        report['missing_handled'][col] = 'Mean'
                    else:
                        if not cleaned_df[col].mode().empty:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                            report['missing_handled'][col] = 'Mode'
            
            # Remove duplicates
            duplicates = cleaned_df.duplicated().sum()
            if duplicates > 0:
                cleaned_df = cleaned_df.drop_duplicates()
                report['duplicates_removed'] = int(duplicates)
            
            # Cap outliers using IQR
            numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].any().any():
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                    report['outliers_capped'][col] = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            
            report['rows_after'] = len(cleaned_df)
            
            logger.info(f"Auto cleaning completed. Report: {report}")
            return cleaned_df, report
            
        except Exception as e:
            logger.error(f"Auto cleaning error: {str(e)}")
            raise
    
    def get_all_outliers(self, df):
        """Get outlier information for all numeric columns"""
        try:
            outliers_info = {}
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outliers_info[col] = {
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(len(outliers)),
                    'outlier_percentage': float(len(outliers) / len(df) * 100),
                    'total_rows': int(len(df))
                }
            
            return outliers_info
            
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            raise
    
    def treat_outliers(self, df, column, method='cap'):
        """Treat outliers in a specific column"""
        try:
            df_treated = df.copy()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
            
            if method == 'cap':
                df_treated[column] = df_treated[column].clip(lower_bound, upper_bound)
                treatment_desc = f"Capped to [{lower_bound:.2f}, {upper_bound:.2f}]"
            elif method == 'remove':
                df_treated = df_treated[(df_treated[column] >= lower_bound) & (df_treated[column] <= upper_bound)]
                treatment_desc = f"Removed {len(df) - len(df_treated)} rows"
            elif method == 'log':
                if (df[column] > 0).all():
                    df_treated[column] = np.log1p(df_treated[column])
                    treatment_desc = "Applied log transformation"
                else:
                    raise ValueError("Log transformation requires all positive values")
            
            outliers_after = len(df_treated[(df_treated[column] < lower_bound) | (df_treated[column] > upper_bound)])
            
            report = {
                'column': column,
                'method': method,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outliers_before': int(outliers_before),
                'outliers_after': int(outliers_after),
                'rows_before': int(len(df)),
                'rows_after': int(len(df_treated)),
                'treatment_desc': treatment_desc
            }
            
            logger.info(f"Outliers treated in {column} using {method}")
            return df_treated, report
            
        except Exception as e:
            logger.error(f"Outlier treatment error: {str(e)}")
            raise
    
    def treat_all_outliers(self, df, method='cap', exclude_cols=None):
        """Treat outliers in all numeric columns"""
        try:
            df_treated = df.copy()
            reports = []
            exclude_cols = exclude_cols or []
            
            numeric_cols = df_treated.select_dtypes(include=['float64', 'int64']).columns
            cols_to_treat = [col for col in numeric_cols if col not in exclude_cols]
            
            for col in cols_to_treat:
                try:
                    df_treated, report = self.treat_outliers(df_treated, col, method)
                    reports.append(report)
                except Exception as e:
                    logger.warning(f"Could not treat {col}: {str(e)}")
                    continue
            
            summary_report = {
                'method': method,
                'columns_treated': len(reports),
                'total_outliers_before': sum(r['outliers_before'] for r in reports),
                'total_outliers_after': sum(r['outliers_after'] for r in reports),
                'rows_before': int(len(df)),
                'rows_after': int(len(df_treated)),
                'detailed_reports': reports
            }
            
            logger.info(f"All outliers treated. Columns: {len(reports)}")
            return df_treated, summary_report
            
        except Exception as e:
            logger.error(f"Treat all outliers error: {str(e)}")
            raise
    
    
    def execute_code(self, code, df, other_df=None):
        """Safely execute user code with optional secondary dataset"""
        try:
            import sys
            from io import StringIO
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Create safe execution environment
            safe_globals = {
                "__builtins__": {
                    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                    "range": range, "list": list, "dict": dict, "str": str,
                    "int": int, "float": float, "bool": bool, "set": set,
                    "tuple": tuple, "sorted": sorted, "enumerate": enumerate,
                    "zip": zip, "map": map, "filter": filter, "round": round,
                    "type": type, "isinstance": isinstance, "print": print
                }
            }
            
            safe_locals = {
                "df": df.copy(),
                "pd": pd,
                "np": np,
                "other_df": other_df.copy() if other_df is not None else None
            }
            
            # Execute code
            exec(code, safe_globals, safe_locals)
            
            # Get output
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Get result
            result = None
            try:
                lines = code.strip().split('\n')
                if lines:
                    last_line = lines[-1].strip()
                    if last_line and not last_line.startswith(('import', 'from', 'def', 'class')):
                        result = eval(last_line, safe_globals, safe_locals)
            except:
                pass
            
            response = {
                'success': True,
                'output': output,
                'result_type': type(result).__name__ if result is not None else None
            }
            
            if isinstance(result, pd.DataFrame):
                response['result'] = result.head(100).to_dict('records')
                response['result_shape'] = result.shape
            elif isinstance(result, pd.Series):
                response['result'] = result.head(100).to_dict()
            elif result is not None:
                response['result'] = str(result)
            
            return response
            
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }