import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import logging

logger = logging.getLogger(__name__)


class MLEngine:
    """Handles all machine learning operations"""
    
    def train_model(self, df, target_column, task_type, model_type, test_size=0.2, tune_params=False):
        """Train a machine learning model"""
        try:
            # Preprocess data
            X, y, features, label_encoder = self._preprocess_data(df, target_column, task_type)
            
            # Identify column types
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Create pipeline
            pipeline = self._create_pipeline(task_type, model_type, categorical_cols, numerical_cols, X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Tune hyperparameters if enabled
            if tune_params:
                pipeline = self._tune_hyperparameters(pipeline, X_train, y_train, task_type, model_type)
            
            # Train model
            pipeline.fit(X_train, y_train)
            logger.info(f"Model {model_type} trained for {task_type}")
            
            # Evaluate model
            report, cm, cm_fig = self._evaluate_model(pipeline, X_test, y_test, task_type, label_encoder)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=3, 
                                       scoring='f1_weighted' if task_type == 'classification' else 'r2')
            report['Cross_Validation_Score'] = float(cv_scores.mean())
            
            # Feature importance
            feature_importance = self._get_feature_importance(pipeline, features, categorical_cols)
            if feature_importance is not None:
                report['Feature_Importance'] = feature_importance.to_dict('records')[:20]  # Top 20
            
            return pipeline, report, cm, cm_fig, features, label_encoder
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise
    
    def _preprocess_data(self, df, target_column, task_type):
        """Preprocess data for training"""
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            features = df.columns.drop(target_column).tolist()
            X = df[features].copy()
            y = df[target_column]
            
            # Handle missing values
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
            
            for col in numerical_cols:
                if X[col].isnull().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            
            label_encoder = None
            if task_type == 'classification':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            return X, y, features, label_encoder
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def _create_pipeline(self, task_type, model_type, categorical_cols, numerical_cols, X):
        """Create sklearn pipeline"""
        try:
            cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', cat_transformer, categorical_cols),
                    ('num', StandardScaler(), numerical_cols)
                ])
            
            model_configs = {
                'classification': {
                    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                    'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                    'SVC': SVC(probability=True, random_state=42)
                },
                'regression': {
                    'LinearRegression': LinearRegression(),
                    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                    'XGBRegressor': XGBRegressor(random_state=42),
                    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                    'SVR': SVR()
                }
            }
            
            model = model_configs[task_type][model_type]
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Pipeline creation error: {str(e)}")
            raise
    
    def _tune_hyperparameters(self, pipeline, X_train, y_train, task_type, model_type):
        """Tune hyperparameters using GridSearchCV"""
        try:
            param_grid = {
                'LogisticRegression': {'model__C': [0.1, 1.0]},
                'RandomForestClassifier': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [5, 10]
                },
                'XGBClassifier': {
                    'model__max_depth': [3, 6],
                    'model__learning_rate': [0.1, 0.3],
                    'model__n_estimators': [50, 100]
                },
                'DecisionTreeClassifier': {'model__max_depth': [5, 10]},
                'SVC': {'model__C': [0.1, 1.0], 'model__kernel': ['rbf']},
                'LinearRegression': {},
                'RandomForestRegressor': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [5, 10]
                },
                'XGBRegressor': {
                    'model__max_depth': [3, 6],
                    'model__learning_rate': [0.1, 0.3],
                    'model__n_estimators': [50, 100]
                },
                'DecisionTreeRegressor': {'model__max_depth': [5, 10]},
                'SVR': {'model__C': [0.1, 1.0], 'model__epsilon': [0.1]}
            }
            
            if model_type in param_grid and param_grid[model_type]:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid[model_type],
                    cv=3,
                    scoring='f1_weighted' if task_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                logger.info(f"Best params: {grid_search.best_params_}")
                return grid_search.best_estimator_
            
            return pipeline
            
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {str(e)}")
            return pipeline
    
    def _evaluate_model(self, pipeline, X_test, y_test, task_type, label_encoder=None):
        """Evaluate trained model"""
        try:
            y_pred = pipeline.predict(X_test)
            
            if task_type == 'classification':
                y_test_decoded = label_encoder.inverse_transform(y_test) if label_encoder else y_test
                y_pred_decoded = label_encoder.inverse_transform(y_pred) if label_encoder else y_pred
                
                report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
                cm = confusion_matrix(y_test_decoded, y_pred_decoded)
                
                cm_fig = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title='Confusion Matrix',
                    labels=dict(x="Predicted", y="Actual")
                )
                
                if len(np.unique(y_test)) == 2:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    report['ROC_AUC'] = float(roc_auc_score(y_test, y_proba))
                
                report['F1_Score'] = float(f1_score(y_test_decoded, y_pred_decoded, average='weighted'))
                
            else:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                report = {
                    'Mean Squared Error': float(mse),
                    'Mean Absolute Error': float(mae),
                    'R² Score': float(r2)
                }
                cm = None
                cm_fig = None
            
            return report, cm, cm_fig
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise
    
    def _get_feature_importance(self, pipeline, features, categorical_cols):
        """Extract feature importance"""
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['model']
            
            feature_names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'cat':
                    cat_features = transformer.get_feature_names_out(cols)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(cols)
            
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-class: use mean of absolute coefficients
                    coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': coef
                }).sort_values(by='Importance', ascending=False)
            else:
                return None
            
            return importances
            
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {str(e)}")
            return None
    
    def predict(self, model, input_data, features, label_encoder=None, task_type='classification'):
        """Make prediction with probability if classification"""
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)
            
            # Ensure correct feature order
            input_df = input_df[features]
            
            # Predict
            prediction = model.predict(input_df)
            
            result = {}
            
            # Decode if classification
            if label_encoder:
                prediction = label_encoder.inverse_transform(prediction)
            
            result['prediction'] = prediction.tolist()
            
            # Add probabilities for classification
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                classes = label_encoder.classes_ if label_encoder else model.classes_
                result['probability'] = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
