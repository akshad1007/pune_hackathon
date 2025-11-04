"""
Enhanced Model Training Pipeline
===============================

Comprehensive ML pipeline for India Environmental Monitoring System.
Trains multiple models with hyperparameter optimization and evaluation.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import structlog

logger = structlog.get_logger()


class ModelTrainer:
    """Comprehensive model training and evaluation pipeline."""
    
    def __init__(self, output_path: str = "ml/models/artifacts"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.evaluation_results = {}
        
        logger.info("ModelTrainer initialized", output_path=str(self.output_path))
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train all models and return comprehensive results.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set proportion
            random_state: Random seed
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting comprehensive model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info("Data split completed",
                   train_samples=len(X_train),
                   test_samples=len(X_test),
                   features=X_train.shape[1])
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard_scaler'] = scaler
        
        # Train baseline models
        baseline_results = self.train_baseline_models(X_train, X_test, y_train, y_test)
        
        # Train tree-based models
        tree_results = self.train_tree_models(X_train, X_test, y_train, y_test)
        
        # Train boosting models
        boosting_results = self.train_boosting_models(X_train, X_test, y_train, y_test)
        
        # Train neural network models (if available)
        if TENSORFLOW_AVAILABLE and len(X_train) > 100:
            nn_results = self.train_neural_networks(X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            nn_results = {}
            logger.warning("Neural networks skipped - TensorFlow not available or insufficient data")
        
        # Ensemble models
        ensemble_results = self.create_ensemble_models(X_test, y_test)
        
        # Combine all results
        all_results = {
            'baseline': baseline_results,
            'tree_based': tree_results,
            'boosting': boosting_results,
            'neural_networks': nn_results,
            'ensemble': ensemble_results
        }
        
        # Find best model
        best_model_info = self.find_best_model(all_results)
        all_results['best_model'] = best_model_info
        
        # Save models and results
        self.save_models_and_results(all_results)
        
        logger.info("Model training completed",
                   models_trained=len(self.models),
                   best_model=best_model_info['name'],
                   best_mae=best_model_info['mae'])
        
        return all_results
    
    def train_baseline_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train baseline models for comparison."""
        logger.info("Training baseline models...")
        
        results = {}
        
        # 1. Persistence model (use mean as prediction)
        mean_prediction = np.full_like(y_test, y_train.mean())
        results['persistence'] = {
            'mae': mean_absolute_error(y_test, mean_prediction),
            'rmse': np.sqrt(mean_squared_error(y_test, mean_prediction)),
            'r2': r2_score(y_test, mean_prediction),
            'predictions': mean_prediction.tolist()
        }
        
        # 2. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        
        self.models['linear_regression'] = lr
        results['linear_regression'] = {
            'mae': mean_absolute_error(y_test, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'r2': r2_score(y_test, lr_pred),
            'predictions': lr_pred.tolist()
        }
        
        # 3. Ridge Regression with hyperparameter tuning
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge()
        ridge_grid = GridSearchCV(ridge, ridge_params, cv=3, scoring='neg_mean_absolute_error')
        ridge_grid.fit(X_train, y_train)
        ridge_pred = ridge_grid.predict(X_test)
        
        self.models['ridge_regression'] = ridge_grid.best_estimator_
        results['ridge_regression'] = {
            'mae': mean_absolute_error(y_test, ridge_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
            'r2': r2_score(y_test, ridge_pred),
            'best_params': ridge_grid.best_params_,
            'predictions': ridge_pred.tolist()
        }
        
        logger.info("Baseline models trained", models=list(results.keys()))
        return results
    
    def train_tree_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train tree-based models."""
        logger.info("Training tree-based models...")
        
        results = {}
        
        # Random Forest with hyperparameter tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        rf_pred = rf_grid.predict(X_test)
        
        self.models['random_forest'] = rf_grid.best_estimator_
        results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred),
            'best_params': rf_grid.best_params_,
            'feature_importance': dict(zip(X_train.columns, rf_grid.best_estimator_.feature_importances_)),
            'predictions': rf_pred.tolist()
        }
        
        logger.info("Tree-based models trained", models=list(results.keys()))
        return results
    
    def train_boosting_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train gradient boosting models."""
        logger.info("Training boosting models...")
        
        results = {}
        
        # XGBoost
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        xgb_pred = xgb_grid.predict(X_test)
        
        self.models['xgboost'] = xgb_grid.best_estimator_
        results['xgboost'] = {
            'mae': mean_absolute_error(y_test, xgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'r2': r2_score(y_test, xgb_pred),
            'best_params': xgb_grid.best_params_,
            'feature_importance': dict(zip(X_train.columns, xgb_grid.best_estimator_.feature_importances_)),
            'predictions': xgb_pred.tolist()
        }
        
        # LightGBM
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        lgb_grid.fit(X_train, y_train)
        lgb_pred = lgb_grid.predict(X_test)
        
        self.models['lightgbm'] = lgb_grid.best_estimator_
        results['lightgbm'] = {
            'mae': mean_absolute_error(y_test, lgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
            'r2': r2_score(y_test, lgb_pred),
            'best_params': lgb_grid.best_params_,
            'feature_importance': dict(zip(X_train.columns, lgb_grid.best_estimator_.feature_importances_)),
            'predictions': lgb_pred.tolist()
        }
        
        # Gradient Boosting
        gb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb_model, gb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        gb_pred = gb_grid.predict(X_test)
        
        self.models['gradient_boosting'] = gb_grid.best_estimator_
        results['gradient_boosting'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'r2': r2_score(y_test, gb_pred),
            'best_params': gb_grid.best_params_,
            'feature_importance': dict(zip(X_train.columns, gb_grid.best_estimator_.feature_importances_)),
            'predictions': gb_pred.tolist()
        }
        
        logger.info("Boosting models trained", models=list(results.keys()))
        return results
    
    def train_neural_networks(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train neural network models."""
        logger.info("Training neural network models...")
        
        results = {}
        
        try:
            # Simple Dense Neural Network
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            nn_pred = model.predict(X_test, verbose=0).flatten()
            
            self.models['neural_network'] = model
            results['neural_network'] = {
                'mae': mean_absolute_error(y_test, nn_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, nn_pred)),
                'r2': r2_score(y_test, nn_pred),
                'training_history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history['mae'],
                    'val_mae': history.history['val_mae']
                },
                'predictions': nn_pred.tolist()
            }
            
            logger.info("Neural network trained successfully")
            
        except Exception as e:
            logger.error("Neural network training failed", error=str(e))
            results['neural_network'] = {'error': str(e)}
        
        return results
    
    def create_ensemble_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Create ensemble models from trained base models."""
        logger.info("Creating ensemble models...")
        
        results = {}
        
        # Get predictions from all models
        model_predictions = {}
        for name, model in self.models.items():
            if name != 'neural_network':  # Skip neural network for now
                try:
                    pred = model.predict(X_test)
                    model_predictions[name] = pred
                except Exception as e:
                    logger.warning(f"Could not get predictions from {name}", error=str(e))
        
        if len(model_predictions) >= 2:
            # Simple averaging ensemble
            pred_array = np.array(list(model_predictions.values()))
            ensemble_pred = np.mean(pred_array, axis=0)
            
            results['simple_average'] = {
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'r2': r2_score(y_test, ensemble_pred),
                'component_models': list(model_predictions.keys()),
                'predictions': ensemble_pred.tolist()
            }
            
            # Weighted ensemble (weights based on inverse MAE)
            weights = []
            for name in model_predictions.keys():
                try:
                    pred = model_predictions[name]
                    mae = mean_absolute_error(y_test, pred)
                    weights.append(1.0 / (mae + 1e-8))  # Inverse MAE as weight
                except:
                    weights.append(0.1)  # Small weight for failed models
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            weighted_pred = np.average(pred_array, axis=0, weights=weights)
            
            results['weighted_average'] = {
                'mae': mean_absolute_error(y_test, weighted_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, weighted_pred)),
                'r2': r2_score(y_test, weighted_pred),
                'component_models': list(model_predictions.keys()),
                'weights': weights.tolist(),
                'predictions': weighted_pred.tolist()
            }
            
            logger.info("Ensemble models created", ensembles=list(results.keys()))
        
        return results
    
    def find_best_model(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing model across all categories."""
        best_model = {'name': None, 'mae': float('inf'), 'category': None}
        
        for category, models in all_results.items():
            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'mae' in metrics:
                    if metrics['mae'] < best_model['mae']:
                        best_model = {
                            'name': model_name,
                            'mae': metrics['mae'],
                            'rmse': metrics.get('rmse', 0),
                            'r2': metrics.get('r2', 0),
                            'category': category
                        }
        
        logger.info("Best model identified", **best_model)
        return best_model
    
    def save_models_and_results(self, results: Dict[str, Any]):
        """Save trained models and results to disk."""
        logger.info("Saving models and results...")
        
        # Save models
        models_file = self.output_path / "trained_models.pkl"
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'metadata': {
                'training_time': datetime.now().isoformat(),
                'model_count': len(self.models)
            }
        }, models_file)
        
        # Save results
        results_file = self.output_path / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save feature importance summary
        feature_importance_summary = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                for sub_model, metrics in model_results.items():
                    if isinstance(metrics, dict) and 'feature_importance' in metrics:
                        feature_importance_summary[f"{model_name}_{sub_model}"] = metrics['feature_importance']
        
        if feature_importance_summary:
            importance_file = self.output_path / "feature_importance.json"
            with open(importance_file, 'w') as f:
                json.dump(feature_importance_summary, f, indent=2)
        
        # Save model performance summary
        performance_summary = self.create_performance_summary(results)
        summary_file = self.output_path / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        logger.info("Models and results saved",
                   models_file=str(models_file),
                   results_file=str(results_file))
    
    def create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of model performance."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_rankings': [],
            'category_best': {},
            'overall_best': results.get('best_model', {}),
            'performance_metrics': {}
        }
        
        # Collect all model performances
        all_models = []
        for category, models in results.items():
            if category == 'best_model':
                continue
            
            category_best = {'name': None, 'mae': float('inf')}
            
            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'mae' in metrics:
                    model_info = {
                        'name': f"{category}_{model_name}",
                        'category': category,
                        'mae': metrics['mae'],
                        'rmse': metrics.get('rmse', 0),
                        'r2': metrics.get('r2', 0)
                    }
                    all_models.append(model_info)
                    
                    if metrics['mae'] < category_best['mae']:
                        category_best = {
                            'name': model_name,
                            'mae': metrics['mae'],
                            'rmse': metrics.get('rmse', 0),
                            'r2': metrics.get('r2', 0)
                        }
            
            if category_best['name']:
                summary['category_best'][category] = category_best
        
        # Sort models by performance
        all_models.sort(key=lambda x: x['mae'])
        summary['model_rankings'] = all_models
        
        # Calculate performance statistics
        if all_models:
            maes = [m['mae'] for m in all_models]
            summary['performance_metrics'] = {
                'best_mae': min(maes),
                'worst_mae': max(maes),
                'mean_mae': np.mean(maes),
                'std_mae': np.std(maes),
                'models_count': len(all_models)
            }
        
        return summary


def run_comprehensive_training(df: pd.DataFrame, target_col: str = 'aqi',
                             output_path: str = "ml/models/artifacts") -> Dict[str, Any]:
    """
    Run comprehensive model training pipeline.
    
    Args:
        df: Dataframe with features and target
        target_col: Target column name
        output_path: Output directory for saving results
    
    Returns:
        Training results dictionary
    """
    logger.info("Starting comprehensive model training pipeline...")
    
    # Prepare data
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in data")
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Remove non-numeric columns for now
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    logger.info("Data prepared for training",
               samples=len(X),
               features=len(X.columns),
               target_range=(y.min(), y.max()))
    
    # Initialize trainer
    trainer = ModelTrainer(output_path)
    
    # Train all models
    results = trainer.train_all_models(X, y)
    
    logger.info("Comprehensive training completed",
               models_trained=len(trainer.models),
               best_model=results['best_model']['name'])
    
    return results


if __name__ == "__main__":
    # Demo run
    from data_processing import load_and_prepare_data
    from feature_engineering import create_ml_ready_features
    
    # Load and prepare data
    datasets, integrated = load_and_prepare_data()
    
    # Add synthetic AQI target for demo
    if 'aqi' not in integrated.columns:
        np.random.seed(42)
        integrated['aqi'] = np.random.normal(120, 40, len(integrated))
        integrated['aqi'] = np.clip(integrated['aqi'], 0, 500)
    
    # Create features
    featured_df, engineer = create_ml_ready_features(integrated, 'aqi')
    
    # Train models
    results = run_comprehensive_training(featured_df, 'aqi')
    
    print(f"\nModel Training Results:")
    print(f"Best Model: {results['best_model']['name']}")
    print(f"Best MAE: {results['best_model']['mae']:.2f}")
    print(f"Best R²: {results['best_model']['r2']:.3f}")
    
    print(f"\nTop 5 Models:")
    if 'model_rankings' in results.get('performance_summary', {}):
        for i, model in enumerate(results['performance_summary']['model_rankings'][:5]):
            print(f"{i+1}. {model['name']}: MAE={model['mae']:.2f}, R²={model['r2']:.3f}")
