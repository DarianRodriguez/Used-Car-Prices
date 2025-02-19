import os
import sys
from dataclasses import dataclass, field

import optuna
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object
from src.exception import CustomException

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

    models: dict = field(default_factory=lambda: {
        # "ElasticNet": ElasticNet(),
        # "Decision Tree": DecisionTreeRegressor(),
        #"Random Forest": RandomForestRegressor(),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(),
    })

    params: dict = field(default_factory=lambda: {
        "ElasticNet": {
            'model__alpha': Real(1e-3, 100, prior='log-uniform'),
            'model__l1_ratio': Real(0, 1),
        },

        "Decision Tree": {
            'model__max_depth': Integer(3, 15),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 10),
        },

        "Random Forest": {
            'model__n_estimators': Integer(50, 500),
            'model__max_depth': Integer(5, 50),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(5, 10),
        },

        "HistGradientBoosting Regressor": {
            'model__max_iter': Integer(100, 1000),
            'model__learning_rate': Real(1e-4, 1.0, prior='log-uniform'),
            'model__max_depth': Integer(3, 15),  
            'model__min_samples_leaf': Integer(5, 50),  
            'model__max_bins': Integer(50, 250), 
        }
    })

class ModelTrainer:
    def __init__(self, pipeline, n_splits=5):
        self.model_trainer_config = ModelTrainerConfig()
        self.preprocessor = pipeline
        self.cv = KFold(n_splits=n_splits, shuffle=True)

    def optimize_model(self, model, param_space, X_train, y_train, n_trials=20):
        pipeline = self.get_full_pipeline(model)
        opt = BayesSearchCV(
            pipeline,
            param_space,
            n_iter=n_trials,
            cv=self.cv,
            n_jobs=-1,
            random_state=42,
            scoring='neg_root_mean_squared_error',
            verbose=0
        )
        opt.fit(X_train, y_train)
        return opt.best_estimator_, opt.best_score_

    def select_best_model(self, X_train, y_train, n_trials=20):
        try:
            best_model = None
            best_score = float('inf')
            best_model_name = None

            for model_name, model in self.model_trainer_config.models.items():
                if model_name not in self.model_trainer_config.params:
                    continue
                print(f"Optimizing {model_name}...")
                best_estimator, best_score_cv = self.optimize_model(
                    model, self.model_trainer_config.params[model_name], X_train, y_train, n_trials
                )
                print(f"Best Score (RMSE) for {model_name}: {-best_score_cv:.4f}")
                if -best_score_cv < best_score:
                    best_score = -best_score_cv
                    best_model = best_estimator
                    best_model_name = model_name

            print(f"\nThe best model is: {best_model_name} with cross-validation RMSE: {best_score:.4f}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model, best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_full_pipeline(self, model):
        return Pipeline([
            ('preprocessing', self.preprocessor),
            ('model', model)
        ])
