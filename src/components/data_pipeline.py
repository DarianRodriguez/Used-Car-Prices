import os
from dataclasses import dataclass, field

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils import GroupBasedImputer
from src.exception import CustomException
from src.logger import logging

@dataclass(frozen=True)
class PipelineConfig:
    num_features_impute: list = field(default_factory=lambda: ['hp', 'cylinders', 'liters'])
    cat_features_impute: list = field(default_factory=lambda: ['fuel_type', 'transmission', 'ext_col', 'int_col'])
    num_features: list = field(default_factory=lambda: ['hp', 'cylinders', 'milage', 'model_year'])
    cat_features: list = field(default_factory=lambda: ['brand', 'model', 'accident', 'int_col', 'is_luxury'])
    group_cols: list = field(default_factory=lambda: ['brand', 'model', 'model_year'])


class PipelineBuilder:
    def __init__(self):
        self.config = PipelineConfig()

    def get_full_pipeline(self):

        num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])


        full_pipeline = Pipeline([
            ("group_imputer_num", GroupBasedImputer(self.config.group_cols, self.config.num_features_impute, strategy="median")),
            ("group_imputer_cat", GroupBasedImputer(self.config.group_cols, self.config.cat_features_impute, strategy="mode")),
            ("column_transformer", ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, self.config.num_features),
                    ("cat", cat_pipeline, self.config.cat_features),
                ],
                remainder='drop'
            ))
        ])
        return full_pipeline
    
   
