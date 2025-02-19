import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object, evaluate_model
from src.components.data_transformation import FeatureTransformation

class PredictPipeline:
    def __init__(self):
        self.model_path=os.path.join("artifacts","model.pkl")


    def predict(self,X):
        try:

            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame([X])

            # Load the trained model
            model=load_object(file_path=self.model_path)

            feature_transformer = FeatureTransformation()
            X_modified = feature_transformer.transform_data(X)

            preds = model.predict(X_modified)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    file_path = os.path.join('artifacts',"test.csv")
    test_data = pd.read_csv(file_path)

    X_test = test_data.drop(columns=['price'])
    y_test = test_data['price']

    predict_pipeline = PredictPipeline()
    y_pred = predict_pipeline.predict(X_test)

    evaluate_model(y_test,y_pred)


