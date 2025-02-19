import os
import sys
import re

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class CreateNewFeatures:
    def __init__(self, basic_colors=None, transmission_keywords=None, luxury_brands =None):
     
        self.basic_colors = basic_colors or [
            'Black', 'White', 'Blue', 'Gray', 'Red', 'Silver', 'Gold', 'Yellow', 'Beige', 
            'Green', 'Brown', 'Purple', 'Orange', 'Ebony'
        ]
        
        self.transmission_keywords = transmission_keywords or {
            'Automatic': ['automatic', 'a/t', 'at'],
            'Manual': ['manual', 'm/t', 'mt'],
            'Dual Clutch': ['dct'],
            'CVT': ['cvt', 'variable'],
        }

        self.luxury_brands = luxury_brands or [
            'Lamborghini', 'Rolls-Royce', 'Bentley', 'Bugatti', 'Ferrari', 'McLaren', 'Aston'
        ]

    # Function to extract color from a string
    def extract_color(self, color_name):
        if isinstance(color_name, str):
            for basic_color in self.basic_colors:
                if basic_color.lower() in color_name.lower():
                    return basic_color
            return 'Other'  # Return 'Other' if no match found
        return np.nan  # Return NaN if not a string

    # Function to identify wrong transmitions and substitute by NaN
    def simplify_transmission(self, transmission):
        if pd.isna(transmission) or transmission in ['F', '2', 'SCHEDULED FOR OR IN PRODUCTION', '–']:
            return np.nan  # Replace non-standard values with NaN
        
        transmission = str(transmission).lower()
        
        # Loop through transmission keywords and match
        for transmission_type, keywords in self.transmission_keywords.items():
            if any(keyword in transmission for keyword in keywords):
                return transmission_type  # Return matching transmission type
        
        return transmission  # If no match, return the original value

    # Function to preprocess color columns (external & internal)
    def preprocess_colors(self, df):
        df['ext_col'] = df['ext_col'].apply(self.extract_color)
        df['int_col'] = df['int_col'].apply(self.extract_color)
        return df

    # Function to preprocess transmission column
    def preprocess_transmission(self, df):
        df['transmission'] = df['transmission'].apply(self.simplify_transmission)
        return df
    
    def extract_engine_features(self,engine_str):
        """Extract hp, cylinders, and liters from engine description."""
        if pd.isna(engine_str):
            return np.nan, np.nan,np.nan

        engine_str = str(engine_str)

        # Extract Horsepower (e.g., "500.0HP" → 500.0)
        hp_match = re.search(r'(\d+\.?\d*)HP', engine_str)
        hp = float(hp_match.group(1)) if hp_match else np.nan

        # Extract Liters (e.g., "5.0L" → 5.0)
        liters_match = re.search(r'(\d+\.\d*)L', engine_str)
        liters = float(liters_match.group(1)) if liters_match else np.nan

        # Extract Cylinders (e.g., "V8" → 8, "12 Cylinder" → 12)
        pattern = r'(V|I)(\d+)|(\d+)\sCylinder'
        cyl_match = re.search(pattern, engine_str, re.IGNORECASE)
        cylinders = int(cyl_match.group(2) or cyl_match.group(3)) if cyl_match else np.nan

        return hp, cylinders,liters

    # Main function to preprocess data
    def add_features(self, df):
        try:

            logging.info("Creating new features")

            # Step 1: Replace invalid values with NaN
            df.replace({'–': np.nan, 'not supported': np.nan}, inplace=True)
            
            # Step 2: Drop unnecessary columns
            columns_to_drop = ['id', 'clean_title']
            df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
            
            # Step 3: Process color columns (external & internal)
            df = self.preprocess_colors(df)
            
            # Step 4: Simplify transmission column
            df = self.preprocess_transmission(df)
            
            # Step 5: Extract base model (just the first word)
            df['model'] = df['model'].str.split().str[0]

            # Apply the function to create new features
            df[['hp','cylinders','liters']] = df['engine'].apply(lambda x: pd.Series(self.extract_engine_features(x)))

            df['is_luxury'] = df['brand'].isin(self.luxury_brands).map({True: 'Yes', False: 'No'})
        
            return df
        
        except Exception as e:
            raise CustomException(e,sys)


class FuelHandler:
    """Class to handle fuel type simplification"""
    
    def __init__(self):

        self.fuel_keywords = {
            'Gasoline': ['gasoline'],
            'E85 Flex Fuel': ['flex', 'e85'],
            'Plug-In Hybrid': ['plug', 'plug-in'],
            'Hybrid': ['hybrid'],
            'Diesel': ['diesel'],
            'Electric': ['electric', 'motor', 'battery']
        }

    def impute_fuel_type(self, engine_str):
        """Impute the fuel type based on engine description.

        Args:
            engine_str (str): The engine description.

        Returns:
            str or np.nan: The corresponding fuel type or np.nan if no match is found.
        """
        if pd.isna(engine_str):
            return np.nan
        
        engine_str = engine_str.lower()

        # Check for each fuel type keyword in the engine description
        for fuel_type, keywords in self.fuel_keywords.items():
            if any(keyword in engine_str for keyword in keywords):
                return fuel_type

        return np.nan  # Return NaN if no match is found

    def transform_fuel_type(self, df):

        logging.info("Managing Fuel type")

        # Impute missing fuel types based on the engine description
        df['fuel_type'] = df['fuel_type'].fillna(df['engine'].apply(self.impute_fuel_type))
       
        return df


class FeatureTransformation:
    """Class that orchestrates all transformations."""
    
    def __init__(self, fuel_handler=None, feature_creator=None):
        self.fuel_handler = fuel_handler or FuelHandler()  
        self.feature_creator = feature_creator or CreateNewFeatures()  

    def transform_data(self, df):
        """Apply all transformations to the data."""

        try:
            #df=pd.read_csv(path)
        
            # Step 1: Add new features
            df = self.feature_creator.add_features(df)

            # Step 2: Apply fuel transformation
            df = self.fuel_handler.transform_fuel_type(df)

            # Eliminate skewness on milage, reducing outliers effect
            df['milage'] = np.sqrt(df['milage'])

            logging.info("Feature Transformation is successful")

            return df

        except Exception as e:
            raise CustomException(e,sys)
