# Cars Price Prediction

This project uses machine learning techniques to predict the price of used cars based on various features such as engine specifications, mileage, brand, and more. The goal is to develop a well-optimized model that generalizes well to unseen data, ensuring accurate and reliable price estimates. 

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Environment Setup](#environment-setup)

---
## Project Overview
This project predicts the price of used cars based on the following factors:
- **Categorical Features**: 'brand', 'model', 'accident', 'int_col', 'is_luxury'
- **Numerical Features**: 'hp', 'cylinders', 'milage', 'model_year'

I have implemented a pipeline to handle data preprocessing, feature engineering, model training, and evaluation. The model is optimized using techniques like cross-validation and hyperparameter tuning. 

For detailed references, go to the [notebook](https://github.com/DarianRodriguez/Used-Car-Prices/tree/main/notebook) where the dataset is analyzed thoroughly.


## Features

The dataset contains the following features:
- **id**: Unique identifier for each vehicle.
- **brand**: Manufacturer or brand name of the vehicle (e.g., Toyota, Ford).
- **model**: Specific model of the vehicle (e.g., Corolla, Mustang).
- **model_year**: Year of manufacture.
- **milage**: Total miles driven by the vehicle.
- **fuel_type**: Type of fuel (gasoline, diesel, electric).
- **engine**: Engine specifications (hp, cylinders, liter).
- **transmission**: Type of transmission (manual, automatic).
- **ext_col**: Exterior color.
- **int_col**: Interior color.
- **accident**: Whether the vehicle has been in any accidents (yes/no).
- **clean_title**: Whether the vehicle has a clean title (yes/no).
- **price**: Sale price of the vehicle.

## Data Cleaning and Feature Engineering

### 1. **Handling Missing Values**:
- **Not MCAR (Missing Completely at Random)**: I analyzed the missing data and found that many features were **Not Missing Completely at Random (MCAR)**. For those, we used **group-based imputation**:

   - Missing values `hp`, `cylinders`, and `milage` were imputed using the **mean** of similar groups based on features:`brand`,`model`,`model_year`, and the **mode** for categorical features.
   - Features like `fuel_type` had missing values for **electric** vehicles, which is added after performing an **Exploratory Data Analysis (EDA)**.

  ### 2. **Handling High Cardinality Categorical Features**:
  - **Grouping Model Names**: Some vehicle models had detailed names, e.g., `Corolla iM Base`, which were reduced to just `Corolla`. I extracted the **first word** of the model name to reduce cardinality, reducing the model feature from **1897** unique values to **517**.
   - **Colors**: The `ext_col` and `int_col` (exterior color) feature had many unique values with slight variations. To reduce cardinality:
     - Colors like `Platinum White Pearl` were categorized as **White**, and `Garnet Red Metallic` became **Red** (extract the base color).
   - **Transmission Variations**: The `transmission` feature contained many variations that referred to the same information. To make this more uniform and reduce complexity, we grouped them into the following categories:
     - **Automatic**
     - **Manual**
     - **CVT** (Continuously Variable Transmission)
     - **Dual Clutch**

### 3. **Feature Creation**:
   - The `engine` column was split into individual features:
     - `hp`: Horsepower
     - `cylinders`: Number of cylinders
     - `liter`: Engine displacement

- **Luxury Brands Feature**:
  - A new feature, `is_luxury`, was created to capture whether a vehicle belongs to one of the top luxury car brands. The list of luxury brands includes:
    - 'Lamborghini', 'Rolls-Royce', 'Bentley', 'Bugatti', 'Ferrari', 'McLaren', 'Aston'.
  

### 4. **Outlier Handling**:
   - The `milage` feature exhibited **right skewness** with **outliers**, so I applied a **square root (sqrt) transformation** to reduce the impact of extreme values and normalize the distribution. This transformation helps bring down the influence of large mileage values and improves the model's performance by stabilizing variance.

### 5. **Feature Selection**:
  - Utilized **Cramér's V**, **Pearson correlation**, and **Correlation Ratio** to assess relationships between categorical and numerical features.
   - Removed features with low importance or high correlation to improve model performance.
   - Retained features with strong predictive power for the target variable.

### 6. **Model Training**:
   - The baseline model used was **Linear Regression**.
   - For model optimization, **Bayesian Search** and **K-Fold Cross Validation** were applied to tune hyperparameters.
   - The best-performing model was selected based on cross-validation performance on **RMSE**.
   - Models tested include:
     - **HistGradientBoostingRegressor**
     - **ElasticNet**
     - **RandomForestRegressor**

## Evaluation

- The residual plots were generated to visually assess the model's performance. For more details and to view the plots, please refer to the [**`figures`**](https://github.com/DarianRodriguez/Used-Car-Prices/tree/main/artifacts/figures) folder, located in the **`artifacts`** directory.  The best model was **HistGradientBoostingRegressor**.

The model performs well for **lower to mid-range vehicle prices** but struggles with **luxury cars**. For future work a good idea is creating a two-stage model: One model for regular cars and another specialized for luxury vehicles.

### **Key Insights from Residual Plots**
- **Accurate for Most Cars**:  
  - Residuals are **tightly clustered around zero**, meaning the model predicts many prices well.
  - The **high peak near zero** confirms this.

- **Luxury Cars Are Underpredicted**:  
  - A **long tail of positive residuals** (500K - 3M) suggests the model **significantly underestimates** some car prices.
  - Likely affects **luxury and rare models**.


## Environment Setup

To set up the environment for this project, use the provided `environment.yml` file to create a conda environment.

```bash
conda env create -f environment.yml