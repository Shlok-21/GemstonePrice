# Gemstone Price Prediction

## Overview
This repository contains code for building a machine learning model to predict gemstone prices based on various features such as carat, cut, color, clarity, depth, table, and dimensions (x, y, z). The primary goal is to develop an accurate model that can assist in estimating gemstone prices based on their characteristics.

## Dataset
The dataset utilized for this project comprises the following columns:
- Carat (weight of the gemstone)
- Cut (quality of the cut)
- Color (color grade of the gemstone)
- Clarity (clarity grade of the gemstone)
- Depth (total depth percentage)
- Table (width of top of the gem relative to the widest point)
- Dimensions (x, y, z)
- Price (target variable)

## Machine Learning Models Explored
Various machine learning algorithms were explored to develop the predictive model. The models investigated include:
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression
- K-Nearest Neighbors Regression
- Gradient Boosting Regression
- AdaBoost Regression
- CatBoost Regression
- XGBoost Regression

### Best Performers in ipynb
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Hence The hyperparameter tuning process was conducted on the three models, and subsequently, the best performing model was selected in the model_trainer.py

## Model Evaluation and Selection
To identify the best performing models, GridSearchCV was used to tune hyperparameters and optimize model performance. Evaluation metrics R2 score was utilized to assess predictive accuracy and generalization capabilities.

## Final Model Selection
Following thorough evaluation, the most effective models were selected based on their predictive accuracy and performance metrics. These models were potentially combined or further fine-tuned to create the best final predictive model for estimating gemstone prices based on the given features.

## Summary 
This project entails developing a comprehensive data processing and modeling pipeline utilizing Python and Flask for local deployment. Key tasks involve conducting thorough EDA, feature engineering, model training, website creation. 
The objective was to analyze industry-standard project structures and coding practices, drawing from the typical spaghetti code written in Jupyter notebooks.

## Project Structure
1. **Setup GitHub Repository and Local Environment**
    * Create GitHub repository and .gitignore file
    * Establish a virtual environment
    * Create `setup.py` for package management
    * Define dependencies in `requirements.txt` 

2. **Define Source Code Structure**
    * Organize code into `src` directory and define package (`requirements.txt`)
        * Implement component files: `data_ingestion.py`, `data_transformation.py`, `model_trainer.py`
        * Develop pipeline files: `predict_pipeline.py`, `train_pipeline.py`
        * Establish exception handling, logging, and utility files: `exceptions.py`, `logger.py`, `utils.py`

3. **Conduct Exploratory Data Analysis (EDA) in Jupyter Notebook**
    * Perform EDA tasks
    * Handle missing values
    * Remove duplicate entries
    * Perform data cleaning and imputation
    * Engage in feature engineering
    * Conduct train-test split
    * Identify top-performing models
    * Evaluate models using metrics such as R2 Score

4. **Create User Interface for Input**

5. **Write Modular Code based on Jupyter Notebook and Test on Local Server (Flask)**
