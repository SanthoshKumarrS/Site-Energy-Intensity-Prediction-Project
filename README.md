# Site-Energy-Intensity-Prediction-Project
I've uploaded a project to estimate Energy Intensity to assess a site's EUI (Energy Usage Intensity)  in this repository.

### **Description:** 
According to a report issued by the International Energy Agency (IEA), the lifecycle of buildings from construction to demolition was responsible for 37% of global energy-related and process-related CO2 emissions in 2020. Yet it is possible to drastically reduce the energy consumption of buildings by a combination of easy-to-implement fixes and state-of-the-art strategies. 

The dataset consists of building characteristics, weather data for the location of the building, as well as the energy usage for the building, and the given year, measured as Site Energy Usage Intensity (Site EUI). Each row in the data corresponds to a single building observed in a given year.

### Source of dataset & data dictionary - [Click Here](https://www.kaggle.com/c/widsdatathon2022/data)

### **Problem Statement:** 
You are provided with two datasets: (1) the train\_dataset where the observed values of the Site EUI for each row are provided and (2) the x\_test dataset the observed values of the Site EUI for each row are removed and provided separately in y\_test. Your task is to predict the Site EUI for each row (using the complete training dataset), given the characteristics of the building and the weather data for the location of the building. Use the test sets for validation and testing. The target variable  is `site_eui` for the predictive analytics problem.

**Evaluation Metric:** Root Mean Squared Error (RMSE)

## Web application:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://siteeuiprediction.onrender.com)


## Tasks and techniques used:

**1. Data Understanding:**
- EDA to visualize numerical features relations with target variable
- Calculated Variance Inflation factors(VIF) using statsmdodels library
- Used scatteplot for bivariate analysis

**Results:**
- 6 Columns are having missing values
- Energy intensity means usage of electricity and heat (The more the use, more the intensity of energy and CO2 emmissions per building)
- Bulding characteristics and energy usage over a period of year is given (Every Month's Min, Max and Mean)
- Samples are from various of states of USA
- We need to check relations between site conditions and usage of energy

**2. Data Preparation:**
- Handled missing values using KNN imputer
- Feature engineering to aggregate the existing columns to reduce dimentionality of dataset
- Calculated VIFs using statamodels library

**3. Modeling and evaluation**
- Built base machine learning models using Linear Regression,Carboost,lasso,XGBoost and RandomForest Regressors
- Baseline RandomForest Regressor had better score
- Evaluation metric - Root Mean Squared Error (RMSE)
- Hyperparameter tuning using gridserchCV

**Results:**
- Baseline score on test dataset using RandomForest Regressor model was 40.43
- Tuned & Final model's `RMSE score is 40.6` 

**4. Explainable AI using SHAP**
- Used Shaply values to learn interaction of inputs with the outputs
- Plotted features shap values charts

**Results:**
- From summary plots it was found `energy_star_rating` and `floor_area` feature were the most imporatant to predict model output.

----------------------

**Acknowledgement:** ***TMLC Academy***

**References:**

1. [KNN Imputer](https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
