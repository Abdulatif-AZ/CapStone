# Capstone Project: Offshore Well Event Classification - Group 4

## Introduction

This project focuses on identifying and classifying undesirable events in offshore wells based on time-series data provided by [Petrobras](https://github.com/petrobras/3W). Using the 3W dataset, which includes real, simulated, and hand-drawn instances, we developed a model that classifies 9 undersirable events in offshore free flowing wells. 

Multiple modeling attempts were made, including:

- **[`Model_LSTM_RF_LR_V1.ipynb`](./ML/Alternative%20Solutions/Model_LSTM_RF_LR_V1.ipynb)**: A stacked ensemble of LSTM, Random Forest, and Logistic Regression classifying events 0, 1, 5, and 7.
- **[`Model_XGBoost_V1.ipynb`](./ML/Alternative%20Solutions/Model_XGBoost_V1.ipynb)**: An XGBoost-based classifier classifying events 0, 1, 5, and 7.
- **[`3W_PySpark_MLlib_Stacked_Model.ipynb`](./ML/Final%20Solution/3W_PySpark_MLlib_Stacked_Model.ipynb)**: The final stacked ensemble model classifying all 9 undesirable events and normal operations.

---

## Repository Structure

### 1. **Cleaning & Preparation**
- **[`Cleaning_Approach_1.ipynb`](./Cleaning%20&%20Preparation/Cleaning_Approach_1.ipynb)**: First cleaning method, focusing on filling missing values and defining classes.
- **[`Cleaning_Approach_2.ipynb`](./Cleaning%20&%20Preparation/Cleaning_Approach_2.ipynb)**: Alternative comprehensive cleaning approach using PySpark for scalability.
- **[`scaler_model`](./Cleaning%20&%20Preparation/scaler_model)**: The scaler model used to scale the data.

### 2. **Configurations**
- **[`3w_env.yml`](./Configurations/3w_env.yml)** and **[`pyspark_env.yml`](./Configurations/pyspark_env.yml)**: Environment configuration files for Python and PySpark.
- **[`configuration_setup.ipynb`](./Configurations/configuration_setup.ipynb)**: Instructions to set up the working environment.

### 3. **Data**
- **[`3W Original`](./Data/3W%20Original)**: Will contain the original data once `3W_Data_Extraction.ipynb` is executed.
- **[`Cleaning & Preparation`](./Data/Cleaning%20&%20Preparation)**: Will contain cleaned data once the `Cleaning_Approach_1.ipynb` and `Cleaning_Approach_2.ipynb` are executed.

### 4. **Data Extraction**
- **[`3W_Data_Extraction.ipynb`](./Data%20Extraction/3W_Data_Extraction.ipynb)**: Extracts and combines the parquet files provided by Petrobras.

### 5. **EDA (Exploratory Data Analysis)**
- **[`3W_EDA.ipynb`](./EDA/3W_EDA.ipynb)**: Exploratory analysis on the dataset to identify trends and anomalies.
- **[`3W_Real.ipynb`](./EDA/3W_Real.ipynb)**: Uses PySpark to explore the real instances available in the 3W dataset.
- **[`Data Overview.ipynb`](./EDA/Data%20Overview.ipynb)**: High-level summary of the data.
- **[`Events.ipynb`](./EDA/Events.ipynb)**: Dives deeper into the undesirable events data.

### 6. **ML (Machine Learning)**
- **Alternative Solutions**:
  - **[`Model_LSTM_RF_LR_V1.ipynb`](./ML/Alternative%20Solutions/Model_LSTM_RF_LR_V1.ipynb)**: Stacked LSTM with Random Forest and Logistic Regression for the events 0, 1, 5, 7.
  - **[`Model_XGBoost_V1.ipynb`](./ML/Alternative%20Solutions/Model_XGBoost_V1.ipynb)**: XGBoost algorithm to classify the events 0, 1, 5, 7.
- **Final Solution**:
  - **[`3W_PySpark_MLlib_Stacked_Model.ipynb`](./ML/Final%20Solution/3W_PySpark_MLlib_Stacked_Model.ipynb)**: Stacked ensemble model.
  - **[`rf_stacked_model_pipeline.py`](./ML/Final%20Solution/rf_stacked_model_pipeline.py)**: **********************
  - **[`rf_stacked_model_streamlit.py`](./ML/Final%20Solution/rf_stacked_model_streamlit.py)**: *********************
  - **[`test_pipeline.py`](./ML/Final%20Solution/test_pipeline.py)**: ************************

### 7. **Other Directories**
- **[`logs`](./ML/logs)**: Contains execution and training logs.
- **[`mlruns`](./ML/mlruns)**: Tracks experiments and model metrics using MLflow.
- **[`models`](./ML/models)**: Stores saved models for deployment.

---

## How to Replicate the Final Solution

1. **Set up the environment**:
   - Follow the steps detailed in [`configuration_setup.ipynb`](./Configurations/configuration_setup.ipynb) to install dependencies using [`3w_env.yml`](./Configurations/3w_env.yml) or [`pyspark_env.yml`](./Configurations/pyspark_env.yml).

2. **Extract the Data**:
   - Run [`3W_Data_Extraction.ipynb`](./Data%20Extraction/3W_Data_Extraction.ipynb) to extract the data from 3W.

3. **Data Processing**:
   - Use [`Cleaning_Approach_2.ipynb`](./Cleaning%20&%20Preparation/Cleaning_Approach_2.ipynb) to clean and preprocess the data.

4. **EDA**:
   - Use notebooks [`3W_EDA.ipynb`](./EDA/3W_EDA.ipynb) and [`3W_Real.ipynb`](./EDA/3W_Real.ipynb) for analysis and understanding of the dataset.

5. **Model Training**:
   - Train the final model using [`3W_PySpark_MLlib_Stacked_Model.ipynb`](./ML/Final%20Solution/3W_PySpark_MLlib_Stacked_Model.ipynb) under the `Final Solution` folder.


