from rf_stacked_model_pipeline import ProductionPipelineRF
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Pipeline_Test") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Define test dataset path
test_dataset_path = "../Cleaning & Preparation/Train Test Data/test_data.parquet"  # Replace with the path to your test dataset

# Define output path
output_data_path = "./pipeline_test_output.parquet"

# Define pipeline configuration
feature_names = [
    'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
    'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
    'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
    'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
    'T-MON-CKP', 'T-PDG', 'T-TPT'
]

feature_names_with_missingness = feature_names + [f"{feature}_missing" for feature in feature_names]
layer_2_feature_names = feature_names_with_missingness + [f"probability_label_{label}" for label in range(1, 10)]

pipeline_config = {
    "scaler_model_path": "../Cleaning & Preparation/scaler_model",  # Path to scaler model
    "model_dir": "./models",  # Directory containing models
    "labels_to_process": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Labels for Layer 1
    "feature_names": feature_names,
    "feature_names_with_missingness": feature_names_with_missingness,
    "features_continuous": [
        'ABER-CKGL', 'ABER-CKP', 'P-ANULAR', 'P-JUS-CKGL', 'P-MON-CKP',
        'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
        'T-MON-CKP', 'T-PDG', 'T-TPT'
    ],
    "features_categorical": [
        'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO', 'ESTADO-M2',
        'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1', 'ESTADO-W2',
        'ESTADO-XO'
    ],
    "layer_2_feature_names": layer_2_feature_names,  # Features for Layer 2
    "output_data_path": output_data_path
}

# Load the dataset
try:
    spark_df = spark.read.parquet(test_dataset_path)
    print("Test dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    spark.stop()
    exit(1)

# Initialize the pipeline
pipeline = ProductionPipelineRF(pipeline_config)

# Run the pipeline
try:
    print("Running pipeline...")
    final_output = pipeline.run(test_dataset_path)
    print("Pipeline executed successfully.")

    # Show the first few rows of the output
    print("Final output preview:")
    final_output.show(10)

    # Save the output
    print(f"Pipeline output saved to {output_data_path}")
except Exception as e:
    print(f"Error running pipeline: {e}")
finally:
    spark.stop()
