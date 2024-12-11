import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, monotonically_increasing_id, mean, count
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array
import mlflow
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionPipelineRF:
    def __init__(self, config):
        """
        Initialize Spark session and configure paths.
        """
        self.spark = SparkSession.builder \
            .appName("ProductionPipeline_RF") \
            .config("spark.driver.memory", "14g") \
            .config("spark.executor.memory", "8g") \
            .getOrCreate()

        self.config = config
        self.scaler_model = StandardScalerModel.load(config['scaler_model_path'])

    def load_data(self, input_data_path):
        """
        Load the base dataset from Parquet and add row indexing.
        """
        df = self.spark.read.parquet(input_data_path)
        return df.withColumn('row_index', monotonically_increasing_id())

    def create_missingness_columns(self, df):
        """
        Adds missingness indicator columns for each feature in the dataset.
        """
        for feature in self.config['feature_names']:
            df = df.withColumn(
                f"{feature}_missing", when(col(feature).isNull(), lit(1)).otherwise(lit(0))
            )
        return df

    def impute_features(self, df):
        """
        Impute missing values for continuous and categorical features.
        """
        for feature in self.config['features_continuous']:
            mean_value = df.select(mean(col(feature))).first()[0]
            df = df.withColumn(feature, when(col(feature).isNull(), lit(mean_value)).otherwise(col(feature)))

        for feature in self.config['features_categorical']:
            mode_row = (
                df.filter(col(feature).isNotNull())
                  .groupBy(feature)
                  .agg(count("*").alias("freq"))
                  .orderBy(col("freq").desc())
                  .first()
            )
            if mode_row:
                mode_value = mode_row[0]
                df = df.withColumn(feature, when(col(feature).isNull(), lit(mode_value)).otherwise(col(feature)))
        
        return df

    def scale_features(self, df):
        """
        Scale features using the pre-trained StandardScaler.
        """
        missing_cols = [col for col in self.config['feature_names_with_missingness'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for scaling: {missing_cols}")

        assembler = VectorAssembler(inputCols=self.config['feature_names_with_missingness'], outputCol="features_vector")
        df_vectorized = assembler.transform(df)
        df_scaled = self.scaler_model.transform(df_vectorized)
        return df_scaled

    def predict_layer_1(self, df):
        """
        Generate predictions using Layer 1 GBT models for each label.
        """
        predictions = df
        for label in self.config['labels_to_process']:
            model_path = os.path.join(self.config['model_dir'], f"Layer_1_BinaryClassification_gbt_model_label_{label}")
            try:
                model = GBTClassificationModel.load(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load GBT model for label {label}: {e}")

            pred = model.transform(predictions).withColumnRenamed("probability", f"probability_label_{label}")
            predictions = predictions.join(pred.select("row_index", f"probability_label_{label}"), on="row_index", how="left").cache()
        return predictions

    def predict_layer_2(self, df):
        """
        Use Layer 2 Random Forest model for multiclass classification.
        """
        # Validate that all required columns exist
        missing_cols = [col for col in self.config['layer_2_feature_names'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Layer 2: {missing_cols}")

        # Assemble features for Layer 2
        assembler = VectorAssembler(inputCols=self.config['layer_2_feature_names'], outputCol="features")
        df_vectorized = assembler.transform(df)

        # Load the Random Forest model
        rf_model_path = os.path.join(self.config['model_dir'], "Layer_2_MultiClassification_RandomForest_Model")
        try:
            rf_model = RandomForestClassificationModel.load(rf_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Random Forest model: {e}")

        # Generate predictions
        predictions = rf_model.transform(df_vectorized)
        predictions = predictions.withColumn("layer_2_probabilities", vector_to_array(col("probability")))

        return predictions


    def save_results(self, df, output_path):
        """
        Save the final DataFrame to Parquet.
        """
        df.write.mode("overwrite").parquet(output_path)

    def run(self, input_data_path):
        """
        Execute the end-to-end pipeline and return the final DataFrame.
        """
        logger.info("Starting pipeline execution...")

        # Load Data
        original_data = self.load_data(input_data_path)
        logger.info("Data loaded successfully.")

        # Add Missingness Features
        data_with_missingness = self.create_missingness_columns(original_data)

        # Impute Missing Values
        imputed_data = self.impute_features(data_with_missingness)
        logger.info("Data cleaned successfully.")

        # Scale Features
        scaled_data = self.scale_features(imputed_data)
        logger.info("Features scaled successfully.")

        # Predict Layer 1
        layer_1_predictions = self.predict_layer_1(scaled_data)
        logger.info("Layer 1 predictions generated.")

        # Predict Layer 2
        final_predictions = self.predict_layer_2(layer_1_predictions)
        logger.info("Layer 2 predictions generated.")

        # Combine Results with Original Data
        final_output = original_data.join(final_predictions, on="row_index", how="inner")

        # Save Final Results
        self.save_results(final_output, self.config['output_data_path'])
        logger.info(f"Pipeline execution completed. Results saved to {self.config['output_data_path']}.")

        return final_output


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Production Pipeline for Random Forest Layered Model")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input dataset (Parquet format).")
    parser.add_argument("--output_data_path", type=str, required=True, help="Path to save the final output (Parquet format).")

    args = parser.parse_args()

    feature_names = [  # Original features
        'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
        'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
        'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
        'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
        'T-MON-CKP', 'T-PDG', 'T-TPT'
    ]

    feature_names_with_missingness = feature_names + [f"{feature}_missing" for feature in feature_names]

    layer_2_feature_names = feature_names_with_missingness + [f"probability_label_{label}" for label in range(1, 10)]

    config = {
        "scaler_model_path": "../Cleaning & Preparation/scaler_model",  # Path to scaler model
        "model_dir": "./models",  # Directory containing models
        "labels_to_process": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Labels for Layer 1
        "feature_names": feature_names,  # Original features
        "feature_names_with_missingness": feature_names_with_missingness,  # Features + missingness indicators
        "features_continuous": [  # Continuous features
            'ABER-CKGL', 'ABER-CKP', 'P-ANULAR', 'P-JUS-CKGL', 'P-MON-CKP',
            'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
            'T-MON-CKP', 'T-PDG', 'T-TPT'
        ],
        "features_categorical": [  # Categorical features
            'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO', 'ESTADO-M2',
            'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1', 'ESTADO-W2',
            'ESTADO-XO'
        ],
        "layer_2_feature_names": layer_2_feature_names,  # Features for Layer 2
        "output_data_path": args.output_data_path  # Output path from CLI
    }

    pipeline_rf = ProductionPipelineRF(config)
    final_output = pipeline_rf.run(args.input_data_path)
