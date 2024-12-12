import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, monotonically_increasing_id, mean, count, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the UDF for vector to array conversion
def vector_to_array_udf(vector):
    if isinstance(vector, DenseVector):
        return list(vector)
    return None

vector_to_array = udf(vector_to_array_udf, ArrayType(DoubleType()))

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
        assembler = VectorAssembler(inputCols=self.config['feature_names_with_missingness'], outputCol="features_vector")
        df_vectorized = assembler.transform(df)
        df_scaled = self.scaler_model.transform(df_vectorized)

        # Rename "scaled_features" back to "features"
        df_scaled = df_scaled.withColumnRenamed("scaled_features", "features")
        return df_scaled

    def predict_layer_1(self, df):
        """
        Generate predictions using Layer 1 GBT models for each label.
        """
        # Retain only necessary columns: features and row_index
        predictions = df.select("row_index", "features")

        # Process each label one at a time
        for label in self.config['labels_to_process']:
            try:
                logger.info(f"Processing label: {label}")

                model_path = os.path.join(self.config['model_dir'], f"Layer_1_BinaryClassification_gbt_model_label_{label}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model for label {label} does not exist at {model_path}")

                logger.info(f"Loading model for label {label} from {model_path}")
                model = GBTClassificationModel.load(model_path)

                pred = model.transform(predictions)
                pred = pred.withColumnRenamed("probability", f"probability_label_{label}")
                predictions = predictions.join(pred.select("row_index", f"probability_label_{label}"), on="row_index", how="left").cache()

                logger.info(f"Label {label} processed successfully.")

            except Exception as e:
                logger.error(f"Failed to process label {label}: {e}")
                raise RuntimeError(f"Error processing label {label}: {e}")

        return predictions



    def predict_layer_2(self, df):
        """
        Use Layer 2 Random Forest model for multiclass classification.
        """
        # Retain only necessary columns: row_index and Layer 2 features
        required_cols = ["row_index"] + self.config["layer_2_feature_names"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns for Layer 2: {missing_cols}")
            raise ValueError(f"Missing required columns for Layer 2: {missing_cols}")

        # Assemble features for Layer 2
        assembler = VectorAssembler(inputCols=self.config["layer_2_feature_names"], outputCol="layer_2_features")
        try:
            df_vectorized = assembler.transform(df)
            logger.info("Features assembled successfully for Layer 2.")
        except Exception as e:
            logger.error(f"Error during VectorAssembler transformation: {e}")
            raise RuntimeError(f"VectorAssembler transformation failed: {e}")

        # Load the Random Forest model
        rf_model_path = os.path.join(self.config["model_dir"], "Layer_2_MultiClassification_RandomForest_Model")
        try:
            rf_model = RandomForestClassificationModel.load(rf_model_path)
            logger.info(f"Random Forest model loaded from {rf_model_path}")
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
            raise RuntimeError(f"Failed to load Random Forest model: {e}")

        # Generate predictions
        try:
            predictions = rf_model.transform(df_vectorized)
            logger.info("Layer 2 predictions generated successfully.")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

        return predictions.select(
            "row_index", "prediction", "layer_2_probabilities", *[f"probability_label_{label}" for label in self.config["labels_to_process"]]
        )


    def save_results(self, df, output_path):
        """
        Save the final DataFrame to Parquet.
        """
        try:
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Results successfully saved to {output_path}.")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise RuntimeError(f"Error saving results to {output_path}: {e}")


    def run(self, input_data_path):
        """
        Execute the end-to-end pipeline and return the final DataFrame.
        """
        logger.info("Starting pipeline execution...")

        original_data = self.load_data(input_data_path)
        logger.info("Data loaded successfully.")

        data_with_missingness = self.create_missingness_columns(original_data)
        imputed_data = self.impute_features(data_with_missingness)
        logger.info("Data cleaned successfully.")

        scaled_data = self.scale_features(imputed_data)
        logger.info("Features scaled successfully.")

        layer_1_predictions = self.predict_layer_1(scaled_data)
        logger.info("Layer 1 predictions generated.")

        final_predictions = self.predict_layer_2(layer_1_predictions)
        logger.info("Layer 2 predictions generated.")

        final_output = original_data.join(final_predictions, on="row_index", how="inner")
        self.save_results(final_output, self.config['output_data_path'])
        logger.info(f"Pipeline execution completed. Results saved to {self.config['output_data_path']}.")

        return final_output


if __name__ == "__main__":
    import argparse

    # Default file paths
    default_input_path = "../../Data/Cleaning & Preparation/Approach_2/Train Test Data/test_data.parquet"
    default_output_path = "./pipeline_output.parquet"

    # Argument parser
    parser = argparse.ArgumentParser(description="Production Pipeline for Random Forest Layered Model")
    parser.add_argument("--input_data_path", type=str, default=default_input_path,
                        help=f"Path to the input dataset (Parquet format). Default: {default_input_path}")
    parser.add_argument("--output_data_path", type=str, default=default_output_path,
                        help=f"Path to save the final output (Parquet format). Default: {default_output_path}")

    args = parser.parse_args()

    # Define original features
    feature_names = [
        'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
        'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
        'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
        'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
        'T-MON-CKP', 'T-PDG', 'T-TPT'
    ]

    # Include missingness features
    feature_names_with_missingness = feature_names + [f"{feature}_missing" for feature in feature_names]

    # Define Layer 2 features
    layer_2_feature_names = ["features"] + [f"probability_label_{label}" for label in range(1, 10)]

    # Define continuous and categorical features
    features_continuous = [
        'ABER-CKGL', 'ABER-CKP', 'P-ANULAR', 'P-JUS-CKGL', 'P-MON-CKP',
        'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
        'T-MON-CKP', 'T-PDG', 'T-TPT'
    ]

    features_categorical = [
        'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO', 'ESTADO-M2',
        'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1', 'ESTADO-W2',
        'ESTADO-XO'
    ]

    # Define pipeline configuration
    config = {
        "scaler_model_path": "../../Cleaning & Preparation/scaler_model",  # Path to scaler model
        "model_dir": "../models",  # Directory containing models
        "labels_to_process": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Labels for Layer 1
        "feature_names": feature_names,  # Original features
        "feature_names_with_missingness": feature_names_with_missingness,  # Features + missingness indicators
        "features_continuous": features_continuous,  # Continuous features
        "features_categorical": features_categorical,  # Categorical features
        "layer_2_feature_names": layer_2_feature_names,  # Features for Layer 2
        "output_data_path": args.output_data_path  # Output path
    }

    # Initialize and run the pipeline
    pipeline_rf = ProductionPipelineRF(config)
    pipeline_rf.run(args.input_data_path)
