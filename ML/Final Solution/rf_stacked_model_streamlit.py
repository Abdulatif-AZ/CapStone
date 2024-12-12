import streamlit as st
from pyspark.sql import SparkSession
from rf_stacked_model_pipeline import ProductionPipelineRF
import os

@st.cache_resource
def create_spark_session():
    return SparkSession.builder \
        .appName("Streamlit_PySpark") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

spark = create_spark_session()

def main():
    st.title("3W PySpark Application with Pipeline Integration")

    # Input dataset path
    dataset_path = st.text_input("Enter the path to your dataset (e.g., s3://bucket-name/dataset.parquet):")

    if dataset_path:
        try:
            # Load the dataset directly from the path
            spark_df = spark.read.parquet(dataset_path)

            # Select unique 'well' values
            unique_wells = [row["well"] for row in spark_df.select("well").distinct().collect()]
            selected_well = st.selectbox("Select a Well", unique_wells)

            if selected_well:
                # Filter dataset by selected 'well'
                filtered_data = spark_df.filter(spark_df["well"] == selected_well)
                st.write(f"### Filtered Data for Well: {selected_well}")

                # Display the filtered dataset
                st.write(filtered_data.limit(10).toPandas())

                # Run the pipeline
                if st.button("Run Pipeline"):
                    st.write("### Running Pipeline...")

                    # Save the filtered data to a temporary Parquet file
                    temp_filtered_data_path = "./temp_filtered_data.parquet"
                    filtered_data.write.mode("overwrite").parquet(temp_filtered_data_path)

                    # Define pipeline configuration
                    pipeline_config = {
                        "scaler_model_path": "../../Cleaning & Preparation/scaler_model",  # Adjust path as needed
                        "model_dir": "../models",  # Adjust path as needed
                        "labels_to_process": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        "feature_names": [
                            'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
                            'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
                            'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
                            'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
                            'T-MON-CKP', 'T-PDG', 'T-TPT'
                        ],
                        "feature_names_with_missingness": [
                            'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
                            'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
                            'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
                            'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
                            'T-MON-CKP', 'T-PDG', 'T-TPT'
                        ] + [f"{feature}_missing" for feature in [
                            'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-PXO',
                            'ESTADO-M2', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1',
                            'ESTADO-W2', 'P-ANULAR', 'ESTADO-XO', 'P-JUS-CKGL', 'P-MON-CKP',
                            'P-JUS-CKP', 'P-MON-CKGL', 'P-PDG', 'P-TPT', 'QGL', 'T-JUS-CKP',
                            'T-MON-CKP', 'T-PDG', 'T-TPT'
                        ]],
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
                        "layer_2_feature_names": ["features"] + [f"probability_label_{label}" for label in range(1, 10)],
                        "output_data_path": "./output_pipeline_results.parquet"
                    }

                    pipeline = ProductionPipelineRF(pipeline_config)
                    final_output = pipeline.run(temp_filtered_data_path)

                    st.write("### Pipeline Output")
                    output_df = final_output.limit(10).toPandas()
                    st.write(output_df)

                    # Clean up temporary files
                    os.remove(temp_filtered_data_path)

                    # Plot results
                    st.write("### Visualizations")

                    # Plot continuous features
                    for feature in pipeline_config["features_continuous"]:
                        if feature in output_df.columns:
                            st.line_chart(output_df[["timestamp", feature]].set_index("timestamp"))

                    # Plot first-layer probabilities
                    for label in pipeline_config["labels_to_process"]:
                        column_name = f"probability_label_{label}"
                        if column_name in output_df.columns:
                            st.line_chart(output_df[["timestamp", column_name]].set_index("timestamp"))

                    # Plot 'label' and 'class'
                    if all(col in output_df.columns for col in ["label", "class"]):
                        st.line_chart(output_df[["timestamp", "label", "class"]].set_index("timestamp"))

        except Exception as e:
            st.error(f"Error reading dataset: {e}")

if __name__ == "__main__":
    main()
