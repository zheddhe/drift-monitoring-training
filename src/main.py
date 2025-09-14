import logging
# from step1_download_data import download_files
from step2_load_and_process_data import load_data, preprocess_data
from step3_train_and_evaluate_model import train_and_evaluate
from step4_save_model_and_data import save_model_and_data
from step5_generate_report import generate_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    try:
        # Load & preprocess data
        logging.info("Loading and preprocessing data...")
        jan_data = load_data('./data/green_tripdata_2022-01.parquet')
        jan_preprocessed_data = preprocess_data(jan_data)
        logging.info("Data loaded and preprocessed successfully.")

        # Define feature and target
        target = "duration_min"
        num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
        cat_features = ["PULocationID", "DOLocationID"]

        # Train and evaluate model
        logging.info("Training and evaluating model...")
        model, train_data, val_data, _, _ = train_and_evaluate(
            jan_preprocessed_data, num_features, cat_features, target
        )
        logging.info("Model trained and evaluated successfully.")

        # Save model and data
        logging.info("Saving model and data...")
        save_model_and_data(model, val_data)
        logging.info("Model and data saved successfully.")

        # Generate report
        print(train_data.head(3))
        logging.info("generating report...")
        report = generate_report(train_data, val_data, num_features, cat_features)
        logging.info("Report generated successfully.")

        # Extract key metrics
        result = report.as_dict()
        print(
            "Drift score of the prediction column: ",
            result['metrics'][0]['result']['drift_score'])
        print(
            "Number of drifted columns: ",
            result['metrics'][1]['result']['number_of_drifted_columns']
        )
        print(
            "Share of missing values: ",
            result['metrics'][2]['result']['current']['share_of_missing_values']
        )
        # report.show(mode='inline')
        report.save_html("Drift_Report.html")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
