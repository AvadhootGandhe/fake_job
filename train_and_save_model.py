import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
import joblib
import logging
import os
from pipeline import create_preprocessing_pipeline, save_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_model():
    try:
        # Load the dataset
        logging.info("Loading dataset...")
        resampled_dataset = pd.read_csv("resampled_dataset.csv")
        
        # Create preprocessing pipeline
        logging.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessing_pipeline()
        
        # Prepare features and target
        X = resampled_dataset
        y = resampled_dataset['fraudulent']
        
        # Train-test split
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Preprocess the data
        logging.info("Preprocessing data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save the preprocessing pipeline
        logging.info("Saving preprocessing pipeline...")
        save_pipeline(preprocessor)
        
        # Train XGBoost model
        logging.info("Training XGBoost model...")
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_processed, y_train)
        
        # Evaluate model
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test_processed)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model Performance:")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        logging.info("Saving model...")
        joblib.dump(model, 'models/xgboost_model.joblib')
        
        logging.info("Model saved successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 