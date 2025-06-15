import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import joblib
import logging

def flatten_text_column(x):
    """Flatten a single-column DataFrame or Series to 1D array for TfidfVectorizer."""
    return x.squeeze()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline that handles both numeric and text features
    """
    # Define numeric features
    numeric_features = ['telecommuting', 'has_company_logo', 'has_questions',
                       'fraud_phrase_count', 'character_count', 'avg_salary']
    
    # Define text features
    text_features = ['combined_text']
    
    # Create numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    # Create text pipeline with flattening
    text_pipeline = Pipeline([
        ('flatten', FunctionTransformer(flatten_text_column, validate=False)),
        ('tfidf', TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2)
        ))
    ])
    
    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('text', text_pipeline, text_features)
        ],
        sparse_threshold=0.0  # This ensures dense output
    )
    
    return preprocessor

def save_pipeline(pipeline, filepath='models/preprocessing_pipeline.joblib'):
    """
    Save the preprocessing pipeline to disk
    """
    try:
        joblib.dump(pipeline, filepath)
        logging.info(f"Pipeline saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving pipeline: {str(e)}")
        raise

def load_pipeline(filepath='models/preprocessing_pipeline.joblib'):
    """
    Load the preprocessing pipeline from disk
    """
    try:
        pipeline = joblib.load(filepath)
        logging.info(f"Pipeline loaded successfully from {filepath}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline: {str(e)}")
        raise

def preprocess_data(df, pipeline=None, fit=False):
    """
    Preprocess the data using the pipeline
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the features
    pipeline : sklearn Pipeline or None
        If None, a new pipeline will be created
    fit : bool
        Whether to fit the pipeline or just transform the data
        
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed features
    pipeline : sklearn Pipeline
        The fitted pipeline if fit=True
    """
    if pipeline is None:
        pipeline = create_preprocessing_pipeline()
    
    try:
        if fit:
            X = pipeline.fit_transform(df)
            return X, pipeline
        else:
            X = pipeline.transform(df)
            return X
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load sample data
        df = pd.read_csv("resampled_dataset.csv")
        
        # Create and fit pipeline
        X, pipeline = preprocess_data(df, fit=True)
        
        # Save pipeline
        save_pipeline(pipeline)
        
        # Load pipeline and transform new data
        loaded_pipeline = load_pipeline()
        X_new = preprocess_data(df, pipeline=loaded_pipeline)
        
        logging.info("Pipeline test completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in pipeline test: {str(e)}")
        raise 