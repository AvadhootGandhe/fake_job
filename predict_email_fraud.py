import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib
import logging
from read_email_latest_email import read_latest_email

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_email_for_prediction(email_data):
    """
    Process email content to match the format expected by the model
    """
    # Extract features from email content
    content = email_data['content']
    
    # Count questions in the email
    question_count = content.count('?')
    
    # Check for common fraud phrases
    fraud_phrases = [
        'urgent', 'immediate', 'verify', 'confirm', 'account', 'password',
        'login', 'security', 'suspicious', 'verify your account',
        'click here', 'verify now', 'account suspended'
    ]
    fraud_phrase_count = sum(1 for phrase in fraud_phrases if phrase.lower() in content.lower())
    
    # Create a DataFrame with the same structure as training data
    email_df = pd.DataFrame({
        'combined_text': [content],
        'telecommuting': [0],  # Default value
        'has_company_logo': [0],  # Default value
        'has_questions': [1 if question_count > 0 else 0],
        'fraud_phrase_count': [fraud_phrase_count],
        'character_count': [len(content)],
        'avg_salary': [0]  # Default value
    })
    
    return email_df

def predict_email_fraud():
    try:
        # Load the saved model and vectorizer
        logging.info("Loading model and vectorizer...")
        model = joblib.load('models/xgboost_model.joblib')
        tfidf = joblib.load('models/tfidf_vectorizer.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        # Read the latest email
        logging.info("Reading latest email...")
        email_data = read_latest_email()
        
        # Print email details
        logging.info("\nEmail Details:")
        logging.info(f"From: {email_data['sender']}")
        logging.info(f"Subject: {email_data['subject']}")
        logging.info(f"Date: {email_data['date']}")
        
        # Process email data
        logging.info("Processing email data...")
        email_df = process_email_for_prediction(email_data)
        
        # Prepare features
        text = email_df['combined_text'].fillna('')
        numeric_features = ['telecommuting', 'has_company_logo', 'has_questions',
                          'fraud_phrase_count', 'character_count', 'avg_salary']
        
        X_numeric = email_df[numeric_features].fillna(0)
        
        # Transform text using saved TF-IDF vectorizer
        X_text = tfidf.transform(text)
        
        # Combine features
        X_combined = hstack([csr_matrix(X_numeric.values), X_text])
        
        # Make prediction
        logging.info("Making prediction...")
        prediction = model.predict(X_combined)
        probability = model.predict_proba(X_combined)
        
        # Get prediction result
        is_fraudulent = bool(prediction[0])
        fraud_probability = probability[0][1]  # Probability of being fraudulent
        
        logging.info(f"\nPrediction Results:")
        logging.info(f"Is Fraudulent: {is_fraudulent}")
        logging.info(f"Fraud Probability: {fraud_probability:.4f}")
        
        return {
            'is_fraudulent': is_fraudulent,
            'fraud_probability': fraud_probability,
            'email_details': email_data
        }
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    result = predict_email_fraud()
    print("\nFinal Results:")
    print(f"Is Fraudulent: {result['is_fraudulent']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}") 