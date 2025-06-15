import shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from predict_email_fraud import predict_email_fraud, process_email_for_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_shap_plot(email_data, model, tfidf, feature_names):
    """
    Generate SHAP values and plot for the prediction
    """
    # Process email data
    email_df = process_email_for_prediction(email_data)
    
    # Prepare features
    text = email_df['combined_text'].fillna('')
    numeric_features = ['telecommuting', 'has_company_logo', 'has_questions',
                       'fraud_phrase_count', 'character_count', 'avg_salary']
    
    X_numeric = email_df[numeric_features].fillna(0)
    X_text = tfidf.transform(text)
    X_combined = np.hstack([X_numeric.values, X_text.toarray()])
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_combined)
    
    # Plot SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_combined, feature_names=feature_names, show=False)
    plt.title('SHAP Values for Email Prediction')
    plt.tight_layout()
    plt.savefig('shap_plot.png')
    plt.close()
    
    return shap_values

def generate_keyword_cloud(email_data, tfidf):
    """
    Generate keyword cloud for the email content
    """
    # Get TF-IDF scores for the email
    text = email_data['content']
    tfidf_matrix = tfidf.transform([text])
    
    # Get feature names and their scores
    feature_names = tfidf.get_feature_names_out()
    scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         max_words=100).generate_from_frequencies(scores)
    
    # Plot word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keyword Cloud for Email Content')
    plt.tight_layout()
    plt.savefig('keyword_cloud.png')
    plt.close()

def explain_prediction():
    try:
        # Load model and vectorizer
        logging.info("Loading model and vectorizer...")
        model = joblib.load('models/xgboost_model.joblib')
        tfidf = joblib.load('models/tfidf_vectorizer.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        # Get prediction
        logging.info("Getting prediction...")
        result = predict_email_fraud()
        
        # Generate explanations
        logging.info("Generating SHAP plot...")
        shap_values = generate_shap_plot(result['email_details'], model, tfidf, feature_names)
        
        logging.info("Generating keyword cloud...")
        generate_keyword_cloud(result['email_details'], tfidf)
        
        logging.info("Explanations generated successfully!")
        logging.info("Check 'shap_plot.png' and 'keyword_cloud.png' for visualizations")
        
        return {
            'prediction': result,
            'shap_values': shap_values
        }
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    explain_prediction() 