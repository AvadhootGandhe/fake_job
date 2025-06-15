from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import logging
from predict_email_fraud import process_email_for_prediction
import os
import pandas as pd
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Job Fraud Detection API")

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and vectorizer
try:
    model = joblib.load('models/xgboost_model.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

class JobPosting(BaseModel):
    # Text fields
    title: str
    description: str
    company_name: str
    location: str
    department: str = None
    salary_range: str = None
    required_experience: str = None
    required_education: str = None
    industry: str = None
    function: str = None
    company_profile: str = None  # Added company profile field
    
    # Boolean fields
    telecommuting: bool = False
    has_company_logo: bool = False
    has_questions: bool = False
    employment_type: str = None
    
    # Numeric fields
    fraud_phrase_count: int = 0
    character_count: int = 0
    avg_salary: float = 0
    required_skills_count: int = 0
    benefits_count: int = 0
    
    # Additional metadata
    posting_date: str = None
    application_deadline: str = None
    job_id: str = None

@app.get("/")
async def read_root():
    return FileResponse('templates/index.html')

@app.post("/scan-job")
async def scan_job(job: JobPosting):
    try:
        # Process job posting with company profile
        job_data = {
            'content': f"{job.title}\n{job.description}\n{job.company_profile if job.company_profile else ''}",
            'sender': job.company_name,
            'subject': job.title,
            'date': None,
            'snippet': job.description[:100]
        }
        
        # Get prediction
        email_df = process_email_for_prediction(job_data)
        
        # Prepare features
        text = email_df['combined_text'].fillna('')
        numeric_features = ['telecommuting', 'has_company_logo', 'has_questions',
                          'fraud_phrase_count', 'character_count', 'avg_salary']
        
        X_numeric = email_df[numeric_features].fillna(0)
        X_text = tfidf.transform(text)
        X_combined = hstack([csr_matrix(X_numeric.values), X_text])
        
        # Make prediction
        prediction = model.predict(X_combined)
        probability = model.predict_proba(X_combined)[0][1]
        
        return {
            "is_fraudulent": bool(prediction[0]),
            "fraud_probability": float(probability),
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
            "features_used": {
                "text_features": ["title", "description", "company_name", "location", "company_profile"],
                "numeric_features": numeric_features,
                "categorical_features": ["employment_type", "industry", "function"]
            }
        }
        
    except Exception as e:
        logging.error(f"Error processing job posting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/analysis-data")
async def get_analysis_data():
    try:
        # Load the cleaned dataset
        df = pd.read_csv('cleaned_data.csv')
        
        # Calculate fraud distribution
        fraud_distribution = df['fraudulent'].value_counts().to_dict()
        
        # Calculate histogram data for character count
        hist_data_fraud = df[df['fraudulent'] == 1]['character_count'].tolist()
        hist_data_legit = df[df['fraudulent'] == 0]['character_count'].tolist()
        
        # Calculate bins that cover both distributions
        all_data = hist_data_fraud + hist_data_legit
        min_val = min(all_data)
        max_val = max(all_data)
        bins = np.linspace(min_val, max_val, 21)  # 20 bins
        
        hist_fraud = np.histogram(hist_data_fraud, bins=bins)
        hist_legit = np.histogram(hist_data_legit, bins=bins)
        
        # Calculate separate histogram for fraudulent jobs with more bins
        fraud_bins = np.linspace(min(hist_data_fraud), max(hist_data_fraud), 31)  # 30 bins for more detail
        fraud_hist = np.histogram(hist_data_fraud, bins=fraud_bins)
        
        # Calculate employment type distribution
        employment_type_dist = df.groupby(['employment_type', 'fraudulent']).size().unstack(fill_value=0)
        employment_type_dist = employment_type_dist.reset_index()
        employment_type_dist.columns = ['employment_type', 'legitimate', 'fraudulent']
        
        return {
            "fraud_distribution": {
                "labels": ["Legitimate", "Fraudulent"],
                "values": [fraud_distribution.get(0, 0), fraud_distribution.get(1, 0)]
            },
            "histogram_data": {
                "values_fraud": hist_fraud[0].tolist(),
                "values_legit": hist_legit[0].tolist(),
                "bins": bins.tolist()
            },
            "fraud_histogram": {
                "values": fraud_hist[0].tolist(),
                "bins": fraud_hist[1].tolist()
            },
            "employment_type_distribution": employment_type_dist.to_dict('records')
        }
    except Exception as e:
        logging.error(f"Error generating analysis data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 