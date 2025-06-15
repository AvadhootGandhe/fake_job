# Job Fraud Detection System

Welcome to the Job Fraud Detection System! This application helps you identify potentially fraudulent job postings using machine learning. It analyzes various aspects of job listings to provide a risk assessment and fraud probability score.

## 🌟 Features

- **Real-time Analysis**: Instantly analyze job postings for potential fraud
- **Comprehensive Detection**: Evaluates multiple aspects of job listings
- **User-friendly Interface**: Simple and intuitive web interface
- **Detailed Results**: Get fraud probability scores and risk levels
- **Feature Insights**: See which aspects of the job posting were analyzed

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd job-fraud-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
uvicorn api:app --reload
```

4. Open your web browser and navigate to:
```
http://localhost:8000
```

## 📝 How to Use

### Analyzing a Job Posting

1. **Basic Information**
   - Enter the job title (required)
   - Provide the company name (required)
   - Specify the job location (required)
   - Add the department (optional)

2. **Company Details**
   - Fill in the company profile (optional)
   - This helps in verifying the company's legitimacy

3. **Job Details**
   - Select the industry
   - Choose the job function
   - Specify the employment type
   - Enter the salary range

4. **Requirements**
   - Add required experience
   - Specify required education
   - Include any specific skills

5. **Additional Information**
   - Provide a detailed job description (required)
   - Check relevant boxes:
     - Telecommuting option
     - Company logo presence
     - Application questions
   - Add posting date and deadline (optional)

6. **Submit for Analysis**
   - Click "Analyze Job Posting"
   - Wait for the results

### Understanding Results

The system provides:

1. **Fraud Prediction**
   - Legitimate or Fraudulent classification
   - Probability score (0-100%)
   - Risk Level (Low/Medium/High)

2. **Features Used**
   - Text Features: title, description, company details
   - Numeric Features: various counts and metrics
   - Categorical Features: job type, industry, function

## 🔍 How It Works

The system uses a trained XGBoost model to analyze job postings. It considers:

1. **Preprocessing Pipeline**
   - All input data is processed through a scikit-learn pipeline that handles missing values, scales numeric features, and vectorizes text features.
   - This ensures that the data is always prepared in the same way for both training and prediction.

2. **Text Analysis**
   - Job title and description
   - Company profile and name
   - Location and requirements

3. **Numeric Features**
   - Character count
   - Fraud phrase detection
   - Required skills count
   - Benefits count

4. **Categorical Features**
   - Employment type
   - Industry
   - Job function

## 🛠️ Technical Details

### Model Information
- Algorithm: XGBoost
- Features: TF-IDF vectorization for text
- Training: Pre-trained on verified job postings

### Preprocessing Pipeline
- The system uses a scikit-learn pipeline to ensure consistent and robust data preprocessing.
- **Numeric features** are imputed (missing values filled with 0) and scaled.
- **Text features** (such as job description) are flattened and vectorized using TF-IDF (with unigrams and bigrams, up to 300 features).
- The pipeline is saved and loaded automatically, ensuring that the same transformations are applied during both training and prediction.
- This approach improves maintainability and reproducibility of the model.

### API Endpoints
- `GET /`: Main application interface
- `POST /scan-job`: Job analysis endpoint
- `GET /health`: System health check

## 📊 Risk Levels

- **Low Risk** (0-40%): Likely legitimate posting
- **Medium Risk** (40-70%): Exercise caution
- **High Risk** (70-100%): Potential fraud

## 🔒 Security

- All data is processed locally
- No personal information is stored
- Secure API endpoints

## 🤝 Contributing

Feel free to contribute to this project by:
1. Reporting bugs
2. Suggesting enhancements
3. Submitting pull requests

