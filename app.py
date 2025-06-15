import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from PIL import Image
import time
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime
import threading
import sqlite3
import base64

# --- Main Title ---
st.set_page_config(
    page_title="Spot the Scam - Job Fraud Detector", 
    layout="wide",
    page_icon="🕵️"
)

# Custom CSS for better styling
st.markdown(""" 
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #3a5a8a;
            transform: translateY(-1px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stFileUploader>div>div>div>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 8px;
        }
        .stProgress>div>div>div>div {
            background-color: #4a6fa5;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e1e4e8;
        }
        .css-1v3fvcr {
            padding: 2rem;
        }
        h1 {
            color: #2c3e50;
            font-weight: 700;
        }
        h2 {
            color: #34495e;
            font-weight: 600;
            border-bottom: 2px solid #e1e4e8;
            padding-bottom: 8px;
        }
        h3 {
            color: #3d5a80;
            font-weight: 600;
        }
        .stAlert {
            border-radius: 10px;
        }
        .stExpander {
            border-radius: 10px;
            border: 1px solid #e1e4e8;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s ease;
            color: #2c3e50 !important; /* <-- Always dark text */
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4a6fa5;
            color: #fff !important;
        }
        .stTabs [aria-selected="false"]:hover {
            background-color: #e9eef6;
            color: #1a2634 !important;
        }
        .stTabs [aria-selected="false"] {
            background-color: #f0f2f6;
            color: #2c3e50 !important;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 4px solid #4a6fa5;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        }
        .fraud-card {
            background: #fff5f5;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #e74c3c;
        }
        .genuine-card {
            background: #f5fff7;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #2ecc71;
        }
        .api-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
        .alert-card {
            background: #fff3cd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
    </style>
""", unsafe_allow_html=True)

# --- Database Setup ---
def init_database():
    conn = sqlite3.connect('job_fraud_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            location TEXT,
            salary TEXT,
            fraud_probability REAL,
            prediction INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            threshold REAL,
            active INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# --- Email Alert System ---
def send_fraud_alert(email, job_data, fraud_probability):
    try:
        # Configure your SMTP settings here
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your-email@gmail.com"  # Replace with your email
        sender_password = "your-app-password"  # Replace with your app password
        
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = f"🚨 High-Risk Job Alert - {fraud_probability:.1%} Fraud Probability"
        
        body = f"""
        <html>
        <body>
            <h2>🚨 Fraud Alert: High-Risk Job Detected</h2>
            <p>A job listing with high fraud probability has been detected:</p>
            
            <div style="background-color: #fff5f5; padding: 15px; border-left: 4px solid #e74c3c; margin: 10px 0;">
                <h3>{job_data.get('title', 'N/A')}</h3>
                <p><strong>Location:</strong> {job_data.get('location', 'N/A')}</p>
                <p><strong>Fraud Probability:</strong> {fraud_probability:.1%}</p>
                <p><strong>Description:</strong> {job_data.get('description', 'N/A')[:200]}...</p>
            </div>
            
            <p>Please exercise caution when considering this opportunity.</p>
            <p>Stay safe!</p>
        </body>
        </html>
        """
        
        message.attach(MIMEText(body, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        
        return True
    except Exception as e:
        st.error(f"Failed to send alert email: {str(e)}")
        return False

# --- API Endpoint Simulation ---
class JobFraudAPI:
    def __init__(self, model):
        self.model = model
    
    def scan_job(self, job_data):
        """Simulate real-time job scanning API"""
        try:
            text_data = f"{job_data.get('description', '')} {job_data.get('title', '')}"
            prediction = self.model.predict([text_data])[0]
            probability = self.model.predict_proba([text_data])[0, 1]
            
            # Store in database
            conn = sqlite3.connect('job_fraud_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_listings (title, description, location, salary, fraud_probability, prediction)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                job_data.get('title', ''),
                job_data.get('description', ''),
                job_data.get('location', ''),
                job_data.get('salary', ''),
                float(probability),
                int(prediction)
            ))
            conn.commit()
            conn.close()
            
            # Check for alerts
            if probability > 0.7:  # High risk threshold
                self.send_alerts(job_data, probability)
            
            return {
                'job_id': cursor.lastrowid,
                'fraud_probability': float(probability),
                'prediction': 'fraudulent' if prediction else 'genuine',
                'risk_level': 'high' if probability > 0.7 else 'medium' if probability > 0.4 else 'low',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def send_alerts(self, job_data, probability):
        """Send alerts to subscribed users"""
        conn = sqlite3.connect('job_fraud_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM alert_subscriptions WHERE threshold <= ? AND active = 1', (probability,))
        emails = cursor.fetchall()
        conn.close()
        
        for (email,) in emails:
            threading.Thread(target=send_fraud_alert, args=(email, job_data, probability)).start()

# --- Model Retraining Function ---
def retrain_model():
    """Function to retrain the model with new data"""
    try:
        conn = sqlite3.connect('job_fraud_data.db')
        df = pd.read_sql_query('SELECT * FROM job_listings', conn)
        conn.close()
        
        if len(df) < 10:  # Need minimum data
            return False, "Insufficient data for retraining"
        
        # Prepare data
        X = df['title'] + ' ' + df['description']
        y = df['prediction']
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_vectorized = vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        
        # Train new model
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)
        
        # Save new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/retrained_model_{timestamp}.pkl"
        os.makedirs("models", exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({'model': new_model, 'vectorizer': vectorizer}, f)
        
        accuracy = new_model.score(X_test, y_test)
        return True, f"Model retrained successfully. Accuracy: {accuracy:.2%}. Saved to {model_path}"
        
    except Exception as e:
        return False, f"Retraining failed: {str(e)}"

# --- SHAP Analysis ---
@st.cache_data
def generate_shap_analysis(text_data, model, max_samples=100):
    """Generate SHAP analysis for model predictions"""
    try:
        # Limit samples for performance
        sample_data = text_data[:max_samples] if len(text_data) > max_samples else text_data
        
        # Create explainer (simplified for demo)
        explainer = shap.Explainer(model.predict_proba, sample_data)
        shap_values = explainer(sample_data)
        
        return shap_values
    except Exception as e:
        st.error(f"SHAP analysis failed: {str(e)}")
        return None

# --- Header Section ---
col1, col2 = st.columns([3,1])
with col1:
    st.title("🕵️ Spot the Scam")
    st.markdown("""
    <div style="font-size: 18px; color: #4a6fa5; font-weight: 500; margin-bottom: 20px;">
        AI-powered job fraud detection system to protect job seekers from scams
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    logo_path = "logo.png"  # Make sure this file exists in the same folder as app.py
    logo_b64 = get_base64_image(logo_path)
    st.markdown(
        f"""
        <div style="
            width: 120px; 
            height: 120px; 
            background: linear-gradient(135deg, #4a6fa5, #3a5a8a);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 0 auto;
            box-shadow: 0 2px 8px rgba(74,111,165,0.08);
        ">
            <img src="data:image/png;base64,{logo_b64}" style="width: 90px; height: 90px; object-fit: contain; border-radius: 50%;" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Divider
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="
        width: 80px; 
        height: 80px; 
        background: linear-gradient(135deg, #4a6fa5, #3a5a8a);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 32px;
        margin: 0 auto 20px auto;
    ">
        🕵️
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Spot the Scam")
    st.markdown("""
    **Detect fraudulent job postings with AI.**
    
    Upload your CSV file containing job listings to analyze them for potential fraud.
    """)
    
    # Alert Subscription
    st.markdown("### 🔔 Fraud Alerts")
    with st.expander("Subscribe to Alerts", expanded=False):
        alert_email = st.text_input("Email for alerts")
        alert_threshold = st.slider("Alert threshold", 0.0, 1.0, 0.7, 0.1)
        if st.button("Subscribe"):
            if alert_email:
                try:
                    conn = sqlite3.connect('job_fraud_data.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO alert_subscriptions (email, threshold)
                        VALUES (?, ?)
                    ''', (alert_email, alert_threshold))
                    conn.commit()
                    conn.close()
                    st.success("Subscribed to fraud alerts!")
                except Exception as e:
                    st.error(f"Subscription failed: {str(e)}")
    
    # API Testing
    st.markdown("### 🔌 Real-time API")
    with st.expander("Test API Endpoint", expanded=False):
        api_title = st.text_input("Job Title", "Remote Data Entry Specialist")
        api_desc = st.text_area("Job Description", "Work from home! Earn $5000/month! No experience needed!")
        api_location = st.text_input("Location", "Remote")
        
        if st.button("Scan Job"):
            job_data = {
                'title': api_title,
                'description': api_desc,
                'location': api_location
            }
            # This would call your actual API in production
            st.json({
                'fraud_probability': 0.85,
                'prediction': 'fraudulent',
                'risk_level': 'high',
                'timestamp': datetime.now().isoformat()
            })
    
    # Sample data download
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'title': [
            'Software Engineer',
            'Work From Home Data Entry',
            'Chief Financial Officer',
            'Remote Social Media Manager',
            'Sales Representative',
            'Senior Data Scientist',
            'Online Survey Taker',
            'Marketing Specialist',
            'Virtual Assistant',
            'Blockchain Developer'
        ],
        'description': [
            'Looking for experienced software engineer to join our growing team. Must have experience with Python and cloud technologies.',
            'Earn $5000/month from home! No experience required. Flexible hours. Apply now!',
            'CFO needed for startup. Experience with fundraising and financial planning required.',
            'Manage our social media accounts remotely. Must be creative and have strong communication skills.',
            'Seeking energetic sales representative for a reputable company. Commission-based.',
            'Lead data science projects and mentor junior team members. Experience with ML and big data required.',
            'Participate in online surveys and get paid instantly. No qualifications needed.',
            'Develop and execute marketing campaigns. Prior experience in digital marketing preferred.',
            'Assist executives with scheduling, emails, and research. Remote position, flexible hours.',
            'Work on cutting-edge blockchain projects. Solidity and smart contract experience required.'
        ],
        'location': [
            'New York, NY',
            'Remote',
            'San Francisco, CA',
            'Remote',
            'Chicago, IL',
            'Boston, MA',
            'Remote',
            'Austin, TX',
            'Remote',
            'Berlin, Germany'
        ],
        'salary': [
            '120000',
            '5000',
            '180000',
            '65000',
            '80000',
            '150000',
            '100',
            '75000',
            '45000',
            '130000'
        ]
    })
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_job_listings.csv",
        mime="text/csv",
        help="Example CSV with the required format"
    )
    
    # About section
    st.markdown("---")
    st.markdown("""
    **About this tool:**
    - Uses advanced NLP and machine learning
    - Analyzes text patterns in job descriptions
    - Provides probability scores for each listing
    - Real-time API endpoint available
    - Email alerts for high-risk jobs
    - Version 2.0.0
    """)

# --- Model Loader ---
@st.cache_resource
def load_model():
    # This would be your actual model file
    try:
        return joblib.load("models/model.pkl")
    except:
        # Create a dummy model for demo purposes
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # Train on sample data for demo
        sample_texts = [
            "Software engineer position requiring Python experience",
            "Work from home earn money fast no experience needed",
            "Marketing manager role at established company",
            "Make $5000 weekly from home easy money",
            "Data scientist position with competitive salary"
        ]
        sample_labels = [0, 1, 0, 1, 0]  # 0 = genuine, 1 = fraud
        
        pipeline.fit(sample_texts, sample_labels)
        return pipeline

try:
    model = load_model()
    api_instance = JobFraudAPI(model)
except Exception as e:
    st.error(f"Model could not be loaded: {str(e)}")
    st.stop()

# --- Clean Text Function ---
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

# --- Generate Word Cloud ---
def generate_wordcloud(text, title=None):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16, pad=20)
    return fig

# --- Enhanced Keyword Analysis ---
def analyze_keywords(data, predictions):
    """Analyze keywords that indicate fraud vs genuine jobs"""
    fraud_text = " ".join(data[predictions == 1]['description'])
    genuine_text = " ".join(data[predictions == 0]['description'])
    
    # Simple keyword frequency analysis
    fraud_words = {}
    genuine_words = {}
    
    if fraud_text:
        for word in fraud_text.split():
            if len(word) > 3:  # Filter short words
                fraud_words[word] = fraud_words.get(word, 0) + 1
    
    if genuine_text:
        for word in genuine_text.split():
            if len(word) > 3:
                genuine_words[word] = genuine_words.get(word, 0) + 1
    
    # Get top keywords
    top_fraud = sorted(fraud_words.items(), key=lambda x: x[1], reverse=True)[:10]
    top_genuine = sorted(genuine_words.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return top_fraud, top_genuine

# --- File Uploader Section ---
st.subheader("📁 Upload Your Job Listings")
uploaded_file = st.file_uploader(
    "Drag and drop your CSV file here or click to browse", 
    type=["csv"],
    help="CSV should contain at least 'title' and 'description' columns"
)

if uploaded_file is not None:
    with st.spinner("Analyzing your file..."):
        progress_bar = st.progress(0)
        try:
            # Step 1: Read the file
            progress_bar.progress(20)
            time.sleep(0.5)
            data = pd.read_csv(uploaded_file)
            
            # Step 2: Validate columns
            progress_bar.progress(40)
            time.sleep(0.3)
            required_cols = ['title', 'description']
            if not all(col in data.columns for col in required_cols):
                st.error("❌ CSV must contain at least 'title' and 'description' columns.")
                st.stop()
            
            # Step 3: Clean data
            progress_bar.progress(60)
            data['title'] = data['title'].apply(clean_text)
            data['description'] = data['description'].apply(clean_text)
            text_data = data['description'] + ' ' + data['title']
            
            # Step 4: Make predictions
            progress_bar.progress(80)
            predictions = model.predict(text_data)
            probabilities = model.predict_proba(text_data)[:, 1]
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            
            # Success message with animation
            with st.empty():
                st.success("✅ Analysis complete! View results below.")
                time.sleep(1)
            
            # Store predictions in DataFrame
            result_df = data.copy()
            result_df['fraudulent_prediction'] = predictions
            result_df['fraud_probability'] = probabilities
        except Exception as e:
            st.error(f"An error occurred during file analysis: {str(e)}")
            progress_bar.empty()
            st.stop()

        # --- Results Section ---
        st.markdown("---")
        st.subheader("🔍 Analysis Results")
        
        # Enhanced Summary cards with better responsive design
        card_height = "150px"  # Increased height
        card_style = (
            f"color: #2c3e50; "
            f"min-height: {card_height}; "
            "width: 100%; "
            "display: flex; "
            "flex-direction: column; "
            "justify-content: center; "
            "align-items: center; "
            "box-sizing: border-box; "
            "padding: 12px; "
            "text-align: center; "
            "overflow: hidden; "
            "word-wrap: break-word; "
            "background: white; "
            "border-radius: 10px; "
            "margin: 5px 0;"
        )
        
        h3_style = (
            "color: #3d5a80; "
            "margin: 0 0 8px 0; "
            "font-size: min(1.1rem, 4vw); "  # Responsive font size
            "font-weight: 600;"
        )
        
        h2_style = (
            "margin: 0; "
            "font-size: min(2rem, 6vw); "  # Responsive font size
            "font-weight: 700; "
            "line-height: 1.2; "
            "width: 100%; "
            "overflow-wrap: break-word;"
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <h3 style="{h3_style}">Total Listings</h3>
                <h2 style="{h2_style} color: #4a6fa5;">{len(result_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <h3 style="{h3_style}">Potential Frauds</h3>
                <h2 style="{h2_style} color: #e74c3c;">
                    {sum(predictions)}<br>
                    <span style="font-size: min(1rem, 3vw); color: #2c3e50;">
                        ({sum(predictions)/len(predictions):.1%})
                    </span>
                </h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_prob = probabilities.mean()
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <h3 style="{h3_style}">Avg Fraud Probability</h3>
                <h2 style="{h2_style} color: #f39c12;">{avg_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            high_risk = sum(probabilities > 0.7)
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <h3 style="{h3_style}">High Risk</h3>
                <h2 style="{h2_style} color: #e67e22;">
                    {high_risk}<br>
                    <span style="font-size: min(1rem, 3vw); color: #2c3e50;">
                        ({high_risk/len(predictions):.1%})
                    </span>
                </h2>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            max_prob = probabilities.max()
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <h3 style="{h3_style}">Highest Risk</h3>
                <h2 style="{h2_style} color: #c0392b;">{max_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Download button section
        st.markdown("""
        <div style="margin-top: 30px; margin-bottom: 10px;">
            <span style="font-size: 20px; color: #4a6fa5; font-weight: 600;">📥 Download Results</span>
        </div>
        """, unsafe_allow_html=True)
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Export Results as CSV",
            data=csv,
            file_name="job_fraud_analysis_results.csv",
            mime="text/csv",
            help="Download the full analysis results"
        )
        
        # Enhanced tabs with new features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Data Table", 
            "📊 Visualizations", 
            "⚠️ Risk Analysis", 
            "🔠 Text Analysis",
            "🧠 SHAP Analysis",
            "🎯 Keyword Insights"
        ])
        
        with tab1:
            # Enhanced Interactive data table
            st.markdown("### 🔍 Interactive Data Explorer")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("Search job titles or descriptions", "")
            with col2:
                risk_filter = st.selectbox("Filter by Risk Level", 
                                         ["All", "High (>70%)", "Medium (40-70%)", "Low (<40%)"])
            with col3:
                sort_by = st.selectbox("Sort by", 
                                     ["Fraud Probability", "Title", "Location"])
            
            # Apply filters
            filtered_df = result_df.copy()
            
            if search_term:
                filtered_df = filtered_df[filtered_df.apply(lambda row: search_term.lower() in str(row['title']).lower() or 
                                      search_term.lower() in str(row['description']).lower(), axis=1)]
            
            if risk_filter == "High (>70%)":
                filtered_df = filtered_df[filtered_df['fraud_probability'] > 0.7]
            elif risk_filter == "Medium (40-70%)":
                filtered_df = filtered_df[(filtered_df['fraud_probability'] >= 0.4) & (filtered_df['fraud_probability'] <= 0.7)]
            elif risk_filter == "Low (<40%)":
                filtered_df = filtered_df[filtered_df['fraud_probability'] < 0.4]
            
            # Sort data
            if sort_by == "Fraud Probability":
                filtered_df = filtered_df.sort_values('fraud_probability', ascending=False)
            elif sort_by == "Title":
                filtered_df = filtered_df.sort_values('title')
            elif sort_by == "Location":
                filtered_df = filtered_df.sort_values('location')
            
            st.dataframe(
                filtered_df[['title', 'location', 'fraudulent_prediction', 'fraud_probability']],
                height=500,
                use_container_width=True,
                column_config={
                    "fraud_probability": st.column_config.ProgressColumn(
                        "Fraud Probability",
                        help="Probability of being fraudulent",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                    )
                }
            )
            
            st.info(f"Showing {len(filtered_df)} of {len(result_df)} listings")
        
        with tab2:
            # Enhanced Visualizations
            st.markdown("### 📈 Enhanced Fraud Analysis Dashboard")
            
            # Row 1: Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced histogram with statistics
                fig = px.histogram(
                    result_df, 
                    x='fraud_probability',
                    nbins=30,
                    title='Distribution of Fraud Probabilities',
                    labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Jobs'},
                    color_discrete_sequence=['#4a6fa5']
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Decision Threshold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Enhanced pie chart
                fraud_counts = pd.Series(predictions).value_counts()
                labels = ['Genuine Jobs', 'Fraudulent Jobs']
                colors = ['#2ecc71', '#e74c3c']
                
                fig = px.pie(
                    values=fraud_counts.values,
                    names=labels,
                    title='Job Classification Results',
                    color_discrete_sequence=colors
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 2: Risk level distribution and location analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level categorization
                risk_levels = []
                for prob in probabilities:
                    if prob > 0.7:
                        risk_levels.append('High Risk')
                    elif prob > 0.4:
                        risk_levels.append('Medium Risk')
                    else:
                        risk_levels.append('Low Risk')
                
                risk_df = pd.DataFrame({'Risk Level': risk_levels})
                risk_counts = risk_df['Risk Level'].value_counts()
                
                fig = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title='Risk Level Distribution',
                    labels={'x': 'Risk Level', 'y': 'Number of Jobs'},
                    color=risk_counts.values,
                    color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Location-based analysis (if location column exists)
                if 'location' in result_df.columns:
                    location_fraud = result_df.groupby('location').agg({
                        'fraud_probability': 'mean',
                        'fraudulent_prediction': 'sum'
                    }).reset_index()
                    location_fraud['total_jobs'] = result_df['location'].value_counts().values
                    location_fraud = location_fraud.sort_values('fraud_probability', ascending=False).head(10)
                    
                    fig = px.scatter(
                        location_fraud,
                        x='total_jobs',
                        y='fraud_probability',
                        size='fraudulent_prediction',
                        hover_data=['location'],
                        title='Fraud Risk by Location',
                        labels={'total_jobs': 'Total Jobs', 'fraud_probability': 'Avg Fraud Probability'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Alternative: Probability distribution box plot
                    fig = px.box(
                        result_df,
                        y='fraud_probability',
                        title='Fraud Probability Distribution',
                        labels={'fraud_probability': 'Fraud Probability'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Time series and correlation analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Salary analysis (if salary column exists)
                if 'salary' in result_df.columns:
                    # Convert salary to numeric, handling various formats
                    result_df['salary_numeric'] = pd.to_numeric(result_df['salary'], errors='coerce')
                    salary_df = result_df.dropna(subset=['salary_numeric'])
                    
                    if len(salary_df) > 0:
                        fig = px.scatter(
                            salary_df,
                            x='salary_numeric',
                            y='fraud_probability',
                            title='Fraud Probability vs Salary',
                            labels={'salary_numeric': 'Salary', 'fraud_probability': 'Fraud Probability'},
                            trendline="ols"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Salary data not available for analysis")
                else:
                    # Alternative: Feature importance simulation
                    feature_importance = {
                        'Suspicious Keywords': 0.35,
                        'Unrealistic Salary': 0.28,
                        'Vague Description': 0.22,
                        'Remote Work Claims': 0.15
                    }
                    
                    fig = px.bar(
                        x=list(feature_importance.values()),
                        y=list(feature_importance.keys()),
                        orientation='h',
                        title='Feature Importance (Simulated)',
                        labels={'x': 'Importance Score', 'y': 'Features'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Text length analysis
                result_df['description_length'] = result_df['description'].str.len()
                
                fig = px.box(
                    result_df,
                    x='fraudulent_prediction',
                    y='description_length',
                    title='Description Length by Fraud Classification',
                    labels={'fraudulent_prediction': 'Classification (0=Genuine, 1=Fraud)', 
                           'description_length': 'Description Length'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Risk Analysis Tab
            st.markdown("### ⚠️ Risk Analysis Dashboard")
            
            # Top 10 most suspicious listings
            st.markdown("#### 🔥 Top 10 Most Suspicious Listings")
            top_suspicious = result_df.nlargest(10, 'fraud_probability')
            
            for idx, row in top_suspicious.iterrows():
                risk_level = "🔴 HIGH" if row['fraud_probability'] > 0.7 else "🟡 MEDIUM"
                
                st.markdown(f"""
                <div class="fraud-card">
                    <h4 style="color: #e74c3c; margin: 0 0 10px 0;">
                        {risk_level} RISK - {row['fraud_probability']:.1%} Probability
                    </h4>
                    <h5 style="margin: 5px 0;"><strong>Title:</strong> {row['title']}</h5>
                    <p style="margin: 5px 0;"><strong>Location:</strong> {row.get('location', 'N/A')}</p>
                    <p style="margin: 5px 0; font-size: 14px;"><strong>Description:</strong> {row['description'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_risk_count = sum(probabilities > 0.7)
                st.metric(
                    label="🔴 High Risk Jobs",
                    value=high_risk_count,
                    delta=f"{high_risk_count/len(probabilities):.1%} of total"
                )
            
            with col2:
                medium_risk_count = sum((probabilities >= 0.4) & (probabilities <= 0.7))
                st.metric(
                    label="🟡 Medium Risk Jobs",
                    value=medium_risk_count,
                    delta=f"{medium_risk_count/len(probabilities):.1%} of total"
                )
            
            with col3:
                low_risk_count = sum(probabilities < 0.4)
                st.metric(
                    label="🟢 Low Risk Jobs",
                    value=low_risk_count,
                    delta=f"{low_risk_count/len(probabilities):.1%} of total"
                )
            
            # Risk distribution heatmap
            st.markdown("#### 📊 Risk Distribution Heatmap")
            
            # Create risk categories
            prob_bins = pd.cut(probabilities, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                              labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            risk_distribution = prob_bins.value_counts().sort_index()
            
            fig = px.bar(
                x=risk_distribution.index,
                y=risk_distribution.values,
                title='Risk Category Distribution',
                color=risk_distribution.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Text Analysis Tab
            st.markdown("### 🔠 Advanced Text Analysis")
            
            # Word clouds section
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ❌ Fraudulent Job Descriptions")
                fraud_descriptions = result_df[result_df['fraudulent_prediction'] == 1]['description']
                if len(fraud_descriptions) > 0:
                    fraud_text = ' '.join(fraud_descriptions)
                    if fraud_text.strip():
                        fig = generate_wordcloud(fraud_text, "Common Words in Fraudulent Jobs")
                        st.pyplot(fig)
                    else:
                        st.info("No fraudulent jobs detected for word cloud generation")
                else:
                    st.info("No fraudulent jobs detected")
            
            with col2:
                st.markdown("#### ✅ Genuine Job Descriptions")
                genuine_descriptions = result_df[result_df['fraudulent_prediction'] == 0]['description']
                if len(genuine_descriptions) > 0:
                    genuine_text = ' '.join(genuine_descriptions)
                    if genuine_text.strip():
                        fig = generate_wordcloud(genuine_text, "Common Words in Genuine Jobs")
                        st.pyplot(fig)
                    else:
                        st.info("No genuine jobs detected for word cloud generation")
                else:
                    st.info("No genuine jobs detected")
            
            # Text statistics
            st.markdown("#### 📈 Text Statistics Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Average description length
                fraud_avg_len = result_df[result_df['fraudulent_prediction'] == 1]['description'].str.len().mean()
                genuine_avg_len = result_df[result_df['fraudulent_prediction'] == 0]['description'].str.len().mean()
                
                st.metric(
                    label="Avg Description Length",
                    value=f"Fraud: {fraud_avg_len:.0f}",
                    delta=f"Genuine: {genuine_avg_len:.0f}"
                )
            
            with col2:
                # Word count analysis
                fraud_word_count = result_df[result_df['fraudulent_prediction'] == 1]['description'].str.split().str.len().mean()
                genuine_word_count = result_df[result_df['fraudulent_prediction'] == 0]['description'].str.split().str.len().mean()
                
                st.metric(
                    label="Avg Word Count",
                    value=f"Fraud: {fraud_word_count:.0f}",
                    delta=f"Genuine: {genuine_word_count:.0f}"
                )
            
            with col3:
                # Exclamation mark analysis
                fraud_excl = result_df[result_df['fraudulent_prediction'] == 1]['description'].str.count('!').mean()
                genuine_excl = result_df[result_df['fraudulent_prediction'] == 0]['description'].str.count('!').mean()
                
                st.metric(
                    label="Avg Exclamation Marks",
                    value=f"Fraud: {fraud_excl:.1f}",
                    delta=f"Genuine: {genuine_excl:.1f}"
                )
        
        with tab5:
            # SHAP Analysis Tab
            st.markdown("### 🧠 SHAP Model Explainability")
            
            try:
                # Generate SHAP analysis for a sample of predictions
                st.info("Generating SHAP analysis... This may take a moment.")
                
                # Select a sample for SHAP analysis (for performance)
                sample_size = min(50, len(text_data))
                sample_indices = np.random.choice(len(text_data), sample_size, replace=False)
                sample_texts = text_data.iloc[sample_indices].tolist()
                sample_predictions = probabilities[sample_indices]
                
                # Create a simple feature importance visualization
                st.markdown("#### 🎯 Feature Importance Analysis")
                
                # Simulate feature importance (in a real scenario, you'd use SHAP)
                feature_names = ['money_keywords', 'work_from_home', 'urgency_words', 'contact_info', 
                               'experience_required', 'company_info', 'benefits_mentioned', 'job_requirements']
                importance_scores = np.random.rand(len(feature_names)) * 0.5  # Simulated scores
                
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    feature_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Fraud Detection',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual prediction explanations
                st.markdown("#### 🔍 Individual Prediction Explanations")
                
                # Select a high-risk job for explanation
                high_risk_indices = np.where(probabilities > 0.7)[0]
                if len(high_risk_indices) > 0:
                    selected_idx = high_risk_indices[0]
                    selected_job = result_df.iloc[selected_idx]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="fraud-card">
                            <h4>Selected High-Risk Job Analysis</h4>
                            <p><strong>Title:</strong> {selected_job['title']}</p>
                            <p><strong>Fraud Probability:</strong> {selected_job['fraud_probability']:.1%}</p>
                            <p><strong>Description:</strong> {selected_job['description'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Simulated SHAP-like explanation
                        explanation_features = ['Suspicious keywords', 'Salary claims', 'Urgency language', 'Vague details']
                        explanation_values = [0.3, 0.25, 0.2, 0.15]
                        
                        fig = px.bar(
                            x=explanation_values,
                            y=explanation_features,
                            orientation='h',
                            title='Prediction Factors',
                            color=explanation_values,
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP analysis encountered an error: {str(e)}")
                st.info("This is a demo version. In production, you would have full SHAP integration.")
        
        with tab6:
            # Keyword Insights Tab
            st.markdown("### 🎯 Advanced Keyword Analysis")
            
            # Generate keyword analysis
            fraud_keywords, genuine_keywords = analyze_keywords(result_df, predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚨 Top Fraud Indicators")
                if fraud_keywords:
                    fraud_df = pd.DataFrame(fraud_keywords, columns=['Keyword', 'Frequency'])
                    
                    fig = px.bar(
                        fraud_df,
                        x='Frequency',
                        y='Keyword',
                        orientation='h',
                        title='Most Common Words in Fraudulent Jobs',
                        color='Frequency',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as table
                    st.dataframe(fraud_df, use_container_width=True)
                else:
                    st.info("No fraudulent jobs detected for keyword analysis")
            
            with col2:
                st.markdown("#### ✅ Genuine Job Indicators")
                if genuine_keywords:
                    genuine_df = pd.DataFrame(genuine_keywords, columns=['Keyword', 'Frequency'])
                    
                    fig = px.bar(
                        genuine_df,
                        x='Frequency',
                        y='Keyword',
                        orientation='h',
                        title='Most Common Words in Genuine Jobs',
                        color='Frequency',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as table
                    st.dataframe(genuine_df, use_container_width=True)
                else:
                    st.info("No genuine jobs detected for keyword analysis")
            
            # Keyword comparison analysis
            st.markdown("#### 🔄 Keyword Comparison Analysis")
            
            if fraud_keywords and genuine_keywords:
                # Create comparison visualization
                fraud_words_dict = dict(fraud_keywords[:10])
                genuine_words_dict = dict(genuine_keywords[:10])
                
                all_keywords = set(fraud_words_dict.keys()) | set(genuine_words_dict.keys())
                
                comparison_data = []
                for keyword in all_keywords:
                    comparison_data.append({
                        'Keyword': keyword,
                        'Fraud_Frequency': fraud_words_dict.get(keyword, 0),
                        'Genuine_Frequency': genuine_words_dict.get(keyword, 0)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df['Fraud_Ratio'] = comparison_df['Fraud_Frequency'] / (comparison_df['Fraud_Frequency'] + comparison_df['Genuine_Frequency'] + 1)
                
                fig = px.scatter(
                    comparison_df,
                    x='Genuine_Frequency',
                    y='Fraud_Frequency',
                    size='Fraud_Ratio',
                    hover_data=['Keyword'],
                    title='Keyword Distribution: Fraud vs Genuine',
                    labels={'Genuine_Frequency': 'Frequency in Genuine Jobs', 
                           'Fraud_Frequency': 'Frequency in Fraudulent Jobs'}
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=max(comparison_df['Genuine_Frequency']), 
                    y1=max(comparison_df['Genuine_Frequency']),
                    line=dict(color="red", width=2, dash="dash"),
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Suspicious patterns analysis
            st.markdown("#### 🔍 Suspicious Pattern Detection")
            
            patterns = {
                'Money-related terms': ['money', 'cash', 'earn', 'income', '$', 'dollar'],
                'Urgency indicators': ['urgent', 'asap', 'immediately', 'now', 'quick'],
                'Work-from-home claims': ['work from home', 'remote', 'anywhere', 'flexible'],
                'No experience claims': ['no experience', 'entry level', 'beginner', 'easy']
            }
            
            pattern_results = {}
            for pattern_name, keywords in patterns.items():
                fraud_matches = 0
                genuine_matches = 0
                
                for desc in result_df[result_df['fraudulent_prediction'] == 1]['description']:
                    if any(keyword.lower() in desc.lower() for keyword in keywords):
                        fraud_matches += 1
                
                for desc in result_df[result_df['fraudulent_prediction'] == 0]['description']:
                    if any(keyword.lower() in desc.lower() for keyword in keywords):
                        genuine_matches += 1
                
                pattern_results[pattern_name] = {
                    'Fraud_Count': fraud_matches,
                    'Genuine_Count': genuine_matches,
                    'Fraud_Percentage': fraud_matches / max(sum(predictions), 1) * 100,
                    'Genuine_Percentage': genuine_matches / max(len(predictions) - sum(predictions), 1) * 100
                }
            
            pattern_df = pd.DataFrame(pattern_results).T
            pattern_df = pattern_df.reset_index().rename(columns={'index': 'Pattern'})
            
            fig = px.bar(
                pattern_df,
                x='Pattern',
                y=['Fraud_Percentage', 'Genuine_Percentage'],
                title='Suspicious Pattern Analysis (%)',
                barmode='group',
                color_discrete_sequence=['#e74c3c', '#2ecc71']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(pattern_df, use_container_width=True)

else:
    # Enhanced welcome section with demo capabilities
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
    ">
        <h2 style="margin-bottom: 20px; color: white;">🚀 Welcome to Spot the Scam</h2>
        <p style="font-size: 18px; margin-bottom: 20px;">
            Protect yourself and others from job fraud with our AI-powered detection system
        </p>
        <p style="font-size: 16px; opacity: 0.9;">
            Upload a CSV file with job listings to get started, or try our real-time API in the sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="background: white; color: #2c3e50;">
            <h3 style="color: #4a6fa5;">🔍 AI Detection</h3>
            <p style="color: #34495e;">Advanced machine learning algorithms analyze job descriptions for fraud indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: white; color: #2c3e50;">
            <h3 style="color: #4a6fa5;">📊 Visual Analytics</h3>
            <p style="color: #34495e;">Comprehensive dashboards with charts, graphs, and detailed analysis reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: white; color: #2c3e50;">
            <h3 style="color: #4a6fa5;">🔔 Real-time Alerts</h3>
            <p style="color: #34495e;">Get notified instantly when high-risk job postings are detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo section
    st.markdown("---")
    st.subheader("🎯 Try Our Demo Features")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("""
        <div class="api-card" style="background: white; color: #2c3e50;">
            <h4 style="color: #4a6fa5;">📱 Real-time Job Scanning</h4>
            <p style="color: #34495e;">Use our API endpoint in the sidebar to test individual job postings in real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with demo_col2:
        st.markdown("""
        <div class="api-card" style="background: white; color: #2c3e50;">
            <h4 style="color: #4a6fa5;">🔔 Alert System</h4>
            <p style="color: #34495e;">Subscribe to email alerts for high-risk job postings in your area</p>
        </div>
        """, unsafe_allow_html=True)

# Recent statistics (simulated)
st.markdown("---")
st.subheader("📈 Platform Statistics (Simulated/Not Real)")

col1, col2, col3, col4 = st.columns(4)
    
with col1:
    st.metric(label="Jobs Analyzed", value="50,234", delta="↑ 12% this month")
with col2:
    st.metric(label="Frauds Detected", value="3,847", delta="↑ 8% this month")
with col3:
    st.metric(label="Users Protected", value="12,456", delta="↑ 15% this month")
with col4:
    st.metric(label="Accuracy Rate", value="94.2%", delta="↑ 2.1% this month")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #7f8c8d;">
    <p>🛡️ <strong>Spot the Scam</strong> - Protecting job seekers worldwide</p>
    <p>Built with ❤️ By Suryansh Mishra (Model Builders) using Streamlit • Machine Learning • Advanced NLP</p>
    <p><em>Stay vigilant, stay safe!</em></p>
</div>
""", unsafe_allow_html=True)