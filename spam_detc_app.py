import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import re
import os

# Set page configuration with a premium look
st.set_page_config(
    page_title="AI Fraud & Spam Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphic styling and dark mode gradients
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at top right, #141124, #0b0914);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title styling */
    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #c084fc, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }
    
    /* Translucent glass card */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Result Banners */
    .valid-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.2));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.8rem;
        color: #e2e8f0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
    }
    .spam-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.2));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.8rem;
        color: #e2e8f0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.15);
    }
    
    /* Metric label styling */
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Robust styles for all inputs, textareas and their text colors */
    input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
        color: #ffffff !important;
        background-color: #1e1b33 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Container styling for input fields and textareas */
    [data-baseweb="input"], [data-baseweb="textarea"], 
    .stTextInput > div > div, .stTextArea > div > div {
        background-color: #1e1b33 !important;
        border: 2px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
    }

    /* Style the labels for all inputs/textareas to make sure they are bright and clear */
    label, [data-testid="stWidgetLabel"] p, .stWidgetLabel {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
DATA_PATH = "spam_detc_data.csv"

# 1. Load Data
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, sep="\t")
    else:
        # Fallback to absolute workspace location if launched from a parent path
        alt_path = os.path.join(os.path.dirname(__file__), "spam_detc_data.csv")
        if os.path.exists(alt_path):
            return pd.read_csv(alt_path, sep="\t")
        else:
            st.error(f"Dataset not found at '{DATA_PATH}'. Please ensure 'spam_detc_data.csv' is in the directory.")
            return None

df_raw = load_data()

# 2. Train and cache models
@st.cache_resource
def train_and_cache_models():
    df = load_data()
    if df is None:
        return None, None, None, None
    
    # Preprocess
    df = df.dropna()
    
    feature_cols = [
        'num_words', 'num_characters', 'num_exclamation_marks', 'num_links', 
        'has_suspicious_link', 'num_attachments', 'has_attachment', 
        'sender_reputation_score', 'email_hour', 'email_day_of_week', 
        'is_weekend', 'num_recipients', 'contains_money_terms', 'contains_urgency_terms'
    ]
        
    X = df[feature_cols]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': svm_model,
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred) * 100
        trained_models[name] = model
        accuracies[name] = acc
        
    return trained_models, accuracies, scaler, (X_test_scaled, y_test, feature_cols)

# Define word lists for text extraction
MONEY_KEYWORDS = ["win", "cash", "free", "money", "prize", "claim", "offer", "budget", "guarantee"]
URGENCY_KEYWORDS = ["urgent", "now", "limited", "hurry", "expire", "action", "today", "immediately", "deadline", "fast"]

def extract_text_features(subject, body):
    full_text = f"{subject} {body}"
    # Standardize spaces and lower case
    cleaned_text = full_text.lower()
    
    # Count words and characters
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text)
    num_words = len(words)
    num_characters = len(full_text)
    
    # Count exclamations
    num_exclamations = full_text.count('!')
    
    # Find links
    links = re.findall(r'https?://\S+|www\.\S+|\b\w+\.(?:com|org|net|edu|gov|xyz|biz|cc|ru|info)\b', cleaned_text)
    num_links = len(links)
    
    # Suspicious domains flag
    suspicious_domains = ['.xyz', '.biz', '.cc', '.ru', 'freemoney', 'chealdealz', 'winbignow', 'unknownmail']
    has_suspicious_link = 0
    for link in links:
        if any(dom in link for dom in suspicious_domains):
            has_suspicious_link = 1
            break
            
    # Key terms detection
    contains_money_terms = 1 if any(w in words for w in MONEY_KEYWORDS) else 0
    detected_money_words = [w for w in MONEY_KEYWORDS if w in words]
    
    contains_urgency_terms = 1 if any(w in words for w in URGENCY_KEYWORDS) else 0
    detected_urgency_words = [w for w in URGENCY_KEYWORDS if w in words]
    
    return {
        'num_words': num_words,
        'num_characters': num_characters,
        'num_exclamations': num_exclamations,
        'num_links': num_links,
        'has_suspicious_link': has_suspicious_link,
        'contains_money_terms': contains_money_terms,
        'contains_urgency_terms': contains_urgency_terms,
        'detected_money': list(set(detected_money_words)),
        'detected_urgency': list(set(detected_urgency_words))
    }

# --- Sidebar Controls ---
st.sidebar.markdown("<h2 style='font-family: Outfit; color: #a855f7;'>⚙️ Model Settings</h2>", unsafe_allow_html=True)

# Load / Train Models
with st.spinner("Training models... Please wait a moment."):
    models, accuracies, scaler, test_data = train_and_cache_models()

# Select Model
selected_model_name = st.sidebar.selectbox(
    "Select Classifier Model:",
    list(models.keys()) if models else []
)

if models and selected_model_name:
    model_acc = accuracies[selected_model_name]
    st.sidebar.markdown(f"""
    <div class="glass-card" style="margin-top: 10px; padding: 1rem; border-color: rgba(168, 85, 247, 0.3);">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value" style="color: #a855f7;">{model_acc:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Additional advanced inputs
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='font-family: Outfit; font-size: 1.1rem; color: #6366f1;'>📥 Extra Email Metadata</h3>", unsafe_allow_html=True)

# Let users tweak settings
sender_email = st.sidebar.text_input("Sender Email Address:", "support@winbignow.ru")

# Auto-flag sender reputation if suspicious domain
suspicious_domains = ['.xyz', '.biz', '.cc', '.ru', 'freemoney', 'chealdealz', 'winbignow', 'unknownmail']
default_reputation = 0.75
if any(dom in sender_email.lower() for dom in suspicious_domains):
    default_reputation = 0.40
    st.sidebar.caption("⚠️ Flagged: Suspicious sender domain detected. Auto-lowered reputation score.")
    
sender_reputation_score = st.sidebar.slider(
    "Sender Reputation Score:",
    min_value=0.0, max_value=1.0, value=default_reputation, step=0.05,
    help="Higher reputation sender means valid source (e.g. Gmail/Outlook is higher, unknown new domains are lower)."
)

num_attachments = st.sidebar.slider("Number of Attachments:", 0, 5, 0)
has_attachment = 1 if num_attachments > 0 else 0

num_recipients = st.sidebar.slider("Number of Recipients:", 1, 100, 1)

email_hour = st.sidebar.slider("Hour Received (0-23):", 0, 23, 12)

email_day = st.sidebar.selectbox(
    "Day of Week Received:",
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    index=0
)

day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
email_day_of_week = day_mapping[email_day]
is_weekend = 1 if email_day in ("Saturday", "Sunday") else 0

# --- Main Layout ---
st.markdown("<div class='title-text'>🛡️ AI Fraud & Spam Guard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Analyze and predict fraudulent or spam emails in real-time using machine learning classifiers</div>", unsafe_allow_html=True)

tab_pred, tab_analytics = st.tabs(["📩 Interactive Predictor", "📊 Analytics & Model Insights"])

with tab_pred:
    st.markdown("### Enter Email Content")
    
    # Set default values for testing
    default_subject = "🔴 URGENT: Claim your $1000 Cash Prize Now!!"
    default_body = "Congratulations! You have been selected to win a free cash gift card of $1000. This limited time offer is guarantee valid only for today! Click this link www.freemoney.biz to claim your prize now before it expires!"
    
    col_sub, col_body = st.columns([1, 2])
    with col_sub:
        subject_input = st.text_input("Subject:", value=default_subject)
    with col_body:
        body_input = st.text_area("Email Content:", value=default_body, height=120)
        
    btn_predict = st.button("Run Security Scan", type="primary")
    
    # Process text features
    parsed = extract_text_features(subject_input, body_input)
    
    # Show real-time parsed characteristics
    st.markdown("#### 🔍 Real-time Extracted Features")
    char_cols = st.columns(4)
    with char_cols[0]:
        st.metric("Characters Count", parsed['num_characters'])
    with char_cols[1]:
        st.metric("Words Count", parsed['num_words'])
    with char_cols[2]:
        st.metric("Exclamation Marks", parsed['num_exclamations'])
    with char_cols[3]:
        st.metric("Links Count", parsed['num_links'])
        
    # Badges for keywords
    if parsed['detected_money'] or parsed['detected_urgency']:
        st.markdown("**Detected Terms:**")
        badge_html = ""
        for w in parsed['detected_money']:
            badge_html += f"<span style='background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid #eab308; border-radius: 4px; padding: 2px 8px; margin-right: 5px; font-size: 0.85rem;'>💰 Money: {w}</span>"
        for w in parsed['detected_urgency']:
            badge_html += f"<span style='background: rgba(236, 72, 153, 0.15); color: #ec4899; border: 1px solid #ec4899; border-radius: 4px; padding: 2px 8px; margin-right: 5px; font-size: 0.85rem;'>⏳ Urgency: {w}</span>"
        st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if btn_predict:
        if not subject_input.strip() and not body_input.strip():
            st.error("Please enter a subject or email content to analyze.")
        else:
            # Build feature vector (14 features order must match training data)
            input_vector = pd.DataFrame([{
                'num_words': parsed['num_words'],
                'num_characters': parsed['num_characters'],
                'num_exclamation_marks': parsed['num_exclamations'],
                'num_links': parsed['num_links'],
                'has_suspicious_link': parsed['has_suspicious_link'], # auto-extracted
                'num_attachments': num_attachments,
                'has_attachment': has_attachment,
                'sender_reputation_score': sender_reputation_score,
                'email_hour': email_hour,
                'email_day_of_week': email_day_of_week,
                'is_weekend': is_weekend,
                'num_recipients': num_recipients,
                'contains_money_terms': parsed['contains_money_terms'],
                'contains_urgency_terms': parsed['contains_urgency_terms']
            }])
                
            # Scale feature vector
            input_vector_scaled = scaler.transform(input_vector)

            # Perform prediction
            model = models[selected_model_name]
            pred = model.predict(input_vector_scaled)[0]
            
            # Predict probability if supported
            try:
                prob = model.predict_proba(input_vector_scaled)[0]
                confidence = prob[pred] * 100
            except:
                confidence = None
                
            # Displays
            st.markdown("### 🔍 Security Scanner Result")
            if pred == 1:
                # Spam/Fraud
                st.markdown(f"""
                <div class="spam-banner">
                    <h2 style="color: #ef4444; margin-top: 0;">🚨 SPAM / FRAUD DETECTED</h2>
                    <p style="font-size: 1.15rem; margin-bottom: 5px;">This email matches fraud and spam indicators. Treat this email with caution.</p>
                    {f'<p style="font-size: 1.5rem; font-weight: bold; color: #fca5a5; margin: 0;">Risk Confidence: {confidence:.2f}%</p>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic suggestions on why it failed
                st.markdown("#### 🛑 Security Alerts Triggered:")
                alerts = []
                if parsed['contains_money_terms'] == 1:
                    alerts.append("Contains high-pressure **money/financial** promotional vocabulary.")
                if parsed['contains_urgency_terms'] == 1:
                    alerts.append("Contains **urgency cues** attempting to force rapid compliance.")
                if parsed['num_exclamations'] > 2:
                    alerts.append(f"Excessive use of exclamation marks (**{parsed['num_exclamations']}** detected).")
                
                if sender_reputation_score < 0.6:
                    alerts.append(f"Sender reputation score is low (**{sender_reputation_score:.2f}**).")
                if parsed['has_suspicious_link'] == 1:
                    alerts.append("Contains **suspicious links** with domains known for spoofing/phishing.")
                if num_recipients > 10:
                    alerts.append(f"Sent to a high number of recipients (**{num_recipients}**).")
                        
                if not alerts:
                    alerts.append("The email layout and lengths strongly align with historical spam patterns.")
                    
                for alert in alerts:
                    st.markdown(f"- {alert}")
            else:
                # Safe
                st.markdown(f"""
                <div class="valid-banner">
                    <h2 style="color: #10b981; margin-top: 0;">✅ SAFE / VALID EMAIL</h2>
                    <p style="font-size: 1.15rem; margin-bottom: 5px;">This email shows normal patterns. No critical security triggers were tripped.</p>
                    {f'<p style="font-size: 1.5rem; font-weight: bold; color: #6ee7b7; margin: 0;">Safe Confidence: {confidence:.2f}%</p>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
                
            # Extracted feature vector details
            with st.expander("🔬 View Extracted Classifier Input Vector"):
                st.dataframe(input_vector)

with tab_analytics:
    st.markdown("### 📊 Model Performance & Dataset Insights")
    
    if df_raw is not None:
        # Layout columns
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### Classifier Model Accuracy Comparison")
            
            # Bar chart comparing accuracies
            df_acc = pd.DataFrame({
                'Model': list(accuracies.keys()),
                'Accuracy (%)': list(accuracies.values())
            }).sort_values(by='Accuracy (%)', ascending=False)
            
            fig = px.bar(
                df_acc, 
                x='Model', 
                y='Accuracy (%)',
                color='Accuracy (%)',
                color_continuous_scale=px.colors.sequential.Purples,
                text='Accuracy (%)',
                template="plotly_dark"
            )
            fig.update_layout(
                yaxis_range=[50, 100],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=300
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_m2:
            st.markdown("#### Dataset Class Distribution (Spam vs. Valid)")
            
            # Pie chart of target label
            label_counts = df_raw['label'].value_counts().reset_index()
            label_counts.columns = ['Status', 'Count']
            label_counts['Status'] = label_counts['Status'].map({0: 'Valid Email', 1: 'Spam/Fraud'})
            
            fig_pie = px.pie(
                label_counts, 
                values='Count', 
                names='Status',
                color='Status',
                color_discrete_map={'Valid Email': '#10b981', 'Spam/Fraud': '#ef4444'},
                hole=0.4,
                template="plotly_dark"
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Confusion Matrix for the currently selected model
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        st.markdown(f"#### 🧩 Confusion Matrix — {selected_model_name}")

        X_test_cm, y_test_cm, _ = test_data
        y_pred_cm = models[selected_model_name].predict(X_test_cm)
        cm = confusion_matrix(y_test_cm, y_pred_cm)
        cm_labels = ['Valid Email', 'Spam/Fraud']

        col_cm1, col_cm2 = st.columns([1.3, 1])
        with col_cm1:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=cm_labels,
                y=cm_labels,
                colorscale='Purples',
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 22, "color": "white"},
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                yaxis=dict(autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                margin=dict(l=20, r=20, t=30, b=20),
                height=320
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm2:
            st.markdown("<br>", unsafe_allow_html=True)
            tn, fp, fn, tp = cm.ravel()
            m1, m2 = st.columns(2)
            with m1:
                st.metric("True Negative", tn, help="Valid email correctly identified as Valid")
                st.metric("False Negative", fn, help="Spam wrongly marked as Valid — the riskiest miss")
            with m2:
                st.metric("False Positive", fp, help="Valid email wrongly flagged as Spam")
                st.metric("True Positive", tp, help="Spam/Fraud correctly caught")

        # Extra stats
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        st.markdown("#### Feature Correlates of Spam")
        
        col_m3, col_m4 = st.columns(2)
        with col_m3:
            # Word Count Distribution by Label
            st.markdown("##### Word Count Distribution")
            df_words = df_raw.copy()
            df_words['Label'] = df_words['label'].map({0: 'Valid', 1: 'Spam'})
            
            fig_box = px.box(
                df_words, 
                x='Label', 
                y='num_words',
                color='Label',
                color_discrete_map={'Valid': '#10b981', 'Spam': '#ef4444'},
                template="plotly_dark",
                labels={'num_words': 'Word Count'}
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=250
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_m4:
            # Sender Reputation vs Label
            st.markdown("##### Sender Reputation Scores by Label")
            fig_reputation = px.histogram(
                df_words,
                x='sender_reputation_score',
                color='Label',
                barmode='overlay',
                color_discrete_map={'Valid': '#10b981', 'Spam': '#ef4444'},
                template="plotly_dark",
                labels={'sender_reputation_score': 'Sender Reputation'}
            )
            fig_reputation.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=250
            )
            st.plotly_chart(fig_reputation, use_container_width=True)
            
        # Display dataset table preview
        with st.expander("📂 View Raw Dataset Preview (First 50 records)"):
            st.dataframe(df_raw.head(50))