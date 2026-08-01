# 🛡️ AI Fraud & Spam Guard

An interactive, premium Streamlit dashboard that detects spam and fraudulent emails in real-time using trained machine learning classifiers. Features a dark-glass glassmorphism design, real-time feature extraction, and model analytics.

---

## 🚀 Features

- 🤖 **Multiple ML Classifiers**: Support for Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.
- 📐 **StandardScaler Pipeline**: Features are automatically scaled to ensure consistent, highly accurate, and balanced predictions across distance-based models.
- 🔍 **Real-Time Feature Parsing**: Dynamically parses entered emails for character count, word count, exclamation marks, suspicious URLs, money terms, and urgency cues.
- 🎨 **Premium Dark UI**: A beautiful user interface using Outfit & Inter typography, glassmorphism cards, and alert banners.
- 📊 **Model & Dataset Analytics**: Interactive Plotly charts showing classifier accuracy comparisons, target class distributions, confusion matrices, and feature correlations.

---

## 🛠️ Tech Stack

- **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (SVM, Random Forest, KNN, Naive Bayes, StandardScaler)
- **Data Visualization**: [Plotly Express & Graph Objects](https://plotly.com/)

---

## 📥 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/priyanshijasoliya-hub/spam-detection-app.git
   cd spam-detection-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run spam_detc_app.py
   ```

---

## 📂 Repository Structure

- `spam_detc_app.py`: Main Streamlit interactive dashboard.
- `spam_detc.py`: Standard standalone training script.
- `spam_detc_data.csv` / `spam_detc.csv`: Dataset with structural features and email labels.
- `requirements.txt`: Python package requirements.
