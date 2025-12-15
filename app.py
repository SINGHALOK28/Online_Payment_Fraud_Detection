import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    
    .header-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .input-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .result-card {
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fraud-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .safe-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .result-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .result-message {
        font-size: 18px;
        margin-bottom: 20px;
        opacity: 0.95;
    }
    
    .confidence-box {
        background: rgba(255, 255, 255, 0.2);
        padding: 15px 25px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 15px;
    }
    
    .confidence-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    .confidence-value {
        font-size: 28px;
        font-weight: bold;
    }
    
    .summary-table {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
    }
    
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #eee;
    }
    
    .summary-row:last-child {
        border-bottom: none;
    }
    
    .summary-label {
        font-weight: 600;
        color: #667eea;
    }
    
    .summary-value {
        color: #333;
        font-weight: 500;
    }
    
    .predict-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 12px 40px !important;
        border-radius: 10px !important;
        border: none !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .predict-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox, .stNumberInput {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1 style='font-size: 48px; margin-bottom: 10px;'>🔐 Fraud Detection</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 16px;'>Real-time Payment Security Check</p>
</div>
""", unsafe_allow_html=True)

# Load or train models with efficiency comparison
@st.cache_resource
def load_models():
    try:
        # Try to load real data
        data = pd.read_csv("onlinefraud.csv")
        
        # Encode transaction types
        type_mapping = {
            'CASH_IN': 1,
            'CASH_OUT': 2,
            'DEBIT': 3,
            'PAYMENT': 4,
            'TRANSFER': 5
        }
        
        data['type'] = data['type'].map(type_mapping)
        
        # Prepare features and target
        X = data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']].fillna(0)
        y = data['isFraud']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        models = {}
        metrics = {}
        
        # Model 1: Decision Tree
        start = time.time()
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=15)
        dt_model.fit(X_scaled, y)
        dt_time = time.time() - start
        dt_pred = dt_model.predict(X_scaled)
        models['Decision Tree'] = dt_model
        metrics['Decision Tree'] = {
            'accuracy': accuracy_score(y, dt_pred),
            'precision': precision_score(y, dt_pred, zero_division=0),
            'recall': recall_score(y, dt_pred, zero_division=0),
            'f1': f1_score(y, dt_pred, zero_division=0),
            'train_time': dt_time
        }
        
        # Model 2: Random Forest
        start = time.time()
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y)
        rf_time = time.time() - start
        rf_pred = rf_model.predict(X_scaled)
        models['Random Forest'] = rf_model
        metrics['Random Forest'] = {
            'accuracy': accuracy_score(y, rf_pred),
            'precision': precision_score(y, rf_pred, zero_division=0),
            'recall': recall_score(y, rf_pred, zero_division=0),
            'f1': f1_score(y, rf_pred, zero_division=0),
            'train_time': rf_time
        }
        
        # Model 3: Logistic Regression
        start = time.time()
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr_model.fit(X_scaled, y)
        lr_time = time.time() - start
        lr_pred = lr_model.predict(X_scaled)
        models['Logistic Regression'] = lr_model
        metrics['Logistic Regression'] = {
            'accuracy': accuracy_score(y, lr_pred),
            'precision': precision_score(y, lr_pred, zero_division=0),
            'recall': recall_score(y, lr_pred, zero_division=0),
            'f1': f1_score(y, lr_pred, zero_division=0),
            'train_time': lr_time
        }
        
        return models, scaler, metrics, 'real_data'
    
    except:
        # Create synthetic training data based on fraud patterns
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic features
        types = np.random.randint(1, 6, n_samples)
        amounts = np.random.exponential(5000, n_samples)
        old_balance = np.random.exponential(10000, n_samples)
        new_balance = old_balance - amounts
        new_balance = np.maximum(new_balance, 0)
        
        X = np.column_stack([types, amounts, old_balance, new_balance])
        
        # Generate fraud labels based on patterns
        y = []
        for i in range(n_samples):
            fraud_score = 0
            
            # Pattern 1: CASH_IN with decreased balance
            if types[i] == 1 and new_balance[i] < old_balance[i]:
                fraud_score += 0.5
            
            # Pattern 2: CASH_OUT/DEBIT amount exceeds original balance
            if types[i] in [2, 3] and amounts[i] > old_balance[i]:
                fraud_score += 0.6
            
            # Pattern 3: Mathematical inconsistency
            expected_balance_out = old_balance[i] - amounts[i]
            expected_balance_in = old_balance[i] + amounts[i]
            
            if types[i] in [2, 3, 4, 5]:
                if abs(new_balance[i] - expected_balance_out) > amounts[i] * 0.05:
                    fraud_score += 0.45
            elif types[i] == 1:
                if abs(new_balance[i] - expected_balance_in) > amounts[i] * 0.05:
                    fraud_score += 0.45
            
            # Pattern 4: Zero new balance after payment
            if new_balance[i] < 0.01 and types[i] == 4:
                fraud_score += 0.4
            
            # Pattern 5: Very large amount vs balance
            if amounts[i] > old_balance[i] * 0.9 and types[i] in [2, 5]:
                fraud_score += 0.3
            
            # Pattern 6: CASH_OUT with suspicious amount
            if types[i] == 2 and amounts[i] > 20000:
                fraud_score += 0.2
            
            # Pattern 7: Transfer with unusual characteristics
            if types[i] == 5 and amounts[i] > old_balance[i] * 0.8:
                fraud_score += 0.15
            
            # Pattern 8: Negative balance
            if new_balance[i] < 0:
                fraud_score += 0.6
            
            y.append(1 if fraud_score > 0.4 else 0)
        
        y = np.array(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        models = {}
        metrics = {}
        
        # Model 1: Decision Tree
        start = time.time()
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=15)
        dt_model.fit(X_scaled, y)
        dt_time = time.time() - start
        dt_pred = dt_model.predict(X_scaled)
        models['Decision Tree'] = dt_model
        metrics['Decision Tree'] = {
            'accuracy': accuracy_score(y, dt_pred),
            'precision': precision_score(y, dt_pred, zero_division=0),
            'recall': recall_score(y, dt_pred, zero_division=0),
            'f1': f1_score(y, dt_pred, zero_division=0),
            'train_time': dt_time
        }
        
        # Model 2: Random Forest
        start = time.time()
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y)
        rf_time = time.time() - start
        rf_pred = rf_model.predict(X_scaled)
        models['Random Forest'] = rf_model
        metrics['Random Forest'] = {
            'accuracy': accuracy_score(y, rf_pred),
            'precision': precision_score(y, rf_pred, zero_division=0),
            'recall': recall_score(y, rf_pred, zero_division=0),
            'f1': f1_score(y, rf_pred, zero_division=0),
            'train_time': rf_time
        }
        
        # Model 3: Logistic Regression
        start = time.time()
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr_model.fit(X_scaled, y)
        lr_time = time.time() - start
        lr_pred = lr_model.predict(X_scaled)
        models['Logistic Regression'] = lr_model
        metrics['Logistic Regression'] = {
            'accuracy': accuracy_score(y, lr_pred),
            'precision': precision_score(y, lr_pred, zero_division=0),
            'recall': recall_score(y, lr_pred, zero_division=0),
            'f1': f1_score(y, lr_pred, zero_division=0),
            'train_time': lr_time
        }
        
        return models, scaler, metrics, 'synthetic_data'

models, scaler, model_metrics, data_source = load_models()

# Model selection sidebar
st.sidebar.markdown("### ⚙️ Model Configuration")
selected_model = st.sidebar.selectbox(
    "Choose ML Model:",
    options=list(models.keys()),
    help="Select the machine learning model for fraud detection"
)

# Display model efficiency metrics
with st.sidebar.expander("📊 Model Performance Metrics", expanded=True):
    st.markdown(f"**Training Data:** {data_source.replace('_', ' ').title()}")
    st.markdown("---")
    
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Show best model recommendation
    best_f1 = metrics_df['f1'].idxmax()
    best_speed = metrics_df['train_time'].idxmin()
    best_accuracy = metrics_df['accuracy'].idxmax()
    
    st.markdown("---")
    st.markdown(f"🏆 **Best F1 Score:** {best_f1}")
    st.markdown(f"⚡ **Fastest:** {best_speed}")
    st.markdown(f"🎯 **Best Accuracy:** {best_accuracy}")

# Get selected model
current_model = models[selected_model]

# Main container
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Input Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    st.subheader("📝 Enter Transaction Details")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        transaction_type = st.selectbox(
            "Type of Transaction",
            options=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            key="type_select"
        )
    
    with col_b:
        amount = st.number_input(
            "Amount (₹)",
            min_value=0.0,
            value=1000.0,
            step=50.0,
            key="amount_input"
        )
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        old_balance = st.number_input(
            "Original Balance (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            key="old_balance_input"
        )
    
    # Auto-calculate new_balance based on transaction type
    if transaction_type == "CASH_IN":
        calculated_balance = old_balance + amount
        help_text = "Auto-calculated: Original Balance + Amount"
    elif transaction_type == "CASH_OUT":
        calculated_balance = max(0, old_balance - amount)  # Prevent negative balance
        help_text = "Auto-calculated: Original Balance - Amount"
    elif transaction_type == "DEBIT":
        calculated_balance = max(0, old_balance - amount)
        help_text = "Auto-calculated: Original Balance - Amount"
    elif transaction_type == "PAYMENT":
        calculated_balance = max(0, old_balance - amount)
        help_text = "Auto-calculated: Original Balance - Amount"
    elif transaction_type == "TRANSFER":
        calculated_balance = max(0, old_balance - amount)
        help_text = "Auto-calculated: Original Balance - Amount"
    
    with col_d:
        new_balance = st.number_input(
            "New Balance (₹)",
            min_value=0.0,
            value=calculated_balance,
            step=100.0,
            help=help_text,
            key="new_balance_input"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show calculation info
    st.info(f"📌 **{transaction_type}**: {old_balance:,.0f} {'+ ' if transaction_type == 'CASH_IN' else '- '} {amount:,.0f} = **{calculated_balance:,.0f}** (Calculated)")
    
    # Predict Button
    st.markdown('<br>', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Analyze Transaction", use_container_width=True, key="predict_btn")
    
    # Process prediction
    if predict_clicked:
        # Map transaction types
        type_mapping = {
            'CASH_IN': 1,
            'CASH_OUT': 2,
            'DEBIT': 3,
            'PAYMENT': 4,
            'TRANSFER': 5
        }
        
        try:
            # Prepare input
            type_encoded = type_mapping[transaction_type]
            input_features = np.array([[type_encoded, amount, old_balance, new_balance]])
            
            # Scale input using the same scaler
            input_scaled = scaler.transform(input_features)
            
            # Make prediction
            prediction = current_model.predict(input_scaled)[0]
            
            # Get confidence - handle both models that have predict_proba and those that don't
            try:
                confidence = current_model.predict_proba(input_scaled)[0]
                fraud_confidence = confidence[1] * 100
                safe_confidence = confidence[0] * 100
            except:
                # For models without predict_proba, use decision function
                fraud_confidence = 50
                safe_confidence = 50
            
            # Result Card
            if prediction == 1:
                st.markdown("""
                <div class="result-card fraud-result">
                    <div class="result-title">⚠️ FRAUD DETECTED</div>
                    <div class="result-message">This transaction appears to be fraudulent</div>
                    <div class="confidence-box">
                        <div class="confidence-label">FRAUD PROBABILITY</div>
                        <div class="confidence-value">{:.1f}%</div>
                    </div>
                </div>
                """.format(fraud_confidence), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card safe-result">
                    <div class="result-title">✅ TRANSACTION SAFE</div>
                    <div class="result-message">This transaction appears to be legitimate</div>
                    <div class="confidence-box">
                        <div class="confidence-label">SAFETY CONFIDENCE</div>
                        <div class="confidence-value">{:.1f}%</div>
                    </div>
                </div>
                """.format(safe_confidence), unsafe_allow_html=True)
            
            # Summary with model info
            st.markdown("""
            <div class="summary-table">
                <div class="summary-row">
                    <span class="summary-label">🤖 Model Used</span>
                    <span class="summary-value">{}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Transaction Type</span>
                    <span class="summary-value">{}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Amount</span>
                    <span class="summary-value">₹{:,.2f}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Original Balance</span>
                    <span class="summary-value">₹{:,.2f}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">New Balance</span>
                    <span class="summary-value">₹{:,.2f}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Result</span>
                    <span class="summary-value">{}</span>
                </div>
            </div>
            """.format(
                selected_model,
                transaction_type,
                amount,
                old_balance,
                new_balance,
                "Fraudulent ⚠️" if prediction == 1 else "Legitimate ✅"
            ), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 40px; color: rgba(255,255,255,0.8); font-size: 12px;'>
        <p>Powered by Machine Learning | For Educational Purposes</p>
    </div>
    """, unsafe_allow_html=True)
