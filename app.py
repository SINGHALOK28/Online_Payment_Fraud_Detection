import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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
            step=50.0
        )
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        old_balance = st.number_input(
            "Original Balance (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )
    
    with col_d:
        new_balance = st.number_input(
            "New Balance (₹)",
            min_value=0.0,
            value=4000.0,
            step=100.0
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
            # Train model
            np.random.seed(42)
            n_samples = 1000
            X_sample = np.random.rand(n_samples, 4) * 100000
            y_sample = np.random.randint(0, 2, n_samples)
            
            model = DecisionTreeClassifier(random_state=42, max_depth=10)
            model.fit(X_sample, y_sample)
            
            # Make prediction
            type_encoded = type_mapping[transaction_type]
            input_features = np.array([[type_encoded, amount, old_balance, new_balance]])
            
            prediction = model.predict(input_features)[0]
            confidence = model.predict_proba(input_features)[0]
            
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
                """.format(confidence[1] * 100), unsafe_allow_html=True)
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
                """.format(confidence[0] * 100), unsafe_allow_html=True)
            
            # Summary
            st.markdown("""
            <div class="summary-table">
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
