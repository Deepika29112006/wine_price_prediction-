import streamlit as st
import pandas as pd
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ===============================
# CUSTOM CSS (WINE THEME)
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #3b0000, #7a0000);
}

.main {
    background-color: #fff5f5;
    padding: 30px;
    border-radius: 18px;
}

h1 {
    color: #7a0000;
    text-align: center;
}

.sub {
    text-align: center;
    color: #800000;
    font-size: 18px;
}

.card {
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
    font-size: 18px;
}

.good {
    background-color: #fff4cc;
    border: 2px solid #d4a017;
}

.excellent {
    background-color: #e6ffe6;
    border: 2px solid #2e8b57;
}

.low {
    background-color: #ffe6e6;
    border: 2px solid #8b0000;
}

.footer {
    text-align: center;
    margin-top: 30px;
    color: #7a0000;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.markdown("<h1>🍷 Wine Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Machine Learning based Wine Quality Analysis</div>", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & SCALER
# ===============================
scaler = pickle.load(open("scaler_model.sav", "rb"))
model = pickle.load(open("finalized_RF_model.sav", "rb"))

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("🍇 Wine Chemical Properties")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 20.0, 0.64)
chlorides = st.sidebar.slider("Chlorides", 0.0, 1.0, 0.09)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0.0, 300.0, 15.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.0, 500.0, 98.0)
density = st.sidebar.slider("Density", 0.9, 1.5, 1.0)
pH = st.sidebar.slider("pH", 0.0, 14.0, 3.0)
sulphates = st.sidebar.slider("Sulphates", 0.0, 5.0, 0.68)
alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 5.3)

# ===============================
# INPUT DATA
# ===============================
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# ===============================
# PREDICTION
# ===============================
st.markdown("### 📊 Prediction Result")

if st.button("🍷 Predict Wine Quality"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    quality = int(prediction[0])

    if quality >= 7:
        st.balloons()
        st.markdown(f"""
        <div class="card excellent">
            <h2>🍾 Premium Wine</h2>
            <h3>⭐⭐⭐⭐⭐</h3>
            <p><b>Predicted Quality:</b> {quality}</p>
            <p>This wine has rich aroma, taste and excellent quality.</p>
        </div>
        """, unsafe_allow_html=True)

    elif quality >= 5:
        st.markdown(f"""
        <div class="card good">
            <h2>🍇 Good Quality Wine</h2>
            <h3>⭐⭐⭐⭐</h3>
            <p><b>Predicted Quality:</b> {quality}</p>
            <p>This wine is well balanced and suitable for regular use.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="card low">
            <h2>⚠️ Low Quality Wine</h2>
            <h3>⭐⭐</h3>
            <p><b>Predicted Quality:</b> {quality}</p>
            <p>Wine quality is below average. Improvement is recommended.</p>
        </div>
        """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>👩‍💻 Developed by <b>Deepika</b> | B.Tech Final Year Project</div>", unsafe_allow_html=True)
