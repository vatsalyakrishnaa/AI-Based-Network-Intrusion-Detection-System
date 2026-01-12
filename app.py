import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---- OPTIONAL GROQ IMPORT (SAFE) ----
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ModuleNotFoundError:
    GROQ_AVAILABLE = False

# ---- PAGE SETUP ----
st.set_page_config(page_title="AI-Based NIDS", layout="wide")

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**

This system uses:
- **Random Forest** for attack detection  
- **Groq LLM (optional)** for explanation of predictions  
""")

# ---- CONFIG ----
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ---- SIDEBAR ----
st.sidebar.header("1️⃣ Settings")

groq_api_key = st.sidebar.text_input(
    "Groq API Key (optional)", 
    type="password",
    disabled=not GROQ_AVAILABLE
)

if not GROQ_AVAILABLE:
    st.sidebar.warning("Groq not installed. AI explanations disabled.")

st.sidebar.header("2️⃣ Model Training")

# ---- DATA LOADING ----
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, nrows=15000)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# ---- MODEL TRAINING ----
def train_model(df):
    features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Fwd Packet Length Max',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Flow Packets/s'
    ]

    target = 'Label'

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, X_test, y_test

# ---- LOAD DATA ----
df = load_data(DATA_FILE)

if df is None:
    st.error(f"Dataset '{DATA_FILE}' not found.")
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

# ---- TRAIN BUTTON ----
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        result = train_model(df)
        if result:
            model, acc, X_test, y_test = result
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.sidebar.success(f"Training complete! Accuracy: {acc:.2%}")

# ---- DASHBOARD ----
st.header("3️⃣ Threat Analysis Dashboard")

if "model" not in st.session_state:
    st.info("Train the model to begin analysis.")
    st.stop()

col1, col2 = st.columns(2)

# ---- SIMULATION ----
with col1:
    st.subheader("Packet Simulation")

    if st.button("🎲 Capture Random Packet"):
        idx = np.random.randint(0, len(st.session_state.X_test))
        st.session_state.packet = st.session_state.X_test.iloc[idx]
        st.session_state.actual = st.session_state.y_test.iloc[idx]

    if "packet" in st.session_state:
        st.dataframe(st.session_state.packet, use_container_width=True)

# ---- PREDICTION ----
with col2:
    if "packet" in st.session_state:
        packet = st.session_state.packet
        prediction = st.session_state.model.predict([packet])[0]

        st.subheader("Detection Result")

        if prediction == "BENIGN":
            st.success("✅ SAFE (BENIGN)")
        else:
            st.error(f"🚨 ATTACK DETECTED ({prediction})")

        st.caption(f"Ground Truth: {st.session_state.actual}")

        # ---- GROQ EXPLANATION ----
        st.markdown("---")
        st.subheader("🤖 AI Explanation (Optional)")

        if st.button("Generate Explanation"):
            if not GROQ_AVAILABLE:
                st.warning("Groq library not installed.")
            elif not groq_api_key:
                st.warning("Please enter Groq API key.")
            else:
                try:
                    client = Groq(api_key=groq_api_key)

                    prompt = f"""
                    You are a cybersecurity analyst.

                    Prediction: {prediction}

                    Packet Details:
                    {packet.to_string()}

                    Explain briefly why this traffic looks {prediction}.
                    """

                    with st.spinner("Analyzing..."):
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6
                        )

                        st.info(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Groq Error: {e}")
