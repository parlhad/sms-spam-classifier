import streamlit as st
import pickle
import string
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data for Streamlit Cloud
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SpamShield | Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .78; font-size: 1.05rem;}
    .result {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.25);
        margin-top: .8rem;
    }
    .result h2 {margin: 0 0 .35rem 0;}
    .small {font-size: .88rem; opacity: .72;}
    .chip {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.28);
        margin-right: .35rem;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------
@st.cache_resource
def setup_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

    # Newer NLTK releases may require punkt_tab.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    return stopwords.words("english")


STOP_WORDS = setup_nltk()
PS = PorterStemmer()

# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"



def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl was not found.")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError("vectorizer.pkl was not found.")

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


try:
    tfidf, model = load_artifacts()
except Exception as e:
    st.error(f"Could not load the ML model files: {e}")
    st.info("Keep model.pkl, vectorizer.pkl, and app.py in the same project folder.")
    st.stop()

# ---------------------------------------------------------
# Exact training-time preprocessing
# ---------------------------------------------------------
def transform_text(text: str) -> str:
    """
    Matches the preprocessing used while training the uploaded model:
    lowercase -> NLTK tokenization -> alphanumeric tokens ->
    English stopword/punctuation removal -> Porter stemming.
    """
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)

    filtered = [token for token in tokens if token.isalnum()]
    filtered = [
        token
        for token in filtered
        if token not in STOP_WORDS and token not in string.punctuation
    ]

    stemmed = [PS.stem(token) for token in filtered]
    return " ".join(stemmed)


def predict_message(message: str):
    transformed = transform_text(message)
    vector = tfidf.transform([transformed])

    prediction = int(model.predict(vector)[0])

    probability = None
    spam_probability = None
    ham_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        class_to_probability = {
            int(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
        spam_probability = class_to_probability.get(1, 0.0)
        ham_probability = class_to_probability.get(0, 0.0)
        probability = max(spam_probability, ham_probability)

    return {
        "prediction": prediction,
        "confidence": probability,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "transformed": transformed,
    }


def message_stats(message: str):
    words = re.findall(r"\b\w+\b", message)
    return {
        "characters": len(message),
        "words": len(words),
        "digits": sum(ch.isdigit() for ch in message),
        "links": len(re.findall(r"(https?://\S+|www\.\S+)", message.lower())),
        "exclamations": message.count("!"),
    }


def confidence_label(confidence):
    if confidence is None:
        return "Model probability unavailable"
    if confidence >= 0.90:
        return "Very high confidence"
    if confidence >= 0.75:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("🛡️ SpamShield")
    st.caption("TF-IDF + Multinomial Naive Bayes")

    st.divider()
    st.subheader("Quick examples")

    examples = {
        "Normal SMS": "Hey, are we still meeting at 6 tonight?",
        "Likely spam": "Congratulations! You have won a free prize. Call now to claim your reward.",
        "Bank-style": "Your account has been credited. Please review the transaction in your official banking app.",
    }

    for label, example in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.example_text = example

    st.divider()
    st.subheader("About the model")
    st.write(
        "The classifier was trained with TF-IDF features and a "
        "Multinomial Naive Bayes model."
    )
    st.caption("Use the prediction as a screening aid, not as proof that a message is malicious.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SpamShield</h1>
        <p>AI-powered Email & SMS Spam Classifier</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Message", "📦 Batch Check", "📊 Activity"])

with tab1:
    default_text = st.session_state.pop("example_text", "")

    st.subheader("Check a message")
    message = st.text_area(
        "Paste an Email/SMS message",
        value=default_text,
        height=180,
        placeholder="Example: Congratulations! You have won a free prize...",
        help="Paste the complete message for a better classification.",
    )

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        predict_clicked = st.button(
            "🔎 Analyze Message",
            type="primary",
            use_container_width=True,
        )
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.rerun()
    with col3:
        st.metric("Model Features", f"{getattr(tfidf, 'max_features', '—')}")

    if predict_clicked:
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
        else:
            with st.spinner("Analyzing message..."):
                result = predict_message(message)
                stats = message_stats(message)

            if result["prediction"] == 1:
                st.markdown(
                    """
                    <div class="result">
                        <h2>🚨 SPAM DETECTED</h2>
                        <p>This message was classified as <b>spam</b> by the trained model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result">
                        <h2>✅ NOT SPAM</h2>
                        <p>This message was classified as <b>not spam</b> by the trained model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Prediction", "SPAM" if result["prediction"] else "NOT SPAM")
            with c2:
                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.1f}%"
                    if result["confidence"] is not None
                    else "N/A",
                )
            with c3:
                st.metric("Characters", stats["characters"])
            with c4:
                st.metric("Words", stats["words"])

            if result["spam_probability"] is not None:
                st.subheader("Prediction probabilities")
                st.progress(
                    result["spam_probability"],
                    text=f"Spam probability: {result['spam_probability'] * 100:.1f}%",
                )
                st.progress(
                    result["ham_probability"],
                    text=f"Not-spam probability: {result['ham_probability'] * 100:.1f}%",
                )
                st.caption(confidence_label(result["confidence"]))

            with st.expander("🔬 View preprocessing details"):
                st.write("Text sent to the TF-IDF vectorizer:")
                st.code(result["transformed"] or "(empty after preprocessing)")
                st.caption(
                    "The displayed text is the stemmed representation used for model inference."
                )

            history_item = {
                "Message": message[:100] + ("..." if len(message) > 100 else ""),
                "Prediction": "Spam" if result["prediction"] else "Not Spam",
                "Confidence": (
                    f"{result['confidence'] * 100:.1f}%"
                    if result["confidence"] is not None
                    else "N/A"
                ),
            }
            st.session_state.history.insert(0, history_item)
            st.session_state.history = st.session_state.history[:20]

with tab2:
    st.subheader("Batch message classification")
    st.write("Upload a CSV containing a message column. The app will classify every row.")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Recommended column names: message, text, sms, or v2.",
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded, encoding="utf-8")
        except UnicodeDecodeError:
            batch_df = pd.read_csv(uploaded, encoding="latin1")

        candidate_columns = [
            col for col in batch_df.columns
            if str(col).strip().lower() in {"message", "text", "sms", "v2", "content"}
        ]

        if candidate_columns:
            message_col = candidate_columns[0]
        else:
            message_col = st.selectbox(
                "Select the message column",
                batch_df.columns,
            )

        if st.button("🚀 Classify CSV", type="primary"):
            messages = batch_df[message_col].fillna("").astype(str)

            transformed = [transform_text(msg) for msg in messages]
            vectors = tfidf.transform(transformed)
            predictions = model.predict(vectors).astype(int)

            batch_df["Prediction"] = [
                "Spam" if value == 1 else "Not Spam" for value in predictions
            ]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vectors)
                class_index = {
                    int(cls): idx for idx, cls in enumerate(model.classes_)
                }
                batch_df["Spam Probability"] = probabilities[:, class_index[1]]

            st.success(f"Classified {len(batch_df):,} messages.")

            a, b = st.columns(2)
            with a:
                st.metric("Spam", int((predictions == 1).sum()))
            with b:
                st.metric("Not Spam", int((predictions == 0).sum()))

            st.dataframe(batch_df, use_container_width=True)

            csv_data = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results",
                data=csv_data,
                file_name="spam_classification_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab3:
    st.subheader("Recent activity")

    if not st.session_state.history:
        st.info("No messages analyzed in this session yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("Clear activity"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "SpamShield • Machine Learning project • "
    "TF-IDF + Multinomial Naive Bayes • Built with Streamlit"
)
