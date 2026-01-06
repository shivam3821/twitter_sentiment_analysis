import streamlit as st
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import nltk

# ---------------- Load Model and Vectorizer ----------------
trained_model = pickle.load(open('trained_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ---------------- Stopwords download check ----------------
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()


# ---------------- Preprocessing function ----------------
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        stemmer.stem(word)
        for word in stemmed_content
        if word not in stopwords.words('english')
    ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# ---------------- Streamlit UI ----------------
st.title('Twitter/X Sentiment Analysis App')
h = st.subheader("Logistic Regression Model using TF-IDF")

# ====== CREATE TABS ======
tab1, tab2 = st.tabs(["📝 Single Text Prediction", "📂 Bulk Prediction"])


# ================= TAB 1: Single Prediction =================
with tab1:
    st.subheader("Enter a single comment")

    user_input = st.text_area('Enter your comment here:')

    if st.button('Predict Sentiment', key="single"):
        st.balloons()
        if user_input.strip():
            stemmed_input = stemming(user_input)
            input_vector = vectorizer.transform([stemmed_input])
            prediction_result = trained_model.predict(input_vector)

            if prediction_result[0] == 1:
                st.success('Positive Sentiment 😊')
            else:
                st.error('Negative Sentiment 😠')
        else:
            st.warning('Please enter some text to predict sentiment.')


# ================= TAB 2: Bulk Prediction =================
with tab2:
    st.subheader("Upload multiple texts for prediction")

    uploaded_file = st.file_uploader(
        "Upload .txt (one per line) or .csv (column name: text)",
        type=['txt', 'csv'],
        key="bulk"
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".txt"):
            lines = uploaded_file.read().decode('utf-8', errors='ignore').splitlines()
            df = pd.DataFrame(lines, columns=["text"])

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a column named 'text'.")
                st.stop()

        st.write("📄 Preview of uploaded data:")
        st.dataframe(df.head())

        # Preprocess
        df["processed_text"] = df["text"].astype(str).apply(stemming)

        # Vectorize
        X_vectors = vectorizer.transform(df["processed_text"])

        # Predict
        df["prediction"] = trained_model.predict(X_vectors)
        df["sentiment"] = df["prediction"].map({1: "Positive", 0: "Negative"})

        # Results table
        st.subheader("📋 Prediction Results")
        st.dataframe(df[["text", "sentiment"]])


        # Download CSV
        st.subheader("⬇️ Download Results")

        csv_data = df[["text", "sentiment"]].to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )
