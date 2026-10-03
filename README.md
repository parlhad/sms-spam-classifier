# 🛡️ SpamShield --- Email & SMS Spam Classifier

A Streamlit web application for classifying Email and SMS messages as
**Spam** or **Not Spam** using **TF-IDF + Multinomial Naive Bayes**.

## 🚀 Live Demo

https://sms-spam-classifier-kae8xcwcpumdatifmkrlr7.streamlit.app/

## 📌 Overview

SpamShield is an end-to-end NLP and machine-learning project. A user can
paste an SMS/email message, preprocess it using the same pipeline used
during training, transform it with a fitted TF-IDF vectorizer, and
classify it with a fitted Multinomial Naive Bayes model.

The application also supports batch CSV classification, prediction
probabilities, preprocessing inspection, message statistics, quick
examples, downloadable results, and session-level activity history.

## 🔄 Prediction Pipeline

``` text
Raw Message
    ↓
Lowercase
    ↓
NLTK Tokenization
    ↓
Alphanumeric Filtering
    ↓
English Stopword/Punctuation Removal
    ↓
Porter Stemming
    ↓
TF-IDF Vectorization (3000 features)
    ↓
Multinomial Naive Bayes
    ↓
SPAM / NOT SPAM
    ↓
Probability + Confidence + Statistics
```

## ✨ Features

-   Single Email/SMS classification
-   Spam / Not Spam prediction
-   Confidence and class probabilities
-   Message character and word statistics
-   Preprocessing/stemming preview
-   Quick example messages
-   Batch CSV classification
-   Automatic detection of common message-column names
-   Manual message-column selection when needed
-   Spam and Not-Spam batch counts
-   Downloadable CSV results
-   Session prediction history
-   Responsive Streamlit interface

## 🧠 Machine Learning

### Text preprocessing

The inference pipeline matches the training pipeline:

1.  Convert text to lowercase.
2.  Tokenize using NLTK.
3.  Keep alphanumeric tokens.
4.  Remove English stopwords and punctuation.
5.  Apply Porter stemming.
6.  Join the processed tokens.
7.  Transform the processed text with the fitted TF-IDF vectorizer.

### TF-IDF

The fitted vectorizer is stored in:

``` text
vectorizer.pkl
```

The model uses:

``` text
TfidfVectorizer(max_features=3000)
```

The 3000-feature limit is the feature-space configuration used by the
deployed application.

### Classifier

The classifier is:

``` text
MultinomialNB
```

The fitted classifier is stored in:

``` text
model.pkl
```

The labels are:

``` text
0 → Not Spam
1 → Spam
```

When supported, `predict_proba()` is used to display spam and not-spam
probabilities.

## 📊 Confidence

The interface reports the highest class probability as the displayed
confidence.

    Probability Label
  ------------- ----------------------
          ≥ 90% Very high confidence
          ≥ 75% High confidence
          ≥ 60% Moderate confidence
         \< 60% Low confidence

Confidence is a model probability estimate, not a guarantee of
correctness.

## 📦 Batch CSV Classification

Upload a CSV containing a message column.

Common automatically recognized column names are:

``` text
message
text
sms
v2
content
```

Example:

``` csv
message
"Hey, are we still meeting at 6 tonight?"
"Congratulations! You have won a free prize. Call now!"
"Your order has been shipped successfully."
```

The application adds:

``` text
Prediction
Spam Probability
```

and provides a downloadable file:

``` text
spam_classification_results.csv
```

## 📁 Project Structure

``` text
sms-spam-classifier/
│
├── app.py
│   └── Streamlit application and inference logic
│
├── model.pkl
│   └── Fitted Multinomial Naive Bayes model
│
├── vectorizer.pkl
│   └── Fitted TF-IDF vectorizer
│
├── requirements.txt
│   └── Python dependencies
│
├── spam.csv
│   └── Training dataset
│
├── spam_classifier_model.py
│   └── Training/model-related Python code
│
├── spam_classifier_model.ipynb
│   └── Training notebook
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Git ignore rules
```

## 🛠️ Technology Stack

  Technology                  Purpose
  --------------------------- --------------------------------
  Python                      Core language
  Streamlit                   Web application
  Scikit-learn                TF-IDF and machine learning
  NLTK                        Natural-language preprocessing
  Pandas                      CSV/data processing
  NumPy                       Numerical support
  Git                         Version control
  GitHub                      Source-code hosting
  Streamlit Community Cloud   Deployment

## 📦 Requirements

The project uses:

``` txt
streamlit>=1.40,<2.0
scikit-learn==1.6.1
nltk==3.9.1
pandas>=2.2,<3.0
numpy>=1.26,<3.0
```

`scikit-learn` is pinned to `1.6.1` because the serialized model
artifacts were trained with that version. Matching the serialization
environment helps avoid model-loading compatibility problems.

## 💻 Run Locally

### 1. Clone

``` bash
git clone https://github.com/parlhad/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Create a virtual environment

Windows:

``` powershell
python -m venv .venv
.venv\Scriptsctivate
```

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Start Streamlit

``` bash
streamlit run app.py
```

or:

``` bash
python -m streamlit run app.py
```

Keep these files beside `app.py`:

``` text
model.pkl
vectorizer.pkl
```

## 🌐 Streamlit Cloud Deployment

1.  Push the project to GitHub.
2.  Open Streamlit Community Cloud.
3.  Create a new app.
4.  Select the GitHub repository.
5.  Select the `main` branch.
6.  Set the main file to `app.py`.
7.  Deploy.
8.  Streamlit installs dependencies from `requirements.txt`.
9.  Test the live application.

The deployed application is available at:

https://sms-spam-classifier-kae8xcwcpumdatifmkrlr7.streamlit.app/

## 🔧 Git Workflow

After changing the application:

``` bash
git status
git add .
git commit -m "Update SpamShield application"
git push origin main
```

Streamlit Cloud can then redeploy the latest GitHub commit.

## 🧪 Model Training and Serialization

The training notebook is:

``` text
spam_classifier_model.ipynb
```

The general workflow is:

``` text
Dataset
  ↓
Data Cleaning
  ↓
Text Preprocessing
  ↓
Train/Test Split
  ↓
TF-IDF Fit
  ↓
MultinomialNB Fit
  ↓
Evaluation
  ↓
Save model.pkl
  ↓
Save vectorizer.pkl
```

The classifier must be fitted before serialization:

``` python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

Then save the fitted artifacts:

``` python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
```

The saved files are then used by the Streamlit application for
inference.

## 🔬 Application Inference

For a single message, the application effectively performs:

``` python
transformed = transform_text(message)
vector = tfidf.transform([transformed])
prediction = model.predict(vector)[0]
```

For probabilities:

``` python
probabilities = model.predict_proba(vector)[0]
```

This keeps training-time preprocessing and inference-time preprocessing
aligned.

## 📈 Message Statistics

The application reports useful input statistics such as:

-   Character count
-   Word count
-   Digit count
-   Number of links
-   Number of exclamation marks

These statistics are displayed as supporting information and are not
separate machine-learning features used by the classifier.

## 📊 Activity History

The Activity tab keeps recent predictions in Streamlit session state.

Each record contains:

-   Message preview
-   Prediction
-   Confidence

The application keeps the most recent 20 records for the current
session.

This is session-level history; it is not a permanent database.

## 🔐 Model and Data Files

### `model.pkl`

Contains the fitted Multinomial Naive Bayes classifier.

### `vectorizer.pkl`

Contains the fitted TF-IDF vectorizer.

Both artifacts are required for prediction.

Do not replace one artifact independently with an artifact created from
a different training pipeline.

## 🐛 Troubleshooting

### `NotFittedError`

If you see:

``` text
This MultinomialNB instance is not fitted yet
```

the loaded `model.pkl` is not a fitted classifier.

The training process must contain:

``` python
model = MultinomialNB()
model.fit(X_train, y_train)
```

before saving:

``` python
pickle.dump(model, open("model.pkl", "wb"))
```

### Scikit-learn version warning

If scikit-learn reports that the pickle was created using a different
version, use the project's pinned version:

``` text
scikit-learn==1.6.1
```

### Missing NLTK resources

The application prepares required NLTK resources such as:

``` text
punkt
punkt_tab
stopwords
```

If an NLTK resource error occurs, verify that `nltk==3.9.1` is installed
and that the required resources are available.

### Missing model files

Verify that:

``` text
app.py
model.pkl
vectorizer.pkl
```

are in the expected project directory.

## ⚠️ Limitations

SpamShield is a machine-learning screening tool, not a complete
email-security or anti-fraud system.

Possible limitations include:

-   Unseen wording may be misclassified.
-   Obfuscated messages can reduce accuracy.
-   Very short messages may contain insufficient information.
-   Legitimate promotional messages may resemble spam.
-   Spam patterns change over time.
-   Model probabilities are not guarantees.
-   Performance depends on the training dataset and preprocessing
    pipeline.

For real-world suspicious messages, independently verify the sender and
avoid opening unknown links or sharing sensitive information.

## 🔒 Privacy

Avoid entering passwords, one-time passwords, financial credentials,
private communications, or other highly sensitive information into a
public demonstration deployment unless the deployment's data-handling
practices are appropriate for that information.

## 🎯 Use Cases

This project demonstrates practical applications of:

-   SMS spam detection
-   Email spam screening
-   Natural-language processing
-   Text classification
-   TF-IDF feature engineering
-   Naive Bayes classification
-   Batch text inference
-   Model serialization
-   Streamlit development
-   GitHub version control
-   Cloud deployment

## 🚀 Future Improvements

Possible next steps include:

-   Larger and more diverse training datasets
-   Cross-validation
-   Precision, recall and F1-score reporting
-   Confusion matrix visualization
-   Model comparison
-   Hyperparameter tuning
-   Explainable predictions
-   URL/domain risk analysis
-   Multilingual spam detection
-   Transformer-based NLP models
-   REST API support
-   Persistent analytics
-   Model versioning and monitoring
-   Automated retraining

## 📚 Learning Outcomes

This project demonstrates:

### Python

-   Functions
-   File handling
-   Exception handling
-   Serialization
-   Data processing

### NLP

-   Tokenization
-   Stopword removal
-   Stemming
-   Text normalization

### Machine Learning

-   Train/test splitting
-   TF-IDF
-   Multinomial Naive Bayes
-   Classification
-   Probability estimation
-   Model persistence

### Deployment

-   Streamlit
-   Git
-   GitHub
-   Dependency management
-   Cloud deployment

## 🏗️ High-Level Architecture

``` text
                 ┌──────────────────────┐
                 │    Streamlit UI      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Email / SMS       │
                 │      Input           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  NLTK Preprocessing  │
                 │  • Lowercase         │
                 │  • Tokenization      │
                 │  • Filtering         │
                 │  • Stopwords         │
                 │  • Stemming          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   TF-IDF Vectorizer  │
                 │    3000 Features     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Multinomial Naive    │
                 │       Bayes          │
                 └──────────┬───────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             ┌──────────┐      ┌────────────┐
             │   SPAM   │      │  NOT SPAM  │
             └──────────┘      └────────────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Probability / Stats │
                 └──────────────────────┘
```

## 📝 Project Summary

**SpamShield** is an end-to-end NLP machine-learning application that
combines:

``` text
Python
+ NLTK
+ TF-IDF
+ Multinomial Naive Bayes
+ Streamlit
+ GitHub
+ Streamlit Community Cloud
```

It provides an interactive interface for individual message screening
and batch CSV classification while preserving the preprocessing pipeline
used during model training.

## 🌐 Live Demo

**SpamShield --- Email & SMS Spam Classifier**

https://sms-spam-classifier-kae8xcwcpumdatifmkrlr7.streamlit.app/

------------------------------------------------------------------------

> **Note:** Predictions are generated by a trained machine-learning
> model and should be treated as screening results rather than
> definitive proof that a message is malicious.
