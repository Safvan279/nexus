import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd

import matplotlib.pyplot as plt
import re
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# File paths
ANALYZED_DATA_PATH = os.path.join('static', 'analyzed_reviews.csv')
CHART_PATH = os.path.join('static', 'sentiment_chart.png')
MODEL_PATH = os.path.join('static', 'random_forest_model.pkl')
VECTORIZER_PATH = os.path.join('static', 'vectorizer.pkl')

# Load the spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Define stop words using NLTK
stop_words_nltk = set(stopwords.words('english'))

# Define a function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = ' '.join(word for word in text.split() if word not in stop_words_nltk)  # Remove stopwords
    doc = nlp(text)  # Apply lemmatization
    text = ' '.join(token.lemma_ for token in doc)
    return text

# Remove previous data files at the start of the application
if os.path.exists(ANALYZED_DATA_PATH):
    os.remove(ANALYZED_DATA_PATH)

if os.path.exists(CHART_PATH):
    os.remove(CHART_PATH)

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

if os.path.exists(VECTORIZER_PATH):
    os.remove(VECTORIZER_PATH)

# Sentiment Analysis Function using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Define a function to assign emotions based on sentiment
def assign_emotion(sentiment):
    if sentiment == 'Positive':
        return 'Joyful😄'
    elif sentiment == 'Negative':
        return 'Angry😠'
    else:
        return 'Calm😌'

# Function to train the Random Forest model
def train_model(df):
    # Create a new DataFrame for modeling
    df['Sentiment_Label'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    X = df['reviews']
    y = df['Sentiment_Label']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train the Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_vec, y_train)

    # Save the trained model and vectorizer
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(rf_classifier, model_file)

    with open(VECTORIZER_PATH, 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'clear' in request.form:
            # Clear previous dataset and chart
            if os.path.exists(ANALYZED_DATA_PATH):
                os.remove(ANALYZED_DATA_PATH)
            if os.path.exists(CHART_PATH):
                os.remove(CHART_PATH)

            return render_template('home.html', chart_exists=False)

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if not file:
            return "No file selected"

        # Clear previous dataset and chart
        if os.path.exists(ANALYZED_DATA_PATH):
            os.remove(ANALYZED_DATA_PATH)
        if os.path.exists(CHART_PATH):
            os.remove(CHART_PATH)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template('home.html', error_message=f"Error processing the file: {str(e)}")

        if 'reviews' not in df.columns:
            return render_template('home.html', error_message="Error: The dataset must contain a 'reviews' column.")

        df = df.drop_duplicates()
        df = df.dropna(subset=['reviews'])
        df['reviews'] = df['reviews'].apply(preprocess_text)
        
        # Analyze sentiment using TextBlob
        df['Sentiment'] = df['reviews'].apply(analyze_sentiment)
        df['Emotion'] = df['Sentiment'].apply(assign_emotion)

        # Train Random Forest model with the data
        train_model(df)

        df.to_csv(ANALYZED_DATA_PATH, index=False)
        
        sentiment_counts = df['Sentiment'].value_counts()
        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['#669963', '#fb5462', '#ffca01'])
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.title('Sentiment Analysis Distribution')
        plt.savefig(CHART_PATH)
        plt.close()

        # Redirect to the dashboard after successful processing
        return redirect(url_for('dashboard'))

    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Only display data if the analyzed data file exists and is not empty
    if os.path.exists(ANALYZED_DATA_PATH) and os.path.getsize(ANALYZED_DATA_PATH) > 0:
        df = pd.read_csv(ANALYZED_DATA_PATH)
        df = df.reset_index().rename(columns={"index": "Column Number"})
        df = df[['Column Number', 'reviews', 'Sentiment', 'Emotion']].sort_values(by='Column Number')
        data = df.to_dict(orient='records')

        # Check if the chart exists to pass to the template
        chart_exists = os.path.exists(CHART_PATH)
        
        return render_template('dashboard.html', reviews=data, chart_exists=chart_exists)

    # Return the dashboard template with an empty list if no data file exists
    return render_template('dashboard.html', reviews=[], chart_exists=False)

@app.route('/about')
def about():
    return render_template('about.html')  # Ensure to create an about.html file in your templates folder

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/download')
def download_file():
    if os.path.exists(ANALYZED_DATA_PATH):
        return send_file(ANALYZED_DATA_PATH, as_attachment=True)
    return "No file to download."

@app.route('/download_chart')
def download_chart():
    if os.path.exists(CHART_PATH):
        return send_file(CHART_PATH, as_attachment=True)
    return "No chart to download."

if __name__ == '__main__':
    app.run(debug=True)
