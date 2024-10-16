import joblib
import os
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

classification_model_path = os.path.join('../model', 'best_classifier.pkl')
regression_model_path = os.path.join('../model', 'lstm_model.keras')
vectorizer_path = os.path.join('../model', 'tfidf_vectorizer.pkl')
tokenizer_path = os.path.join('../model', 'tokenizer.pkl')

classification_model = joblib.load(classification_model_path)
regression_model = load_model(regression_model_path)
vectorizer = joblib.load(vectorizer_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def predict_rating_and_sentiment(review_text):
    """
    Функция предсказания рейтинга и настроения отзыва, 
    с учетом вероятностей точности моделей.

    Args:
    - review_text (str): Текст отзыва

    Returns:
    - tuple: Кортеж (рейтинг, настроение)
    """

    cleaned_text = clean_text(review_text)

    review_sequence = tokenizer.texts_to_sequences([cleaned_text])
    review_padded = pad_sequences(review_sequence, maxlen=350)
    rating_prediction = regression_model.predict(review_padded).flatten()[0]
    rating = int(round(rating_prediction))

    review_vector = vectorizer.transform([cleaned_text])
    sentiment_prediction = classification_model.predict(review_vector)[0]

    if (rating <= 5 and sentiment_prediction == 1) or (rating > 5 and sentiment_prediction == 0):

        if sentiment_prediction == 1:
            sentiment = 'Положительный'
            rating = max(rating, 6)  
        else:
            sentiment = 'Отрицательный'
            rating = min(rating, 5)  
    else:
        
        if sentiment_prediction == 1:
            sentiment = 'Положительный'
        else:
            sentiment = 'Отрицательный'

    return rating, sentiment

