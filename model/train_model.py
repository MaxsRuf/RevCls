import os
import pandas as pd
import numpy as np
import pickle
import string
from glob import glob
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import joblib
nltk.download('stopwords')

def clean_text(text):
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def load_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            rating = int(filename.split('_')[1].split('.')[0])
            reviews.append((text, rating, label))
    return reviews

def load_data(data_dir):
    train_reviews, test_reviews = [], []
    
    train_reviews += load_reviews_from_folder(os.path.join(data_dir, 'train', 'pos'), 1)
    train_reviews += load_reviews_from_folder(os.path.join(data_dir, 'train', 'neg'), 0)
    test_reviews += load_reviews_from_folder(os.path.join(data_dir, 'test', 'pos'), 1)
    test_reviews += load_reviews_from_folder(os.path.join(data_dir, 'test', 'neg'), 0)
    
    train_df = pd.DataFrame(train_reviews, columns=['text', 'rating', 'label'])
    test_df = pd.DataFrame(test_reviews, columns=['text', 'rating', 'label'])
    
    return train_df, test_df

def generate_synthetic_reviews(train_df, n):
    synthetic_reviews_5 = []
    synthetic_reviews_6 = []
    
    for _ in range(n):
        try:
            review1 = train_df[train_df['rating'] == 4]['text'].sample(1).values[0]
            review2 = train_df[train_df['rating'] == 7]['text'].sample(1).values[0]
            synthetic_reviews_5.append(f"{review1} {review2} Это может быть класс 5.")
        except ValueError:
            continue

    for _ in range(n):
        try:
            review1 = train_df[train_df['rating'] == 4]['text'].sample(1).values[0]
            review2 = train_df[train_df['rating'] == 10]['text'].sample(1).values[0]
            synthetic_reviews_6.append(f"{review1} {review2} Это может быть класс 6.")
        except ValueError:
            continue

    synthetic_df_5 = pd.DataFrame({'text': synthetic_reviews_5, 'rating': [5] * len(synthetic_reviews_5), 'label': [0] * len(synthetic_reviews_5)})
    synthetic_df_6 = pd.DataFrame({'text': synthetic_reviews_6, 'rating': [6] * len(synthetic_reviews_6), 'label': [1] * len(synthetic_reviews_6)})

    return pd.concat([train_df, synthetic_df_5, synthetic_df_6], ignore_index=True)

def calculate_accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int).flatten() 
    correct_predictions = np.sum(np.abs(y_true - y_pred_rounded) <= 1) 
    return correct_predictions / len(y_true)

def transform_text(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(data['text']) 
    return X_tfidf, vectorizer

def binary_classification(train_data, test_data):
    X_train_tfidf, vectorizer = transform_text(train_data)
    X_test_tfidf = vectorizer.transform(test_data['text'])

    param_grid = {
        'bootstrap': [False],
        'max_depth': [None],
        'max_features': ['log2'],
        'min_samples_leaf': [1],
        'min_samples_split': [10],
        'n_estimators': [700]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_tfidf, train_data['label']) 

    y_pred = grid_search.predict(X_test_tfidf)

    accuracy = accuracy_score(test_data['label'], y_pred)
    f1 = f1_score(test_data['label'], y_pred)
    precision = precision_score(test_data['label'], y_pred)
    recall = recall_score(test_data['label'], y_pred)
    
    print(f"Точность бинарной классификации: {accuracy:.2f}")
    print(f"F1-мера: {f1:.2f}")
    print(f"Точность (Precision): {precision:.2f}")
    print(f"Полнота (Recall): {recall:.2f}")

    joblib.dump(grid_search.best_estimator_, 'model/best_classifier.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

    return grid_search.best_estimator_, vectorizer


data_dir = "Dataset"
train_df, test_df = load_data(data_dir) 

train_df = generate_synthetic_reviews(train_df, 2000)
test_df = generate_synthetic_reviews(test_df, 2000)

train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)

max_words = 5000
max_len = 350

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['cleaned_text'])

X_train = tokenizer.texts_to_sequences(train_df['cleaned_text'])
X_test = tokenizer.texts_to_sequences(test_df['cleaned_text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train_rating = train_df['rating'] 
y_test_rating = test_df['rating'] 

print(f"Shape of X_train: {X_train.shape}")  
print(f"Shape of X_test: {X_test.shape}")

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len)) 
model.add(LSTM(256, return_sequences=False)) 
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear')) 

model.compile(optimizer=Adam(learning_rate=0.001 ), loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

try:
    history = model.fit(X_train, y_train_rating, epochs=10, batch_size=64, validation_data=(X_test, y_test_rating))
except Exception as e:
    print(f"Ошибка во время обучения модели: {str(e)}")

y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test_rating, y_pred)
mae = mean_absolute_error(y_test_rating, y_pred)
accuracy = calculate_accuracy(y_test_rating, y_pred)

print(f"LSTM Mean Squared Error: {mse}")
print(f"LSTM Mean Absolute Error: {mae}")
print(f"LSTM Accuracy: {accuracy}")

model.save('model/lstm_model.keras')

with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

clf, vectorizer = binary_classification(train_df, test_df)