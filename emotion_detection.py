# ===============================================
# EMOTION DETECTION FROM TWEETS
# Logistic Regression Model
# ===============================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ===============================================
# 1. DATASET (Sample Emotion-labeled Tweets)
# ===============================================
data = {
    'tweet': [
        # Joy 😊
        "I just got promoted! So happy! 🎉", "Best day ever! Love my friends ❤️", 
        "Sun is shining, feeling amazing! ☀️", "My dog is the cutest! 🐶",
        "Finally finished my project! Yay! 🚀", "Wedding day! Pure bliss 💍",
        
        # Sad 😢
        "Lost my job today... feeling down 💔", "Miss my family so much 😭", 
        "Another breakup... heart broken 💔", "Rainy days make me sad 🌧️",
        "Failed my exam, feeling terrible 📉", "Grandma passed away 😢",
        
        # Anger 😡
        "This traffic is ridiculous! 😡", "Boss yelled at me for no reason! 🤬",
        "Fake news everywhere! So angry! 🔥", "People cutting in line! 😤",
        "Scam call again! Unbelievable! 💢", "Lost my wallet! So mad! 😠",
        
        # Fear 😨
        "Hearing weird noises at night 😱", "Storm is coming, scared 🌪️",
        "Job interview tomorrow, nervous 😰", "Dark alley alone, frightened 😨",
        "Doctor appointment, anxious 🏥", "Earthquake warning! Scared! 🌍"
    ],
    'emotion': ['joy']*6 + ['sad']*6 + ['anger']*6 + ['fear']*6
}

df = pd.DataFrame(data)
print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nEmotion distribution:")
print(df['emotion'].value_counts())

# ===============================================
# 2. TEXT PREPROCESSING
# ===============================================
def preprocess_text(text):
    """Clean and preprocess tweets"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (keep words)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
df['clean_tweet'] = df['tweet'].apply(preprocess_text)
print("\nSample preprocessed tweets:")
print(df[['tweet', 'clean_tweet']].head())

# ===============================================
# 3. FEATURE EXTRACTION & MODEL TRAINING
# ===============================================
X = df['clean_tweet']
y = df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ===============================================
# 4. MODEL EVALUATION
# ===============================================
# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 MODEL PERFORMANCE")
print(f"Accuracy: {accuracy:.2%}")

# Classification report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['fear', 'joy', 'anger', 'sad'],
            yticklabels=['fear', 'joy', 'anger', 'sad'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ===============================================
# 5. PREDICTION FUNCTION
# ===============================================
def predict_emotion(tweet):
    """Predict emotion of a new tweet"""
    clean_tweet = preprocess_text(tweet)
    prediction = pipeline.predict([clean_tweet])[0]
    probability = pipeline.predict_proba([clean_tweet]).max()
    
    # Emojis for emotions
    emoji_dict = {
        'joy': '😊', 'sad': '😢', 
        'anger': '😡', 'fear': '😨'
    }
    
    return f"{prediction.upper()} {emoji_dict[prediction]} (Confidence: {probability:.2%})"

# ===============================================
# 6. INTERACTIVE PREDICTOR
# ===============================================
print("\n" + "="*50)
print("🧠 EMOTION DETECTOR READY!")
print("="*50)

# Test with sample tweets
test_tweets = [
    "I love this new song! So happy 🎵",
    "Why is everyone so rude today? 😡",
    "Feeling lonely tonight 😔",
    "Excited for vacation! ✈️",
    "Scared of the thunderstorm 🌩️"
]

print("\n🧪 Testing with sample tweets:")
for tweet in test_tweets:
    print(f"Tweet: '{tweet}'")
    print(f"Emotion: {predict_emotion(tweet)}")
    print("-" * 40)

# Interactive mode
print("\n🎯 Try your own tweets! (type 'quit' to exit)")
while True:
    user_tweet = input("\nEnter a tweet: ")
    if user_tweet.lower() == 'quit':
        break
    result = predict_emotion(user_tweet)
    print(f"Predicted: {result}")

print("\n🎉 Emotion Detection Model Complete!")
print("💡 Pro Tip: Train on larger datasets like GoEmotions for better accuracy!")