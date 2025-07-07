
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import pickle


# Step 1: Load dataset
df = pd.read_csv('data.csv')

# Step 2: Prepare X and y
X = df['text']
y = df['intent']

# Step 3: Convert intent to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Convert text to vector
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_vector = vectorizer.fit_transform(X)

# Step 5: Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vector, y_encoded, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Save files
joblib.dump(model, 'ecommerce_classifier.pkl')
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model trained and saved successfully.")