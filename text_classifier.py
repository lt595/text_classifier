import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
X = pd.read_csv("data//train.csv")
y = pd.read_csv("data//test.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_tf, y_train)

# Accuracy scores
train_accuracy = classifier.score(X_train_tf, y_train)
test_accuracy = classifier.score(X_test_tf, y_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")