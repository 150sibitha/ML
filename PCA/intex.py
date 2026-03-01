import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam_datarefined.csv")

X = data[["Prize", "Link", "Urgent"]]
y = data["Spam"]

# Encode Yes/No values
encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])

y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Sample testing
samples = pd.DataFrame(
    [[1, 0, 1], [0, 1, 1], [0, 0, 0]],
    columns=["Prize", "Link", "Urgent"]
)

print(list(zip(samples.values.tolist(), model.predict(samples))))
