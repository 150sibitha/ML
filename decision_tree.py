import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# ==============================
# STEP 1: CREATE DATASET
# ==============================

data = {
    "Income": ["High", "High", "Medium", "Medium", "Low", "Low"],
    "Credit_Score": ["Good", "Bad", "Good", "Bad", "Good", "Bad"],
    "Loan": ["Yes", "Yes", "Yes", "No", "No", "No"]
}

df = pd.DataFrame(data)

print("ORIGINAL DATASET:\n")
print(df)

# ==============================
# STEP 2: LABEL ENCODING
# ==============================

le_income = LabelEncoder()
le_credit = LabelEncoder()
le_loan = LabelEncoder()

df["Income"] = le_income.fit_transform(df["Income"])
df["Credit_Score"] = le_credit.fit_transform(df["Credit_Score"])
df["Loan"] = le_loan.fit_transform(df["Loan"])

print("\nENCODED DATASET:\n")
print(df)

# ==============================
# STEP 3: SPLIT INPUT & OUTPUT
# ==============================

X = df[["Income", "Credit_Score"]]
y = df["Loan"]

# ==============================
# STEP 4: TRAIN DECISION TREE (ID3)
# ==============================

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print("\nDecision Tree Model Trained Successfully!")

# ==============================
# STEP 5: PRINT DECISION TREE STRUCTURE
# ==============================

print("\nDECISION TREE STRUCTURE (TEXT FORMAT):\n")
tree_rules = export_text(model, feature_names=["Income", "Credit Score"])
print(tree_rules)

# ==============================
# STEP 6: PREDICTION FOR NEW DATA
# ==============================

income_input = "Medium"
credit_input = "Bad"

income_encoded = le_income.transform([income_input])[0]
credit_encoded = le_credit.transform([credit_input])[0]

prediction = model.predict([[income_encoded, credit_encoded]])
result = le_loan.inverse_transform(prediction)

print("\nNEW INSTANCE PREDICTION:")
print("Income =", income_input)
print("Credit Score =", credit_input)
print("Loan Approved =", result[0])
