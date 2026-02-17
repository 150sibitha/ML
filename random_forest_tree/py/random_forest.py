# =====================================
# Random Forest - Student Pass/Fail
# Dataset from CSV (Attendance: Good/Poor)
# User Defined Input
# =====================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# STEP 1: LOAD CSV DATASET
# -----------------------------

df = pd.read_csv("RFdataset.csv")

print("DATASET LOADED SUCCESSFULLY\n")
print(df.head())

# -----------------------------
# STEP 2: ENCODE CATEGORICAL DATA
# -----------------------------

le_study = LabelEncoder()
le_att = LabelEncoder()
le_res = LabelEncoder()

df["Study_Hours"] = le_study.fit_transform(df["Study_Hours"])
df["Attendance"] = le_att.fit_transform(df["Attendance"])
df["Result"] = le_res.fit_transform(df["Result"])

# -----------------------------
# STEP 3: SPLIT INPUT & OUTPUT
# -----------------------------

X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# -----------------------------
# STEP 4: TRAIN RANDOM FOREST
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=42
)

model.fit(X, y)

print("\n✅ Random Forest Model Trained Successfully!")

# -----------------------------
# STEP 5: USER INPUT
# -----------------------------

print("\nEnter Student Details")

study_input = input("Enter Study Hours (High / Medium / Low): ").strip().capitalize()
attendance_input = input("Enter Attendance (Good / Poor): ").strip().capitalize()

# Validate input
if study_input not in ["High", "Medium", "Low"]:
    print("\n❌ Invalid Study Hours input")
    exit()

if attendance_input not in ["Good", "Poor"]:
    print("\n❌ Invalid Attendance input")
    exit()

# Encode input
study_encoded = le_study.transform([study_input])[0]
attendance_encoded = le_att.transform([attendance_input])[0]

# -----------------------------
# STEP 6: PREDICTION
# -----------------------------

prediction = model.predict([[study_encoded, attendance_encoded]])
result = le_res.inverse_transform(prediction)

print("\n📌 PREDICTION RESULT")
print("Study Hours :", study_input)
print("Attendance  :", attendance_input)
print("Result      :", result[0])