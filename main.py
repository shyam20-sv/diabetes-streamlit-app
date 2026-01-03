import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Checkup", layout="centered")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("diabetes_dataset-interface.csv")

binary_map = {'Yes': 1, 'No': 0}

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

symptoms = [
    'Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia',
    'Genital thrush','visual blurring','Itching','Irritability',
    'delayed healing','partial paresis','muscle stiffness',
    'Alopecia','Obesity'
]

for col in symptoms:
    df[col] = df[col].map(binary_map)

df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})

# ---------------- MODEL ----------------
X = df.drop('class', axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=50
)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# ---------------- UI ----------------
st.title("🩺 Diabetes Checkup Application")
st.write("Enter age and symptoms to check diabetes status")

st.subheader("👤 Patient Details")
Age = st.slider("Age", 18, 100)
Gender = st.selectbox("Gender", ["Male", "Female"])

st.subheader("🧪 Symptoms")

def yn(label):
    return st.selectbox(label, ["Select here", "Yes", "No"])

user_input = {
    "Age": Age,
    "Gender": 1 if Gender == "Male" else 0
}

severity_score = 0
symptom_values = []

for symptom in symptoms:
    choice = yn(symptom.replace("_", " ").title())

    if choice == "Yes":
        value = 1
    elif choice == "No":
        value = 0
    else:
        value = None

    user_input[symptom] = value
    symptom_values.append(value)

# ---------------- BUTTON ACTION ----------------
if st.button("🔍 Check Diabetes"):

    # Validation (all at once, NOT one-by-one)
    if None in symptom_values:
        st.warning("⚠️ Please select Yes or No for all symptoms before proceeding.")
        st.stop()

    severity_score = sum(symptom_values)

    user_df = pd.DataFrame(user_input, index=[0])
    user_df = user_df[X.columns]

    # Prediction
    prediction = rf.predict(user_df)[0]

    # ---------------- GRAPH CHART ----------------
    st.subheader("📈 Symptom Severity Flow")

    severity_flow = pd.DataFrame({
        "Symptom Index": list(range(1, len(symptoms) + 1)),
        "Cumulative Severity": pd.Series(symptom_values).cumsum()
    })

    fig, ax = plt.subplots()
    ax.plot(
        severity_flow["Symptom Index"],
        severity_flow["Cumulative Severity"],
        marker='o'
    )

    ax.set_xlabel("Symptoms Progression")
    ax.set_ylabel("Cumulative Severity Score")
    ax.set_title("Diabetes Symptom Severity Flow")

    st.pyplot(fig)

    # ---------------- SEVERITY INTERPRETATION ----------------
    if severity_score <= 4:
        st.success(f"🟢 Severity Level: LOW ({severity_score}/14)")
    elif severity_score <= 9:
        st.warning(f"🟠 Severity Level: MODERATE ({severity_score}/14)")
    else:
        st.error(f"🔴 Severity Level: HIGH ({severity_score}/14)")

    # ---------------- RESULT ----------------
    st.subheader("🧠 Diabetes Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetes Result: POSITIVE")
    else:
        st.success("✅ Diabetes Result: NEGATIVE")
