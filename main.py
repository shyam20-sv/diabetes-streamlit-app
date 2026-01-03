import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("diabetes_dataset-interface.csv")

# Encoding categorical values
binary_map = {'Yes': 1, 'No': 0}
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
cols = ['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia',
        'Genital thrush','visual blurring','Itching','Irritability',
        'delayed healing','partial paresis','muscle stiffness',
        'Alopecia','Obesity']

for col in cols:
    df[col] = df[col].map(binary_map)

df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})

# Train-test split
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction System")

def user_report():
    Age = st.slider('Age', 18, 100)
    Gender = st.selectbox('Gender', [1, 0])
    Polyuria = st.selectbox('Polyuria', [1, 0])
    Polydipsia = st.selectbox('Polydipsia', [1, 0])
    sudden_weight_loss = st.selectbox('Sudden Weight Loss', [1, 0])
    weakness = st.selectbox('Weakness', [1, 0])
    Polyphagia = st.selectbox('Polyphagia', [1, 0])
    Genital_thrush = st.selectbox('Genital Thrush', [1, 0])
    visual_blurring = st.selectbox('Visual Blurring', [1, 0])
    Itching = st.selectbox('Itching', [1, 0])
    Irritability = st.selectbox('Irritability', [1, 0])
    delayed_healing = st.selectbox('Delayed Healing', [1, 0])
    partial_paresis = st.selectbox('Partial Paresis', [1, 0])
    muscle_stiffness = st.selectbox('Muscle Stiffness', [1, 0])
    Alopecia = st.selectbox('Alopecia', [1, 0])
    Obesity = st.selectbox('Obesity', [1, 0])

    data = {
        'Age': Age,
        'Gender': Gender,
        'Polyuria': Polyuria,
        'Polydipsia': Polydipsia,
        'sudden weight loss': sudden_weight_loss,
        'weakness': weakness,
        'Polyphagia': Polyphagia,
        'Genital thrush': Genital_thrush,
        'visual blurring': visual_blurring,
        'Itching': Itching,
        'Irritability': Irritability,
        'delayed healing': delayed_healing,
        'partial paresis': partial_paresis,
        'muscle stiffness': muscle_stiffness,
        'Alopecia': Alopecia,
        'Obesity': Obesity
    }

    return pd.DataFrame(data, index=[0])

user_data = user_report()
user_data = user_data[X.columns]  # IMPORTANT

# Prediction
prediction = rf.predict(user_data)

st.subheader("Prediction Result")
st.success("Positive" if prediction[0] == 1 else "Negative")
