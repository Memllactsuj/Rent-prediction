import streamlit as st
import numpy as np
import joblib
import pickle

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("features.pkl", "rb") as f:
	all_features = pickle.load(f)

st.title("Prediction price of flat 🏙")

rooms = st.number_input("Rooms count", min_value=1, max_value=10)
area = st.number_input("Area", min_value=10.0, max_value=300.0, value=50.0)
floor = st.number_input("Floor", min_value=1, max_value=30, value=3)
district = st.selectbox("District", ["halytskyi", "frankivskyi", "shevchenkivskyi", "lychakivskyi", "sykhivskyi"])
condition = st.selectbox("Condition", ["excellent", "renovated", "average", "good"])


if st.button("Predict price"):
	input_data = {
		'rooms': rooms,
		'area': area,
		'floor': floor,
		f'district_{district}': 1,
		f'condition_{condition}': 1
	}

	X = np.zeros(len(all_features))
	for i, feat in enumerate(all_features):
		X[i] = input_data.get(feat, 0)

	X_scaled = scaler.transform([X])
	prediction = model.predict(X_scaled)[0]

	st.success(f"Predicted price: {round(prediction)} uah")

