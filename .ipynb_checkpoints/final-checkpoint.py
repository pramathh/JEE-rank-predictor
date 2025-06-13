import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load JEE Rank Dataset
jee_file_path = "jee/mark_VS_rank.csv"  # Update if needed
jee_df = pd.read_csv(jee_file_path)

# Load CET Rank Dataset
cet_file_path = "jee/marks_range.xlsx"  # Ensure correct file format
cet_df = pd.read_excel(cet_file_path)

# --- JEE RANK MODEL ---
# Ensure dataset has at least two columns
if jee_df.shape[1] < 2:
    st.error("The JEE dataset must have at least two columns.")
    st.stop()

# Extract features and target for JEE
X_jee = jee_df.iloc[:, [0]]  # First column as feature
y_jee = jee_df.iloc[:, 1]  # Second column as target

# Train-Test Split for JEE
X_train_jee, X_test_jee, y_train_jee, y_test_jee = train_test_split(X_jee, y_jee, test_size=0.2, random_state=42)

# Train RandomForest Model for JEE
jee_model = RandomForestRegressor(n_estimators=100, random_state=42)
jee_model.fit(X_train_jee, y_train_jee)

# Model Accuracy for JEE
y_pred_jee = jee_model.predict(X_test_jee)
jee_accuracy = r2_score(y_test_jee, y_pred_jee)

# --- CET RANK MODEL ---
# Cleaning the "Marks Range (out of 180)" column
def parse_marks_range(mark_range):
    """Parses the Marks Range column into a single numeric value (midpoint)."""
    if isinstance(mark_range, str):
        mark_range = mark_range.strip()
        if '-' in mark_range:  # Case: "166-168"
            try:
                lower, upper = map(int, mark_range.split('-'))
                return (lower + upper) / 2  # Use midpoint
            except ValueError:
                return None  # Invalid data
        else:  # Single value case: "175"
            try:
                return int(mark_range)
            except ValueError:
                return None
    return None  # If it's not a valid string

cet_df["CET_Marks"] = cet_df["Marks Range (out of 180)"].apply(parse_marks_range)

def parse_rank_range(rank_range):
    """Parses the Rank column into a single numeric value (midpoint)."""
    if isinstance(rank_range, str):
        rank_range = rank_range.strip()
        if '-' in rank_range:
            try:
                lower, upper = map(int, rank_range.split('-'))
                return (lower + upper) / 2
            except ValueError:
                return None
        else:
            try:
                return int(rank_range)
            except ValueError:
                return None
    return None

cet_df["Rank"] = cet_df["Rank"].apply(parse_rank_range)

cet_df = cet_df.dropna(subset=["CET_Marks", "Rank"])

# Calculate final score based on 50% CET Marks and 50% of (user input for Chemistry + Maths + Physics)
def calculate_final_score(cet_marks, chem, maths, physics):
    return (0.5 * cet_marks) + (0.5 * (chem + maths + physics) / 3)

# Extract features and target for CET
X_cet = cet_df[["CET_Marks"]]
y_cet = cet_df["Rank"]

# Train-Test Split for CET
X_train_cet, X_test_cet, y_train_cet, y_test_cet = train_test_split(X_cet, y_cet, test_size=0.2, random_state=42)

# Train RandomForest Model for CET
cet_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
cet_model.fit(X_train_cet, y_train_cet)

# Model Accuracy for CET
y_pred_cet = cet_model.predict(X_test_cet)
cet_accuracy = r2_score(y_test_cet, y_pred_cet)

# Function to predict rank using JEE model
def predict_rank_jee(marks):
    """Predicts the JEE rank using the ML model."""
    predicted_rank = jee_model.predict(np.array([[marks]]))[0]
    return round(predicted_rank)

# Function to predict rank using CET model
def predict_rank_cet(cet_marks, chem, maths, physics):
    """Predicts the CET rank using the ML model based on the final score."""
    final_score = calculate_final_score(cet_marks, chem, maths, physics)
    predicted_rank = cet_model.predict(np.array([[final_score]]))[0]
    return round(predicted_rank)

# --- STREAMLIT UI ---
st.title("JEE & CET Rank Prediction Web App")

# Dropdown to select JEE or CET rank prediction
option = st.selectbox("Select an option:", ["JEE Rank Prediction", "CET Rank Prediction"])

# User input for marks
marks = st.number_input("Enter your marks (JEE Marks for JEE, CET Marks for CET):", min_value=0, max_value=180, value=90)

# For CET, user needs to input Chemistry, Maths, and Physics marks (JEE doesn't require these)
chemistry = 0  # Default value for JEE, no input required
maths = 0  # Default value for JEE, no input required
physics = 0  # Default value for JEE, no input required

if option == "CET Rank Prediction":
    # For CET rank prediction, ask user for Chemistry, Maths, and Physics marks
    chemistry = st.number_input("Enter Chemistry marks (0-100):", min_value=0, max_value=100, value=50)
    maths = st.number_input("Enter Maths marks (0-100):", min_value=0, max_value=100, value=50)
    physics = st.number_input("Enter Physics marks (0-100):", min_value=0, max_value=100, value=50)

# Predict button
if st.button("Predict"):
    if option == "JEE Rank Prediction":
        predicted_rank = predict_rank_jee(marks)
        st.success(f"Predicted JEE Rank: {predicted_rank}")
        st.info(f"Model Accuracy (R² Score): {jee_accuracy:.2%}")

    elif option == "CET Rank Prediction":
        predicted_rank = predict_rank_cet(marks, chemistry, maths, physics)
        st.success(f"Predicted CET Rank: {predicted_rank}")
        st.info(f"Model Accuracy (R² Score): {cet_accuracy:.2%}")