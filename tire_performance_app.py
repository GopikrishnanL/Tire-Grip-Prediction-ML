import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load the trained models (if already saved, else train them first)
def load_models():
    try:
        grip_model = joblib.load("grip_model.pkl")
        wear_model = joblib.load("wear_model.pkl")
        encoder = joblib.load("encoder.pkl")
    except:
        grip_model, wear_model, encoder = train_models()
    return grip_model, wear_model, encoder

# Train models if not already available
def train_models():
    np.random.seed(42)
    num_samples = 500

    speed = np.random.choice([50, 60, 70, 80, 90, 100], num_samples)
    temperature = np.random.choice([45, 46, 47, 48, 49, 50], num_samples)
    road_surface = np.random.choice(["Asphalt", "Wet Asphalt", "Concrete", "Gravel"], num_samples)

    grip = 1 - (speed / 200) - (temperature / 200) + np.random.uniform(-0.05, 0.05, num_samples)
    grip = np.clip(grip, 0.2, 1.0)
    
    wear = (speed / 100) * (temperature / 50) * 100 + np.random.uniform(-5, 5, num_samples)
    wear = np.clip(wear, 10, 100)
    
    df = pd.DataFrame({"Speed": speed, "Temperature": temperature, "Road Surface": road_surface, "Grip": grip, "Wear": wear})
    
    encoder = OneHotEncoder()
    road_surface_encoded = encoder.fit_transform(df[["Road Surface"]]).toarray()
    road_surface_cols = encoder.get_feature_names_out(["Road Surface"])
    df_encoded = pd.concat([df.drop("Road Surface", axis=1), pd.DataFrame(road_surface_encoded, columns=road_surface_cols)], axis=1)
    
    X = df_encoded.drop(columns=["Grip", "Wear"])
    y_grip = df_encoded["Grip"]
    y_wear = df_encoded["Wear"]
    
    grip_model = RandomForestRegressor(n_estimators=100, random_state=42)
    wear_model = RandomForestRegressor(n_estimators=100, random_state=42)
    grip_model.fit(X, y_grip)
    wear_model.fit(X, y_wear)
    
    joblib.dump(grip_model, "grip_model.pkl")
    joblib.dump(wear_model, "wear_model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    
    return grip_model, wear_model, encoder

# Load models
grip_model, wear_model, encoder = load_models()

# Streamlit UI
st.title("🏍️ Tire Performance Predictor")
st.sidebar.header("Enter Test Conditions")

# User Inputs
speed = st.sidebar.slider("Speed (km/h)", 50, 100, 70, step=10)
temperature = st.sidebar.slider("Temperature (°C)", 45, 50, 47)
road_surface = st.sidebar.selectbox("Road Surface", ["Asphalt", "Wet Asphalt", "Concrete", "Gravel"])

# Encode user input
user_data = pd.DataFrame({"Speed": [speed], "Temperature": [temperature], "Road Surface": [road_surface]})
road_surface_encoded = encoder.transform(user_data[["Road Surface"]]).toarray()
road_surface_cols = encoder.get_feature_names_out(["Road Surface"])
user_df = pd.concat([user_data.drop("Road Surface", axis=1), pd.DataFrame(road_surface_encoded, columns=road_surface_cols)], axis=1)

# Predictions
grip_pred = grip_model.predict(user_df)[0]
wear_pred = wear_model.predict(user_df)[0]

# Display Results
st.subheader("Predicted Tire Performance")
st.metric(label="Grip Level", value=f"{grip_pred:.2f}")
st.metric(label="Wear Percentage", value=f"{wear_pred:.2f}%")

# Explanation
st.write("🔹 **Grip Level:** A higher value (close to 1) indicates better grip.")
st.write("🔹 **Wear Percentage:** Higher wear means faster tire degradation.")

# Interactive Visualization
st.subheader("Performance Trends")

speed_values = [50, 60, 70, 80, 90, 100]
grip_values = [1 - (s / 200) - (temperature / 200) for s in speed_values]
wear_values = [(s / 100) * (temperature / 50) * 100 for s in speed_values]

# Grip vs Speed Plot
grip_fig = px.line(x=speed_values, y=grip_values, markers=True, title="Grip vs Speed", labels={"x": "Speed (km/h)", "y": "Grip Level"})
st.plotly_chart(grip_fig)

# Wear vs Speed Plot
wear_fig = px.line(x=speed_values, y=wear_values, markers=True, title="Wear vs Speed", labels={"x": "Speed (km/h)", "y": "Wear Percentage"})
st.plotly_chart(wear_fig)

# Multi-variable Analysis
st.subheader("Multi-Variable Analysis")
df_analysis = pd.DataFrame({"Speed": speed_values, "Temperature": [temperature]*6, "Grip": grip_values, "Wear": wear_values})
fig_multi = px.scatter_3d(df_analysis, x="Speed", y="Temperature", z="Grip", color="Wear", title="Speed vs Temperature vs Grip (Colored by Wear)")
st.plotly_chart(fig_multi)
