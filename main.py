import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ======================
# STEP 1: Load Dataset
# ======================
df = pd.read_csv("car details.csv")
df = df.dropna()

# ======================
# STEP 2: Features & Target
# ======================
# Drop unnecessary columns
if "torque" in df.columns: df = df.drop("torque", axis=1)
if "max_power" in df.columns: df = df.drop("max_power", axis=1)

# Add brand & model if not present
if "brand" not in df.columns: df["brand"] = "Toyota"
if "model" not in df.columns: df["model"] = "Fortuner"

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# ======================
# STEP 3: One-Hot Encoding
# ======================
X = pd.get_dummies(X)
cols = X.columns  # save for later

# ======================
# STEP 4: Train/Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# STEP 5: Scaling
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# STEP 6: Train Model
# ======================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ======================
# STREAMLIT APP
# ======================
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Add Toyota Fortuner GIF background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/originals/0f/b2/4a/0fb24a7ccaa58d34d1f81c5a003a1ae8.gif");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .title {
        font-size: 42px !important;
        color: white;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 10px #000;
    }
    .sub {
        font-size: 20px !important;
        color: #f8f9fa;
        text-align: center;
        text-shadow: 1px 1px 5px #000;
    }
    .stButton>button {
        background-color: #d90429;
        color: white;
        border-radius: 12px;
        font-size: 20px;
        padding: 12px 25px;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #9d0208;
        color: white;
        transform: scale(1.05);
    }
    .prediction-card {
        background: rgba(0,0,0,0.7);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title & description
st.markdown("<p class='title'>🚘 Car Price Prediction App</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Enter the details of your car and get an instant price prediction with style!</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar inputs
st.sidebar.header("📋 Enter Car Details")

brand = st.sidebar.text_input("Car Brand", "Toyota")
model_name = st.sidebar.text_input("Car Model", "Fortuner")
year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=50000)
fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
mileage = st.sidebar.text_input("Mileage (e.g., '19.7 kmpl')", "19.7 kmpl")
engine = st.sidebar.text_input("Engine (e.g., '1248 CC')", "1248 CC")
seats = st.sidebar.number_input("Seats", min_value=2, max_value=10, value=5)

# Prediction
if st.sidebar.button("🔮 Predict Price"):
    sample_car = {
        "brand": brand,
        "model": model_name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "seats": seats
    }
    sample_df = pd.DataFrame([sample_car])

    # One-hot encode & align with training
    sample_df = pd.get_dummies(sample_df)
    sample_df = sample_df.reindex(columns=cols, fill_value=0)

    # Scale
    sample_scaled = scaler.transform(sample_df)

    # Predict
    predicted_price = model.predict(sample_scaled)[0]

    # Show styled card
    st.markdown(
        f"<div class='prediction-card'>💰 Predicted Car Selling Price: <br> ₹ {predicted_price:,.2f}</div>",
        unsafe_allow_html=True
    )
    st.balloons()

    # ======================
    # Comparison Graph
    # ======================
    avg_price = df[(df["brand"] == brand) & (df["model"] == model_name)]["selling_price"].mean()

    fig, ax = plt.subplots(figsize=(6,4))
    labels = ["Predicted Price", f"Avg {brand} {model_name}"]
    values = [predicted_price, avg_price if not pd.isna(avg_price) else 0]

    colors = ["red", "blue"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Price (₹)")
    ax.set_title(f"Comparison for {brand} {model_name}")

    st.pyplot(fig)
