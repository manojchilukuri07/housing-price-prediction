
import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/xgb_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏠 AI Housing Price Predictor")

st.markdown("""
### Machine Learning Based Real Estate Price Estimation

Predict house prices using an XGBoost regression model
trained on housing dataset features.
""")

st.divider()
st.write(
    "Predict the estimated selling price of a house "
    "using Machine Learning."
)

st.success("Machine Learning Model Loaded Successfully!")
col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "XGBoost")
col2.metric("Problem Type", "Regression")
col3.metric("Test R²", "90.10%")
# -----------------------------
# House Input Section
# -----------------------------
st.header("🏡 Enter House Details")

overall_qual = st.slider(
    "Overall House Quality",
    min_value=1,
    max_value=10,
    value=5
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

gr_liv_area = st.number_input(
    "Living Area (sq ft)",
    min_value=100,
    max_value=10000,
    value=1500
)

garage_area = st.number_input(
    "Garage Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=400
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=10,
    value=3
)

full_bath = st.number_input(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

neighborhood = st.selectbox(
    "Neighborhood",
    [
        "NAmes", "CollgCr", "OldTown", "Edwards",
        "Somerst", "Gilbert", "NridgHt", "Sawyer",
        "NWAmes", "BrkSide", "Crawfor", "Mitchel",
        "NoRidge", "Timber", "IDOTRR", "ClearCr",
        "SWISU", "StoneBr", "Blmngtn", "MeadowV",
        "Veenker", "NPkVill", "Blueste"
    ]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict House Price"):

    # Calculate house age
    house_age = 2010 - year_built

    # Create input data
    input_data = {
        "TotalSF": gr_liv_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "HouseAge": house_age,
        "GarageArea": garage_area,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": 6,
        "MasVnrArea": 0,
        "Fireplaces": 0,
        "1stFlrSF": gr_liv_area,
        "2ndFlrSF": 0,
        "TotalBathrooms": full_bath,
        "TotalPorchSF": 0,
        "TotalOutdoorSF": 0
    }

    # Initialize neighborhood columns
    for col in model_columns:
        if col.startswith("Neighborhood_"):
            input_data[col] = 0

    # Set selected neighborhood
    neighborhood_column = "Neighborhood_" + neighborhood

    if neighborhood_column in model_columns:
        input_data[neighborhood_column] = 1

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training columns
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict price
    prediction = model.predict(input_df)

    # Display result
    st.success(
        f"Estimated House Price: ${prediction[0]:,.2f}"
    )

    st.metric(
    label="Predicted Selling Price",
    value=f"${prediction[0]:,.2f}"
    )

    st.subheader("Price Visualization")

    st.bar_chart(
    pd.DataFrame({
        "Price": [prediction[0]]
    })
)

    st.info(
        "This prediction is an estimate generated "
        "by the trained XGBoost model."
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Developed using Python, XGBoost, Pandas and Streamlit")
