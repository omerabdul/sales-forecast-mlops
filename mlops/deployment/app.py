import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Sales Forecast Predictor", layout="wide")

# 1. Load Model from Hub
REPO_ID = "flyingdragon98/salesforecast-model"
FILENAME = "productionmodel.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return joblib.load(model_path)

model = load_model()

st.title("📈 Product-Store Sales Forecast")
st.write(
    "Enter the product and store attributes below to get a predicted total "
    "sales revenue for that product at that store."
)

# 2. Form with all features used at training time
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Product")
        product_type = st.selectbox("Product Type", [
            "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
            "Health and Hygiene", "Baking Goods", "Breads", "Breakfast", "Frozen Foods",
            "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"
        ])
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_weight = st.number_input("Product Weight (kg)", 0.0, 50.0, 12.5, step=0.1)
        product_mrp = st.number_input("Product MRP", 0.0, 1000.0, 140.0, step=1.0)

    with c2:
        st.subheader("Store")
        store_id = st.selectbox("Store Id", ["OUT001", "OUT002", "OUT003", "OUT004"])
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        store_city_tier = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", [
            "Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"
        ])

    with c3:
        st.subheader("Placement")
        store_est_year = st.number_input(
            "Store Establishment Year", 1950, datetime.now().year, 2005, step=1
        )
        product_allocated_area = st.slider(
            "Product Allocated Area (share of total store display area)",
            0.0, 1.0, 0.05, step=0.01
        )

    submit = st.form_submit_button("Generate Forecast")

# 3. Prediction Logic
if submit:
    store_age = datetime.now().year - store_est_year

    data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city_tier,
        "Store_Type": store_type,
        "Store_Age": store_age,
    }

    input_df = pd.DataFrame([data])

    try:
        predicted_total = float(model.predict(input_df)[0])

        st.divider()
        st.success(f"### Predicted Total Sales Revenue: ₹{predicted_total:,.2f}")

        # The model predicts a single point-in-time sales total

        with st.expander("Weekly / Monthly / Quarterly breakdown (naive periodization)"):
            st.caption(
                "This dataset has no date/time field, so these figures are an even "
                "split of the predicted total -- useful for rough planning, but not "
                "a substitute for a true time-series forecast. To get period-aware "
                "forecasts, retrain with historical time-stamped sales data."
            )
            col_w, col_m, col_q, col_a = st.columns(4)
            col_w.metric("Weekly", f"₹{predicted_total / 52:,.2f}")
            col_m.metric("Monthly", f"₹{predicted_total / 12:,.2f}")
            col_q.metric("Quarterly", f"₹{predicted_total / 4:,.2f}")
            col_a.metric("Annual", f"₹{predicted_total:,.2f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the column names in app.py match training data exactly.")
