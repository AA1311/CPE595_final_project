import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
import sys
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from car_price_prediction_improved import (
    DATASET_PATH,
    train_or_load_models,
    resolve_inputs,
    predict_price,
)

st.set_page_config(page_title="Car Price Estimator", layout="centered")

with open(HERE / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    return train_or_load_models(DATASET_PATH)


bundle = load_models()


st.markdown(
    """
    <div class="header">
        <h1>What's your car worth?</h1>
        <p>Fill in the details and get an instant price estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


c1, c2 = st.columns(2)
manufacturer_input = c1.text_input("Manufacturer", placeholder="e.g. Toyota")
model_input = c2.text_input("Model", placeholder="e.g. Camry")

c3, c4 = st.columns(2)
car_age = c3.number_input("Age (years)", min_value=0, max_value=50, step=1)
mileage = c4.number_input("Mileage", min_value=0, max_value=500_000, step=1000)

c5, c6 = st.columns(2)
accidents = c5.selectbox("Accidents or damage?", ["Select one", "No", "Yes"])
one_owner = c6.selectbox("One owner?", ["Select one", "No", "Yes"])


if st.button("Estimate price →", type="primary", use_container_width=True):

    if not manufacturer_input.strip() or not model_input.strip() or "Select one" in (accidents, one_owner):
        st.warning("Please fill in all fields.")
        st.stop()

    print("\n=== NEW PREDICTION ===")

    user_df, notice, is_fallback = resolve_inputs(
        bundle,
        manufacturer_input,
        model_input,
        car_age,
        mileage,
        1 if accidents == "Yes" else 0,
        1 if one_owner == "Yes" else 0,
    )

    price = predict_price(bundle, user_df)
    error_metrics = bundle["fallback_metrics"] if is_fallback else bundle["best_metrics"]

    print(f"Predicted Price: ${price:,.2f}")
    print(f"Model Used: {'Fallback' if is_fallback else bundle['best_model_name']}")
    print(
        f"Range: ${max(price - error_metrics['MAE'], 0):,.2f} – "
        f"${price + error_metrics['MAE']:,.2f}"
    )
    print("=====================\n")

    if notice:
        st.markdown(f'<div class="notice info">{notice}</div>', unsafe_allow_html=True)

    if is_fallback:
        st.markdown(
            '<div class="notice warn">Unknown manufacturer — using fallback model.</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">Estimated price</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-range"> Range:
                ${max(price - error_metrics["MAE"], 0):,.0f} – ${price + error_metrics["MAE"]:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )