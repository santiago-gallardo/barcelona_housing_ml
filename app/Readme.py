import streamlit as st

st.set_page_config(
    page_title="BCN Housing | Technical README",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📘 Barcelona Housing — Technical README")
st.caption("Quick usage guide for the project and the app.")

st.markdown("""
## What is this?
This app provides a simple interface to estimate **Barcelona apartment prices** using a trained ML model.

## How to use
1. Open the **Prediction** page from the left sidebar.
2. Fill in the required inputs.
3. Click **Predict** to run the model and get an estimated price.

## Notes on interpretation
- The output is a **statistical reference** based on historical data.
- It is not a guaranteed “correct” price.
- Uncertainty tends to increase for high-end properties.

## Project structure
- `app/`: Streamlit application (this UI)
- `artifacts/`: saved model artifact (`model.joblib`)
- `src/`: reusable code (training / inference utilities)
- `notebooks/`: analysis and narrative notebooks
""")
