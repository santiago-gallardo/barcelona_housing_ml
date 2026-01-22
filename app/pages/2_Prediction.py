import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

# ------------------------ Make src/ importable (BEFORE importing app_utils) ------------------------
PROJECT_ROOT = Path("..").resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app_utils.prediction_utils import (  # noqa: E402
    FEATURE_UI_CONFIG,
    normalize_name,
    key_for,
    nice_label,
    default_step,
    ensure_model_loaded,
)

# ------------------------ Page config ------------------------
st.set_page_config(
    page_title="BCN Housing | Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 16px 18px;
}
h1, h2, h3 {margin-bottom: 0.35rem;}
small {opacity: 0.85;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------ Paths ------------------------
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

# ------------------------ Header ------------------------
st.title("🏠 Barcelona Apartment Price Predictor")
st.caption(
    "Select a location in Barcelona and fill in the inputs to run the model.")

# Barcelona bounds (dataset-based)
BCN_LAT_MIN, BCN_LAT_MAX = 41.3257633248147, 41.4677671782831
BCN_LON_MIN, BCN_LON_MAX = 2.09158572910443, 2.22187362250261
BCN_LAT_CENTER, BCN_LON_CENTER = 41.3851, 2.1734

left, right = st.columns([1.0, 1.1], gap="large")

submitted = False
inputs: dict = {}

with left:
    st.subheader("1) Location (Barcelona only)")

    lat = st.slider(
        "Latitude",
        min_value=float(BCN_LAT_MIN),
        max_value=float(BCN_LAT_MAX),
        value=float(BCN_LAT_CENTER),
        step=0.0001,
        key="lat_slider",
    )
    lon = st.slider(
        "Longitude",
        min_value=float(BCN_LON_MIN),
        max_value=float(BCN_LON_MAX),
        value=float(BCN_LON_CENTER),
        step=0.0001,
        key="lon_slider",
    )

    st.divider()
    st.subheader("2) Property features")

    feats_state = st.session_state.get("features")
    model_state = st.session_state.get("model")

    if feats_state is None or model_state is None:
        st.info("The model will be loaded once to avoid memory issues during reruns.")

        if st.button(
            "Load model and show inputs",
            type="primary",
            width="stretch",
            key="btn_load_model",
        ):
            ok = ensure_model_loaded(MODEL_PATH)
            if ok and st.session_state.get("features") is not None:
                st.rerun()

        st.stop()

    feats = st.session_state["features"]
    lat_col = st.session_state.get("lat_col")
    lon_col = st.session_state.get("lon_col")

    if lat_col and lon_col:
        inputs[lat_col] = lat
        inputs[lon_col] = lon

    with st.form("predict_form"):
        for f in feats:
            if f in (lat_col, lon_col):
                continue

            cfg = FEATURE_UI_CONFIG.get(normalize_name(f))

            if cfg and cfg.get("kind") == "slider":
                inputs[f] = st.slider(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["value"]),
                    step=float(cfg["step"]),
                    key=key_for(f),
                    help=cfg.get("help"),
                )
            else:
                inputs[f] = st.number_input(
                    nice_label(f),
                    value=float(st.session_state.get(key_for(f), 0.0)),
                    step=float(default_step(f)),
                    key=key_for(f),
                )

        submitted = st.form_submit_button(
            "Predict", type="primary", width="stretch")

with right:
    st.subheader("Map")

    map_df = pd.DataFrame([{"lat": lat, "lon": lon}])

    map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius=80,
        radius_min_pixels=7,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12.5)

    deck = pdk.Deck(
        map_style=map_style,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Selected point\nLat: {lat}\nLon: {lon}"},
    )
    st.pydeck_chart(deck, width="stretch")

    st.divider()
    st.subheader("Result")

    if not submitted:
        st.info("Fill the inputs and click **Predict**.")
    else:
        model = st.session_state["model"]
        feats = st.session_state["features"]
        lat_col = st.session_state.get("lat_col")
        lon_col = st.session_state.get("lon_col")

        if lat_col and lon_col:
            inputs[lat_col] = lat
            inputs[lon_col] = lon

        try:
            safe_inputs = {f: 0.0 for f in feats}
            safe_inputs.update(inputs)

            X = pd.DataFrame([safe_inputs], columns=feats)
            y_pred = float(model.predict(X)[0])

            st.metric("Estimated price", f"€{y_pred:,.0f}")
        except Exception as e:
            st.error(str(e))
