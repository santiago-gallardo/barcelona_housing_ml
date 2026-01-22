import re
import unicodedata

import joblib
import streamlit as st


# ------------------------ Helpers ------------------------
def normalize_name(name: str) -> str:
    """Normalize feature names (accents/case/spaces) for reliable matching."""
    s = unicodedata.normalize("NFKD", str(name)).encode(
        "ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def key_for(name: str) -> str:
    """Safe Streamlit key for any feature name."""
    s = normalize_name(name)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s).strip("_")
    return f"in_{s}"


def nice_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def is_int_feature(name: str) -> bool:
    n = str(name).lower()
    keywords = ["rooms", "room", "bath", "baths",
                "toilets", "floor", "floors", "elevator", "parking"]
    return any(k in n for k in keywords)


def default_step(name: str) -> float:
    n = str(name).lower()
    if "lat" in n or "lon" in n or "lng" in n:
        return 0.0001
    if is_int_feature(name):
        return 1
    if "area" in n or "surface" in n or "m2" in n:
        return 1.0
    return 0.1


# Feature config (normalized keys so it matches even if the model changes casing/accents)
FEATURE_UI_CONFIG = {
    normalize_name("ÁREA CONSTRUIDA"): {
        "kind": "slider",
        "min": 15.0,
        "max": 1000.0,
        "value": 80.0,
        "step": 1.0,
        "label": "Constructed Area (m²)",
    },
    normalize_name("DISTANCIA AL CENTRO DE LA CIUDAD"): {
        "kind": "slider",
        "min": 0.0,
        "max": 25.0,
        "value": 3.0,
        "step": 0.1,
        "label": "Distance to City Center (km)",
    },
    normalize_name("DISTANCIA A LA DIAGONAL"): {
        "kind": "slider",
        "min": 0.0,
        "max": 25.0,
        "value": 2.0,
        "step": 0.1,
        "label": "Distance to Diagonal (km)",
    },
}


# ------------------------ Model helpers ------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path_str: str):
    return joblib.load(model_path_str, mmap_mode="r")


def get_expected_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    reg = getattr(model, "regressor_", None)
    if reg is not None and hasattr(reg, "feature_names_in_"):
        return list(reg.feature_names_in_)

    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "feature_names_in_"):
            return list(last.feature_names_in_)

    return None


def find_lat_lon_columns(features):
    low = {str(f).lower(): f for f in features}
    lat = next((low[k]
               for k in ["lat", "latitude", "latitud"] if k in low), None)
    lon = next(
        (low[k] for k in ["lon", "lng", "longitude", "longitud"] if k in low), None)
    return lat, lon


def init_session_state():
    """Initialize keys only once."""
    for k, default in [
        ("model", None),
        ("features", None),
        ("lat_col", None),
        ("lon_col", None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = default


def ensure_model_loaded(model_path) -> bool:
    """
    Loads model + features into session_state (once).
    model_path: pathlib.Path or str
    """
    init_session_state()

    if st.session_state.model is not None and st.session_state.features is not None:
        return True

    try:
        with st.spinner("Loading model..."):
            m = load_model_cached(str(model_path))

        feats = get_expected_features(m)
        if feats is None:
            st.error(
                "Could not infer model feature names (feature_names_in_). "
                "If the model was trained without DataFrame columns, save the feature list alongside the model."
            )
            return False

        lat_col, lon_col = find_lat_lon_columns(feats)

        st.session_state.model = m
        st.session_state.features = feats
        st.session_state.lat_col = lat_col
        st.session_state.lon_col = lon_col
        return True

    except MemoryError:
        st.error(
            "MemoryError while loading the model (~287MB).\n\nQuick fixes:\n"
            "- Close heavy apps and restart Streamlit\n"
            "- Enable/increase Windows virtual memory (pagefile)\n"
            "- Create a lighter deploy artifact"
        )
        return False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False
