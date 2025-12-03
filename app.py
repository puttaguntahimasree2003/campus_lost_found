import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------- OPTIONAL WORDNET -----------------------
WORDNET_AVAILABLE = False
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False

# ----------------------- FILE PATHS -----------------------------
FOUND_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"

EXPECTED_COLUMNS = [
    "id",
    "description",
    "location",
    "date",
    "contact",
    "img_r",
    "img_g",
    "img_b",
]


# ----------------------- DATA HELPERS ---------------------------

def ensure_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize columns and guarantee all expected columns exist."""
    if df is None:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ["img_r", "img_g", "img_b"]:
                df[col] = 0.0
            else:
                df[col] = ""

    df = df[EXPECTED_COLUMNS]

    if df["id"].isna().any() or (df["id"] == "").any():
        df["id"] = range(1, len(df) + 1)
    df["id"] = df["id"].astype(int)

    for col in ["img_r", "img_g", "img_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_found_items() -> pd.DataFrame:
    if os.path.exists(FOUND_FILE):
        df = pd.read_csv(FOUND_FILE)
    else:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    return ensure_dataframe(df)


def save_found_item(description: str, location: str, date, contact: str,
                    image_features: Optional[List[float]]) -> None:
    df = load_found_items()
    new_id = len(df) + 1

    if image_features is None:
        img_r, img_g, img_b = 0.0, 0.0, 0.0
    else:
        img_r, img_g, img_b = image_features

    new_row = {
        "id": new_id,
        "description": description,
        "location": location
