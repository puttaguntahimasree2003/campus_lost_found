# app.py
# Campus Lost & Found with AutoMatch – Extended Version
# Text TF-IDF + Cosine Similarity + Simple Image Features + Extra ML Models

import os
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

FOUND_FILE = "items.csv"
EXPECTED_COLUMNS = ["id", "description", "location", "date", "contact", "img_r", "img_g", "img_b"]


# -------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------

def ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and make sure all expected columns exist."""
    if df is None:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    # lower-case column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # add missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ["img_r", "img_g", "img_b"]:
                df[col] = 0.0
            else:
                df[col] = ""

    # keep only expected columns
    df = df[EXPECTED_COLUMNS]

    # fix ids
    if df["id"].isna().any() or (df["id"] == "").any():
        df["id"] = range(1, len(df) + 1)

    df["id"] = df["id"].astype(int)

    # numeric image cols
    for col in ["img_r", "img_g", "img_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_found_items() -> pd.DataFrame:
    """Load found items from CSV."""
    if os.path.exists(FOUND_FILE):
        df = pd.read_csv(FOUND_FILE)
    else:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    return ensure_dataframe(df)


def save_found_item(description, location, date, contact, image_features):
    """Append a new found item row to CSV."""
    df = load_found_items()
    new_id = len(df) + 1

    img_r, img_g, img_b = image_features if image_features is not None else (0.0, 0.0, 0.0)

    new_row = {
        "id": new_id,
        "description": description,
        "location": location,
        "date": str(date),
        "contact": contact,
        "img_r": img_r,
        "img_g": img_g,
        "img_b": img_b,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = ensure_dataframe(df)
    df.to_csv(FOUND_FILE, index=False)


def extract_image_features(file) -> list | None:
    """
    Very simple image feature:
    average R, G, B values (0–1).
    This counts as 'image features' (classical, not deep learning).
    """
    if file is None:
        return None

    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))
        arr = np.array(img) / 255.0  # normalize
        mean_rgb = arr.mean(axis=(0, 1))  # shape (3,)
        return mean_rgb.tolist()
    except Exception:
        return None


def compute_combined_similarity(df: pd.DataFrame, lost_description: str,
                                lost_img_features: list | None,
                                alpha: float = 0.7) -> pd.DataFrame:
    """
    Compute combined similarity:
    similarity = alpha * text_similarity + (1-alpha) * image_similarity
    """
    if df.empty:
        return df

    # --- text similarity (TF-IDF + cosine)
    descriptions = [lost_description] + df["description"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    text_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    df = df.copy()
    df["text_sim"] = text_sims

    # --- image similarity (cosine on 3-dim RGB)
    if lost_img_features is not None and (df[["img_r", "img_g", "img_b"]].values.max() > 0):
        img_feats = df[["img_r", "img_g", "img_b"]].values.astype(float)
        lost_vec = np.array(lost_img_features, dtype=float).reshape(1, -1)
        img_sims = cosine_similarity(lost_vec, img_feats).flatten()
    else:
        img_sims = np.zeros(len(df))

    df["img_sim"] = img_sims

    # --- combined
    df["similarity"] = alpha * df["text_sim"] + (1 - alpha) * df["img_sim"]

    return df.sort_values("similarity", ascending=False)


# -------------------------------------------------------
# EXTRA ML MODELS (KNN, Naive Bayes, Logistic Regression)
# -------------------------------------------------------

def derive_category(text: str) -> str:
    """Simple rule-based category from description for training labels."""
    t = text.lower()
    if "bag" in t:
        return "bag"
    if "bottle" in t:
        return "bottle"
    if "id card" in t or "id card" in t or "id" in t:
        return "id_card"
    if "umbrella" in t:
        return "umbrella"
    if "hoodie" in t or "jacket" in t:
        return "clothing"
    if "earphone" in t or "headphone" in t:
        return "earphones"
    if "calculator" in t:
        return "calculator"
    if "phone" in t or "mobile" in t or "case" in t:
        return "phone_item"
    if "wallet" in t or "purse" in t:
        return "wallet"
    if "notebook" in t or "book" in t:
        return "notebook"
    if "keys" in t or "keychain" in t:
        return "keys"
    return "other"


@st.cache_resource
def train_classical_models():
    """
    Train KNN, Naive Bayes, Logistic Regression on description text
    to predict simple categories.
    """
    df = load_found_items()
    if df.empty:
        return None, {}

    df = df.copy()
    df["category"] = df["description"].fillna("").apply(derive_category)
    df = df[df["category"] != "other"]

    if df.empty or df["category"].nunique() < 2:
        return None, {}

    vect = CountVectorizer(stop_words="english")
    X = vect.fit_transform(df["description"])
    y = df["category"].values

    models = {}

    try:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        models["KNN"] = knn
    except Exception:
        pass

    try:
        nb = MultinomialNB()
        nb.fit(X, y)
        models["Naive Bayes"] = nb
    except Exception:
        pass

    try:
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)
        models["Logistic Regression"] = lr
    except Exception:
        pass

    return vect, models


def predict_categories_for_query(query: str):
    vect, models = train_classical_models()
    if vect is None or not models:
        return {}

    Xq = vect.transform([query])
    preds = {}
    for name, model in models.items():
        try:
            preds[name] = model.predict(Xq)[0]
        except Exception:
            continue
    return preds


# -------------------------------------------------------
# EVALUATION METRICS (Precision@k, MRR)
# -------------------------------------------------------

def evaluate_model(df: pd.DataFrame, k: int = 3):
    """
    Tiny evaluation with a few hand-crafted queries and expected ids.
    Used only to show Precision@k and MRR.
    """
    if df.empty:
        return 0.0, 0.0

    test_cases = [
        {"query": "Black Lenovo laptop bag with red zip", "expected_id": 1},
        {"query": "Blue steel water bottle", "expected_id": 2},
        {"query": "ID card named Surya", "expected_id": 3},
    ]

    hits = 0
    rr_sum = 0.0

    for case in test_cases:
        query = case["query"]
        expected = case["expected_id"]

        ranked = compute_combined_similarity(df, query, None, alpha=0.9)
        topk = ranked.head(k)

        ids = list(topk["id"].values)
        if expected in ids:
            hits += 1
            rank = ids.index(expected) + 1
            rr_sum += 1.0 / rank

    precision_at_k = hits / len(test_cases)
    mrr = rr_sum / len(test_cases)

    return precision_at_k, mrr


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.set_page_config(page_title="Campus Lost & Found - AutoMatch+", layout="wide")

st.title("🎒 Campus Lost and Found with AutoMatch+")
st.markdown("""
This version extends the basic TF-IDF + Cosine Similarity system with:

- ✅ *Image upload + simple image features* (mean RGB)
- ✅ *Combined text + image similarity*
- ✅ *Extra ML models*: KNN, Naive Bayes, Logistic Regression (category prediction)
- ✅ *Evaluation metrics*: Precision@k and MRR (on sample queries)
""")

mode = st.sidebar.radio("Choose an action:", ["➕ Add Found Item", "🔍 Search Lost Item", "📊 Evaluation"])


# -------------------------------------------------------
# ADD FOUND ITEM
# -------------------------------------------------------

if mode == "➕ Add Found Item":
    st.header("➕ Add Found Item")

    description = st.text_area(
        "Found Item Description",
        placeholder="Example: Black Lenovo laptop bag with red zip"
    )
    location = st.text_input(
        "Location Found",
        placeholder="Example: Library stairs"
    )
    date = st.date_input("Date Found")
    contact = st.text_input(
        "Contact (Phone/Email)",
        placeholder="9876543210"
    )
    image_file = st.file_uploader(
        "Optional: Upload image of the found item (for image features)",
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Save Found Item"):
        if description.strip() == "":
            st.error("Description cannot be empty!")
        else:
            img_features = extract_image_features(image_file)
            save_found_item(description, location, date, contact, img_features)
            st.success("✅ Found item saved with text + optional image features!")

    st.subheader("📋 Current Found Items")
    df = load_found_items()
    if df.empty:
        st.info("No found items added yet.")
    else:
        df_display = df.copy()
        df_display = df_display[["id", "description", "location", "date", "contact"]]
        st.dataframe(df_display, hide_index=True, use_container_width=True)


# -------------------------------------------------------
# SEARCH LOST ITEM
# -------------------------------------------------------

elif mode == "🔍 Search Lost Item":
    st.header("🔍 Search Lost Item")

    lost_description = st.text_area(
        "Describe the lost item (text)",
        placeholder="Example: Black Lenovo laptop bag with red zip"
    )
    lost_image_file = st.file_uploader(
        "Optional: Upload image of the lost item",
        type=["jpg", "jpeg", "png"],
        key="lost_image"
    )

    top_k = st.slider("How many matches to show?", 1, 10, 5)

    if st.button("Find Matches"):
        if lost_description.strip() == "":
            st.error("Please describe your lost item.")
        else:
            df = load_found_items()
            if df.empty:
                st.warning("No found items to match with yet!")
            else:
                lost_img_features = extract_image_features(lost_image_file)
                ranked = compute_combined_similarity(df, lost_description, lost_img_features, alpha=0.7)
                ranked = ranked.head(top_k)

                st.success(f"Showing top {len(ranked)} matches (combined text + image similarity):")

                for _, row in ranked.iterrows():
                    st.markdown("---")
                    st.markdown(f"### 🎯 Match Score: *{row['similarity']*100:.1f}%*")
                    st.write(f"*Description (found):* {row['description']}")
                    st.write(f"*Location:* {row['location']}")
                    st.write(f"*Date found:* {row['date']}")
                    st.write(f"*Contact:* {row['contact']}")
                    st.write(f"Text similarity: {row['text_sim']100:.1f}%  |  *Image similarity: {row['img_sim']*100:.1f}%")

                # Extra: show predictions from classical models
                st.markdown("### 🧠 Category prediction using extra ML models")
                preds = predict_categories_for_query(lost_description)
                if not preds:
                    st.info("Not enough data to train KNN / Naive Bayes / Logistic Regression yet.")
                else:
                    for model_name, cat in preds.items():
                        st.write(f"{model_name}:** predicted category → {cat}")

                st.info(
                    "The system uses TF-IDF + Cosine Similarity for text, "
                    "mean RGB features for image, and combines them into a single score. "
                    "KNN, Naive Bayes and Logistic Regression are trained on item descriptions "
                    "to predict coarse item categories."
                )


# -------------------------------------------------------
# EVALUATION PAGE
# -------------------------------------------------------

elif mode == "📊 Evaluation":
    st.header("📊 Model Evaluation (Precision@k, MRR)")

    df = load_found_items()
    if df.empty:
        st.warning("No data available for evaluation.")
    else:
        k = st.slider("Select k for Precision@k", 1, 10, 3)
        p_at_k, mrr = evaluate_model(df, k=k)

        st.write(f"*Precision@{k}:* {p_at_k:.2f}")
        st.write(f"*MRR (Mean Reciprocal Rank):* {mrr:.2f}")

        st.markdown("""
These metrics are computed on a small set of hand-crafted test queries:

- Precision@k measures how often the correct item appears in the top-k results.  
- MRR measures how early the correct item appears in the ranking (1 for rank 1, 0.5 for rank 2, etc.).
        """)
