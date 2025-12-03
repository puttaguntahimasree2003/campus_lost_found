import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- OPTIONAL WORDNET (NLP) ----------
WORDNET_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import wordnet as wn  # type: ignore

    nltk.download("wordnet")
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False

# ---------- FILE PATHS ----------
DATA_FILE = "items.csv"
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


# ---------- DATA HELPERS ----------

def ensure_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Make sure the dataframe has all required columns in the right order."""
    if df is None or df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Add missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ["img_r", "img_g", "img_b"]:
                df[col] = 0.0
            else:
                df[col] = ""

    # Keep only expected columns and order
    df = df[EXPECTED_COLUMNS]

    # Fix id column
    if df["id"].isna().any() or (df["id"] == "").any():
        df["id"] = range(1, len(df) + 1)
    df["id"] = df["id"].astype(int)

    # Ensure numeric image columns
    for col in ["img_r", "img_g", "img_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_items() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    return ensure_columns(df)


def save_items(df: pd.DataFrame) -> None:
    """Write the dataframe back to items.csv (so changes persist)."""
    df = ensure_columns(df)
    df.to_csv(DATA_FILE, index=False)


def save_feedback(lost_query: str, matched_id: int, score: float, good: bool) -> None:
    row = {
        "lost_query": lost_query,
        "matched_id": int(matched_id),
        "score": float(score),
        "feedback": "good" if good else "bad",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_FILE, index=False)


def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["lost_query", "matched_id", "score", "feedback", "timestamp"])


# ---------- IMAGE FEATURES (MEAN RGB) ----------

def extract_image_features(file) -> Optional[List[float]]:
    """Return mean R, G, B values in [0,1] for the uploaded image."""
    if file is None:
        return None
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))
        arr = np.array(img, dtype=float) / 255.0
        mean_rgb = arr.mean(axis=(0, 1))  # shape: (3,)
        return mean_rgb.tolist()
    except Exception:
        return None


def image_similarity_scores(df: pd.DataFrame, lost_img_features: Optional[List[float]]) -> np.ndarray:
    """Cosine similarity between query image features and stored item image features."""
    if lost_img_features is None:
        return np.zeros(len(df))

    img_feats = df[["img_r", "img_g", "img_b"]].values.astype(float)
    if np.all(img_feats == 0.0):
        # No stored image features yet
        return np.zeros(len(df))

    lost_vec = np.array(lost_img_features, dtype=float).reshape(1, -1)
    sims = cosine_similarity(lost_vec, img_feats).flatten()
    return np.clip(sims, 0.0, 1.0)


# ---------- WORDNET EXPANSION ----------

def expand_with_wordnet(text: str) -> str:
    """Expand query with WordNet synonyms (if available)."""
    if not WORDNET_AVAILABLE:
        return text

    tokens = [t for t in text.split() if len(t) > 3]
    extra = []

    for t in tokens:
        try:
            synsets = wn.synsets(t)
        except Exception:
            synsets = []
        for syn in synsets[:2]:  # limit per word
            for lemma in syn.lemmas()[:2]:
                w = lemma.name().replace("_", " ")
                if w.lower() != t.lower() and w not in extra:
                    extra.append(w)

    if extra:
        return text + " " + " ".join(extra)
    return text


# ---------- LOCATION / DATE LOGIC ----------

def location_score(found_loc: str, lost_loc: str) -> float:
    """Return 1.0 if same/overlapping, else 0.0."""
    if not lost_loc.strip():
        return 0.0
    f = str(found_loc).lower()
    l = str(lost_loc).lower()
    if not f or not l:
        return 0.0
    if l in f or f in l:
        return 1.0
    shared = set(f.split()) & set(l.split())
    return 1.0 if shared else 0.0


def date_score(found_date: str, lost_date_str: str) -> float:
    """Score closeness between found date and lost date."""
    if not lost_date_str:
        return 0.0
    try:
        d_found = datetime.fromisoformat(str(found_date)).date()
        d_lost = datetime.fromisoformat(str(lost_date_str)).date()
        diff = abs((d_found - d_lost).days)
        if diff == 0:
            return 1.0
        if diff <= 2:
            return 0.7
        if diff <= 7:
            return 0.4
        return 0.0
    except Exception:
        return 0.0


# ---------- HYBRID SCORE ----------

def compute_hybrid_scores(
    df: pd.DataFrame,
    lost_desc: str,
    lost_loc: str,
    lost_date_str: str,
    lost_img_features: Optional[List[float]],
) -> pd.DataFrame:
    """
    Hybrid score = 0.6 * Text + 0.15 * Image + 0.15 * Location + 0.1 * Date
    """
    if df.empty:
        return df

    # 1) Text similarity with optional WordNet expansion
    query_text = expand_with_wordnet(lost_desc)
    descriptions = [query_text] + df["description"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(descriptions)
    text_sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # 2) Image similarity
    img_sims = image_similarity_scores(df, lost_img_features)

    # 3) Location & Date scores
    loc_scores = np.array([location_score(r["location"], lost_loc) for _, r in df.iterrows()])
    date_scores = np.array([date_score(r["date"], lost_date_str) for _, r in df.iterrows()])

    # 4) Hybrid score
    alpha, beta, gamma, delta = 0.6, 0.15, 0.15, 0.1
    hybrid = alpha * text_sims + beta * img_sims + gamma * loc_scores + delta * date_scores

    df = df.copy()
    df["text_sim"] = text_sims
    df["img_sim"] = img_sims
    df["loc_score"] = loc_scores
    df["date_score"] = date_scores
    df["hybrid_score"] = hybrid

    return df.sort_values("hybrid_score", ascending=False)


# ---------- STREAMLIT UI SETUP ----------

st.set_page_config(page_title="Campus Lost & Found – AutoMatch", layout="wide")

st.title("Campus Lost and Found – AutoMatch (Hybrid Matching)")
st.caption(
    "Matches lost items with found items using a hybrid score that combines "
    "text similarity (TF–IDF + cosine), image colour features, location, and date."
)

mode = st.sidebar.radio(
    "Choose an action:",
    ["➕ Add Found Item", "🔍 Search Lost Item", "📊 Feedback & Logs"],
)


# ---------- ADD FOUND ITEM PAGE ----------

if mode == "➕ Add Found Item":
    st.subheader("➕ Add Found Item")

    desc = st.text_area(
        "Found Item Description",
        placeholder="Example: Black Lenovo laptop bag with red zip",
    )
    loc = st.text_input("Location Found", placeholder="Example: Library stairs")
    date = st.date_input("Date Found")
    contact = st.text_input("Contact (Phone/Email)", placeholder="9876543210")
    img_file = st.file_uploader(
        "Optional: Upload image of the found item (colour features will be extracted)",
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Save Found Item"):
        if not desc.strip():
            st.error("Description cannot be empty.")
        else:
            df = load_items()
            new_id = 1 if df.empty else int(df["id"].max()) + 1

            img_feats = extract_image_features(img_file)
            if img_feats is None:
                img_r, img_g, img_b = 0.0, 0.0, 0.0
            else:
                img_r, img_g, img_b = img_feats

            new_row = {
                "id": new_id,
                "description": desc,
                "location": loc,
                "date": str(date),
                "contact": contact,
                "img_r": img_r,
                "img_g": img_g,
                "img_b": img_b,
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_items(df)  # <-- THIS updates items.csv on disk

            st.success("✅ Found item saved and written to items.csv.")

    st.markdown("### Current Found Items")
    df_show = load_items()
    if df_show.empty:
        st.info("No found items stored yet.")
    else:
        st.dataframe(
            df_show[["id", "description", "location", "date", "contact"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------- SEARCH LOST ITEM PAGE ----------

elif mode == "🔍 Search Lost Item":
    st.subheader("🔍 Search Lost Item")

    lost_desc = st.text_area(
        "Describe the lost item (text)",
        placeholder="Example: Black Lenovo laptop bag with red zip",
    )
    lost_loc = st.text_input(
        "Approximate location lost (optional)",
        placeholder="Example: Library / CSE Block / Canteen",
    )
    lost_date = st.date_input("Approximate date lost (optional)", key="lost_date")
    lost_img_file = st.file_uploader(
        "Optional: Upload image of the lost item (for colour matching)",
        type=["jpg", "jpeg", "png"],
        key="lost_img",
    )

    top_k = st.slider("How many matches to show?", 1, 10, 5)

    if st.button("Find Matches"):
        if not lost_desc.strip():
            st.error("Please describe your lost item.")
        else:
            df = load_items()
            if df.empty:
                st.warning("No found items available to match.")
            else:
                lost_img_feats = extract_image_features(lost_img_file)
                lost_date_str = str(lost_date) if lost_date else ""

                ranked = compute_hybrid_scores(
                    df, lost_desc, lost_loc, lost_date_str, lost_img_feats
                ).head(top_k)

                if ranked.empty:
                    st.warning("No matches found.")
                else:
                    st.success(f"Top {len(ranked)} matches (hybrid score).")

                    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                        st.markdown("---")
                        st.markdown(
                            f"### 🔎 Match {rank}: Overall Score *{row['hybrid_score']*100:.1f}%*"
                        )
                        st.write(f"*Found Description:* {row['description']}")
                        st.write(f"*Location:* {row['location']}")
                        st.write(f"*Date Found:* {row['date']}")
                        st.write(f"*Contact:* {row['contact']}")

                        st.caption(
                            f"Text similarity: {row['text_sim']*100:.1f}%  |  "
                            f"Image similarity: {row['img_sim']*100:.1f}%  |  "
                            f"Location score: {row['loc_score']*100:.0f}%  |  "
                            f"Date score: {row['date_score']*100:.0f}%"
                        )

                        # Explanation bullets
                        explanations: List[str] = []
                        if row["text_sim"] > 0.6:
                            explanations.append("High overlap between your description and the found item text.")
                        elif row["text_sim"] > 0.3:
                            explanations.append("Moderate text similarity with your description.")
                        else:
                            explanations.append("Low text similarity (boosted by other signals).")

                        if row["img_sim"] > 0.4:
                            explanations.append("Image colour profile is similar to your uploaded image.")
                        if row["loc_score"] > 0.5:
                            explanations.append("Location is similar or related to the place you entered.")
                        if row["date_score"] > 0.5:
                            explanations.append("Found date is close to your approximate lost date.")

                        st.markdown("*Why this item is suggested:*")
                        for e in explanations:
                            st.markdown(f"- {e}")

                        # Feedback buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"👍 Helpful match (ID {row['id']})", key=f"good_{row['id']}"):
                                save_feedback(lost_desc, row["id"], row["hybrid_score"], True)
                                st.success("Feedback recorded as helpful.")
                        with col2:
                            if st.button(f"👎 Not useful (ID {row['id']})", key=f"bad_{row['id']}"):
                                save_feedback(lost_desc, row["id"], row["hybrid_score"], False)
                                st.warning("Feedback recorded as not useful.")

                    st.info(
                        "The hybrid score combines text similarity (with optional WordNet synonym expansion), "
                        "image colour similarity, and location/date logic into a single ranking."
                    )


# ---------- FEEDBACK PAGE ----------

elif mode == "📊 Feedback & Logs":
    st.subheader("📊 Feedback & Logs")

    df_fb = load_feedback()
    if df_fb.empty:
        st.info("No feedback collected yet.")
    else:
        total = len(df_fb)
        good = (df_fb["feedback"] == "good").sum()
        bad = (df_fb["feedback"] == "bad").sum()

        st.write(f"Total feedback entries: *{total}*")
        st.write(f"👍 Helpful: *{good}*")
        st.write(f"👎 Not useful: *{bad}*")
        if total > 0:
            st.write(f"Overall positive rate: *{(good/total)*100:.1f}%*")

        st.markdown("### Raw Feedback Log")
        st.dataframe(df_fb, use_container_width=True)

        st.caption(
            "This log can be used in future work to adjust the hybrid score or retrain models."
        )
