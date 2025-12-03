import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def ensure_items_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ["img_r", "img_g", "img_b"]:
                df[col] = 0.0
            elif col == "id":
                df[col] = 0
            else:
                df[col] = ""

    df = df[EXPECTED_COLUMNS]

    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    if (df["id"] == 0).any():
        df["id"] = range(1, len(df) + 1)

    for col in ["img_r", "img_g", "img_b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["description", "location", "date", "contact"]:
        df[col] = df[col].fillna("").astype(str)

    return df


def load_items() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    return ensure_items_df(df)


def save_items(df: pd.DataFrame) -> None:
    df = ensure_items_df(df)
    df.to_csv(DATA_FILE, index=False)


def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["lost_query", "matched_id", "score", "feedback", "timestamp"])


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

# ---------- IMAGE FEATURES ----------

def extract_image_rgb(file):
    """Return mean (r,g,b) of image, or zeros if no image/problem."""
    if file is None:
        return 0.0, 0.0, 0.0
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))
        arr = np.array(img, dtype=float)
        r = float(arr[:, :, 0].mean())
        g = float(arr[:, :, 1].mean())
        b = float(arr[:, :, 2].mean())
        return r, g, b
    except Exception:
        return 0.0, 0.0, 0.0

# ---------- MATCHING ----------

def text_similarity(query: str, df: pd.DataFrame) -> np.ndarray:
    descriptions = df["description"].fillna("").astype(str).tolist()
    all_texts = descriptions + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(all_texts)
    query_vec = tfidf[-1]
    sims = cosine_similarity(query_vec, tfidf[:-1])[0]
    return sims


def image_similarity(q_rgb, df: pd.DataFrame) -> np.ndarray:
    qr, qg, qb = q_rgb
    if qr == qg == qb == 0.0:
        return np.zeros(len(df))
    sims = []
    for _, row in df.iterrows():
        r, g, b = row["img_r"], row["img_g"], row["img_b"]
        if r == g == b == 0.0:
            sims.append(0.0)
        else:
            dist = np.sqrt((qr - r) ** 2 + (qg - g) ** 2 + (qb - b) ** 2)
            sims.append(max(0.0, 1.0 - dist / 441.7))
    return np.array(sims)


def location_score(found_loc, lost_loc) -> float:
    f = str(found_loc or "").strip().lower()
    l = str(lost_loc or "").strip().lower()
    if not l or not f:
        return 0.0
    if l in f or f in l:
        return 1.0
    shared = set(f.split()) & set(l.split())
    return 1.0 if shared else 0.0


def date_score(found_date: str, lost_date_str: str) -> float:
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


def rank_matches(df: pd.DataFrame,
                 q_text: str,
                 q_loc: str,
                 q_date: str,
                 q_rgb) -> pd.DataFrame:
    if df.empty:
        return df

    t_sim = text_similarity(q_text, df)
    i_sim = image_similarity(q_rgb, df)
    loc_scores = np.array([location_score(row["location"], q_loc) for _, row in df.iterrows()])
    date_scores = np.array([date_score(row["date"], q_date) for _, row in df.iterrows()])

    # weights for hybrid score
    alpha, beta, gamma, delta = 0.6, 0.15, 0.15, 0.1
    hybrid = alpha * t_sim + beta * i_sim + gamma * loc_scores + delta * date_scores

    df = df.copy()
    df["text_sim"] = t_sim
    df["img_sim"] = i_sim
    df["loc_score"] = loc_scores
    df["date_score"] = date_scores
    df["hybrid_score"] = hybrid

    return df.sort_values("hybrid_score", ascending=False)

# ---------- STYLE ----------

st.set_page_config(page_title="Campus Lost & Found - AutoMatch", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f5f7fb; }
    .stButton>button {
        border-radius: 8px;
        padding: 0.3rem 0.9rem;
        font-weight: 600;
        border: 1px solid #2563eb;
        background-color: #2563eb;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Campus Lost & Found - AutoMatch")
st.caption("Hybrid text + image + location + date matching with feedback.")

menu = st.sidebar.radio(
     "Navigation",
    ["Add Found Item", "Search Lost Item", "Feedback & Logs"],
)

# ---------- ADD FOUND ITEM ----------

if menu == "Add Found Item":
    st.subheader("Add Found Item")
    items_df = load_items()

    col1, col2 = st.columns(2)
    with col1:
        desc = st.text_input("Found Item Description")
        loc = st.text_input("Location Found")
    with col2:
        date = st.date_input("Date Found")
        contact = st.text_input("Contact (Phone/Email)")

    uploaded_image = st.file_uploader(
        "Upload Image of Found Item (optional)",
        type=["jpg", "jpeg", "png"],
        key="add_image",
    )

    if st.button("Save Found Item"):
        if not desc.strip():
            st.error("Description cannot be empty.")
        else:
            items_df = load_items()
            new_id = 1 if items_df.empty else int(items_df["id"].max()) + 1
            img_r, img_g, img_b = extract_image_rgb(uploaded_image)

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

            items_df = pd.concat([items_df, pd.DataFrame([new_row])], ignore_index=True)
            save_items(items_df)
            st.success("Found item saved successfully.")

    st.markdown("### Current Stored Items")
    items_df = load_items()
    if items_df.empty:
        st.info("No items stored yet.")
    else:
        st.dataframe(
            items_df[["id", "description", "location", "date", "contact"]].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

# ---------- SEARCH LOST ITEM ----------

elif menu == "Search Lost Item":
    st.subheader("Search Lost Item")
    items_df = load_items()

    if items_df.empty:
        st.warning("No items stored yet. Please add found items first.")
    else:
        q_text = st.text_area(
            "Describe the lost item",
            placeholder="Example: Black Lenovo laptop bag with red zip",
        )
        col1, col2 = st.columns(2)
        with col1:
            q_loc = st.text_input(
                "Approximate location lost (optional)",
                placeholder="Library / CSE block / Canteen",
            )
        with col2:
            q_date = st.date_input("Approximate date lost (optional)")

        lost_image = st.file_uploader(
            "Upload image of the lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="lost_image",
        )

        top_k = st.slider("Number of matches to display", 1, 10, 5)

        if st.button("Find Matches"):
            if not q_text.strip():
                st.error("Please enter a description of the lost item.")
            else:
                q_r, q_g, q_b = extract_image_rgb(lost_image)
                q_rgb = (q_r, q_g, q_b)
                q_date_str = str(q_date) if q_date else ""

                ranked = rank_matches(items_df, q_text, q_loc, q_date_str, q_rgb)
                ranked = ranked.head(top_k)

                if ranked.empty:
                    st.warning("No matches found.")
                else:
                    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
                        st.markdown("---")
                        st.markdown(
                            f"Match {i}: Overall Score {row['hybrid_score']*100:.1f}%"
                        )
                        st.write(f"Description: {row['description']}")
                        st.write(f"Location: {row['location']}")
                        st.write(f"Date Found: {row['date']}")
                        st.write(f"Contact: {row['contact']}")

                        st.caption(
                            f"Text: {row['text_sim']*100:.1f}%  |  "
                            f"Image: {row['img_sim']*100:.1f}%  |  "
                            f"Location: {row['loc_score']*100:.0f}%  |  "
                            f"Date: {row['date_score']*100:.0f}%"
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button(f"Helpful (ID {row['id']})", key=f"good_{row['id']}"):
                                save_feedback(q_text, row["id"], row["hybrid_score"], True)
                                st.success("Thanks, feedback recorded.")
                        with c2:
                            if st.button(f"Not useful (ID {row['id']})", key=f"bad_{row['id']}"):
                                save_feedback(q_text, row["id"], row["hybrid_score"], False)
                                st.warning("Marked as not useful.")

# ---------- FEEDBACK PAGE ----------




