import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# BASIC PAGE CONFIG + SIMPLE BORDER STYLE
# ----------------------------------------------------
st.set_page_config(page_title="Campus Lost & Found - AutoMatch", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f1f5f9;
    }
    .block-container {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-top: 0.75rem;
        padding: 1.3rem 1.5rem 1.5rem 1.5rem;
        background-color: #ffffff;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.35rem 1.2rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #2563eb, #9333ea);
        color: white;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #7e22ce);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# CONSTANTS & FILES
# ----------------------------------------------------
ITEMS_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"

ITEM_COLS = ["id", "description", "location", "date", "contact", "img_r", "img_g", "img_b"]


# ----------------------------------------------------
# DATA HELPERS
# ----------------------------------------------------
def load_items() -> pd.DataFrame:
    """Load items CSV and guarantee required columns exist."""
    if os.path.exists(ITEMS_FILE):
        df = pd.read_csv(ITEMS_FILE)
    else:
        df = pd.DataFrame(columns=ITEM_COLS)

    for col in ITEM_COLS:
        if col not in df.columns:
            if col in ["img_r", "img_g", "img_b"]:
                df[col] = 0.0
            elif col == "id":
                df[col] = 0
            else:
                df[col] = ""

    df = df[ITEM_COLS]

    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    if (df["id"] == 0).any():
        df["id"] = range(1, len(df) + 1)

    for c in ["img_r", "img_g", "img_b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["description", "location", "date", "contact"]:
        df[c] = df[c].fillna("").astype(str)

    return df


def save_items(df: pd.DataFrame) -> None:
    df.to_csv(ITEMS_FILE, index=False)


def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["lost_query", "matched_id", "score", "feedback", "timestamp"])


def append_feedback(lost_query: str, matched_id: int, score: float, good: bool) -> None:
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


# ----------------------------------------------------
# IMAGE & SIMILARITY HELPERS
# ----------------------------------------------------
def extract_rgb(uploaded_file):
    """Return mean RGB of uploaded image, or (0,0,0) if none or error."""
    if uploaded_file is None:
        return 0.0, 0.0, 0.0

    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))
        arr = np.array(img, dtype=float)
        r = float(arr[:, :, 0].mean())
        g = float(arr[:, :, 1].mean())
        b = float(arr[:, :, 2].mean())
        return r, g, b
    except Exception:
        return 0.0, 0.0, 0.0


def text_similarity_scores(query: str, df: pd.DataFrame) -> np.ndarray:
    descriptions = df["description"].fillna("").astype(str).tolist()
    all_texts = descriptions + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(all_texts)
    query_vec = tfidf[-1]
    sims = cosine_similarity(query_vec, tfidf[:-1])[0]
    return sims


def image_similarity_scores(q_rgb, df: pd.DataFrame) -> np.ndarray:
    qr, qg, qb = q_rgb
    if qr == qg == qb == 0.0:
        return np.zeros(len(df))

    scores = []
    for _, row in df.iterrows():
        r, g, b = row["img_r"], row["img_g"], row["img_b"]
        if r == g == b == 0.0:
            scores.append(0.0)
        else:
            dist = np.sqrt((qr - r) ** 2 + (qg - g) ** 2 + (qb - b) ** 2)
            scores.append(1.0 / (1.0 + dist))
    return np.array(scores)


def location_score(found_loc: str, lost_loc: str) -> float:
    f = str(found_loc or "").strip().lower()
    l = str(lost_loc or "").strip().lower()
    if not f or not l:
        return 0.0
    if f == l or l in f or f in l:
        return 1.0
    if set(f.split()) & set(l.split()):
        return 1.0
    return 0.0


def date_score(found_date: str, lost_date: str) -> float:
    if not lost_date:
        return 0.0
    try:
        d_found = datetime.fromisoformat(str(found_date)).date()
        d_lost = datetime.fromisoformat(str(lost_date)).date()
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


def rank_matches(df, q_text, q_loc, q_date_str, q_rgb):
    if df.empty:
        return df

    t_sim = text_similarity_scores(q_text, df)
    i_sim = image_similarity_scores(q_rgb, df)
    l_scores = np.array([location_score(row["location"], q_loc) for _, row in df.iterrows()])
    d_scores = np.array([date_score(row["date"], q_date_str) for _, row in df.iterrows()])

    alpha, beta, gamma, delta = 0.6, 0.15, 0.15, 0.1
    hybrid = alpha * t_sim + beta * i_sim + gamma * l_scores + delta * d_scores

    df = df.copy()
    df["text_sim"] = t_sim
    df["img_sim"] = i_sim
    df["loc_score"] = l_scores
    df["date_score"] = d_scores
    df["hybrid_score"] = hybrid

    return df.sort_values("hybrid_score", ascending=False)


# ----------------------------------------------------
# PAGE: ADD FOUND ITEM
# ----------------------------------------------------
def add_found_item_page():
    st.subheader("➕ Add Found Item")

    df = load_items()

    col1, col2 = st.columns(2)
    with col1:
        desc = st.text_input("Found Item Description")
        loc = st.text_input("Location Found")
    with col2:
        date = st.date_input("Date Found")
        contact = st.text_input("Contact (Phone/Email)")

    img_file = st.file_uploader(
        "Upload Image of Found Item (optional)",
        type=["jpg", "jpeg", "png"],
        key="add_image",
    )
    if img_file is not None:
        st.markdown("**Preview**")
        st.image(img_file, caption="Found item image", width=220)

    if st.button("Save Found Item"):
        if not desc.strip():
            st.error("Description cannot be empty.")
        else:
            df = load_items()
            new_id = 1 if df.empty else int(df["id"].max()) + 1
            r, g, b = extract_rgb(img_file)
            new_row = {
                "id": new_id,
                "description": desc,
                "location": loc,
                "date": str(date),
                "contact": contact,
                "img_r": r,
                "img_g": g,
                "img_b": b,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_items(df)
            st.success("Found item saved successfully ✅")

    st.markdown("### Current Stored Items")
    df = load_items()
    if df.empty:
        st.info("No items stored yet.")
    else:
        st.dataframe(
            df[["id", "description", "location", "date", "contact"]].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )


# ----------------------------------------------------
# PAGE: SEARCH LOST ITEM
# ----------------------------------------------------
def search_lost_item_page():
    st.subheader("🔍 Search Lost Item")

    df = load_items()
    if df.empty:
        st.warning("No items stored yet. Please add found items first.")
        return

    q_text = st.text_area(
        "Describe the lost item",
        placeholder="Example: Blue steel water bottle with a dent on the side",
    )

    col1, col2 = st.columns(2)
    with col1:
        q_loc = st.text_input(
            "Approximate location lost (optional)",
            placeholder="Library / CSE block / Canteen",
        )
    with col2:
        q_date = st.date_input("Approximate date lost (optional)")

    lost_img = st.file_uploader(
        "Upload image of lost item (optional)",
        type=["jpg", "jpeg", "png"],
        key="lost_image",
    )
    if lost_img is not None:
        st.markdown("**Preview**")
        st.image(lost_img, caption="Lost item image", width=220)

    top_k = st.slider("Number of matches to display", 1, 10, 5)

    if st.button("Find Matches"):
        if not q_text.strip():
            st.error("Please enter a description of the lost item.")
            return

        q_r, q_g, q_b = extract_rgb(lost_img)
        q_rgb = (q_r, q_g, q_b)
        q_date_str = str(q_date) if q_date else ""

        ranked = rank_matches(df, q_text, q_loc, q_date_str, q_rgb).head(top_k)

        if ranked.empty:
            st.warning("No matches found.")
        else:
            st.markdown("### Best Matches")
            for i, (_, row) in enumerate(ranked.iterrows(), start=1):
                st.markdown(
                    f"**Match {i}: Overall Score {row['hybrid_score'] * 100:.1f}%**"
                )
                st.write(f"Description: {row['description']}")
                st.write(f"Location: {row['location']}")
                st.write(f"Date Found: {row['date']}")
                st.write(f"Contact: {row['contact']}")
                st.write(
                    f"Text: {row['text_sim'] * 100:.1f}% | "
                    f"Image: {row['img_sim'] * 100:.1f}% | "
                    f"Location: {row['loc_score'] * 100:.0f}% | "
                    f"Date: {row['date_score'] * 100:.0f}%"
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"Helpful (ID {row['id']})", key=f"good_{row['id']}"):
                        append_feedback(q_text, row["id"], row["hybrid_score"], True)
                        st.success("Thanks, feedback recorded.")
                with c2:
                    if st.button(f"Not useful (ID {row['id']})", key=f"bad_{row['id']}"):
                        append_feedback(q_text, row["id"], row["hybrid_score"], False)
                        st.warning("Marked as not useful.")

                st.markdown("---")


# ----------------------------------------------------
# PAGE: FEEDBACK LOGS
# ----------------------------------------------------
def feedback_page():
    st.subheader("📝 Feedback & Logs")

    fb = load_feedback()
    if fb.empty:
        st.info("No feedback given yet.")
        return

    total = len(fb)
    good = (fb["feedback"] == "good").sum()
    bad = (fb["feedback"] == "bad").sum()

    st.write(f"Total feedback entries: {total}")
    st.write(f"Marked helpful: {good}")
    st.write(f"Marked not useful: {bad}")
    if total > 0:
        st.write(f"Overall positive rate: {(good / total) * 100:.1f}%")

    st.markdown("### Raw Feedback")
    st.dataframe(
        fb.reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )
