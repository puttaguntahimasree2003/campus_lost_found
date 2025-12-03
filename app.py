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

# --------- helpers ---------

def load_items():
    cols = ["id", "description", "location", "date", "contact", "img_r", "img_g", "img_b"]
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0 if c in ["img_r", "img_g", "img_b"] else ""
    df = df[cols]
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    if (df["id"] == 0).any():
        df["id"] = range(1, len(df) + 1)
    for c in ["img_r", "img_g", "img_b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["description", "location", "date", "contact"]:
        df[c] = df[c].fillna("").astype(str)
    return df

def save_items(df):
    df.to_csv(DATA_FILE, index=False)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["lost_query", "matched_id", "score", "feedback", "timestamp"])

def save_feedback(lost_query, matched_id, score, good):
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

def extract_rgb(file):
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

def text_sim(query, df):
    descs = df["description"].fillna("").astype(str).tolist()
    all_text = descs + [query]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(all_text)
    q_vec = tfidf[-1]
    sims = cosine_similarity(q_vec, tfidf[:-1])[0]
    return sims

def img_sim(q_rgb, df):
    qr, qg, qb = q_rgb
    if qr == qg == qb == 0.0:
        return np.zeros(len(df))
    out = []
    for _, row in df.iterrows():
        r, g, b = row["img_r"], row["img_g"], row["img_b"]
        if r == g == b == 0.0:
            out.append(0.0)
        else:
            d = np.sqrt((qr - r) ** 2 + (qg - g) ** 2 + (qb - b) ** 2)
            out.append(max(0.0, 1.0 - d / 441.7))
    return np.array(out)

def loc_score(found_loc, lost_loc):
    f = str(found_loc or "").strip().lower()
    l = str(lost_loc or "").strip().lower()
    if not l or not f:
        return 0.0
    if l in f or f in l:
        return 1.0
    if set(f.split()) & set(l.split()):
        return 1.0
    return 0.0

def date_score(found_date, lost_date):
    if not lost_date:
        return 0.0
    try:
        d_f = datetime.fromisoformat(str(found_date)).date()
        d_l = datetime.fromisoformat(str(lost_date)).date()
        diff = abs((d_f - d_l).days)
        if diff == 0:
            return 1.0
        if diff <= 2:
            return 0.7
        if diff <= 7:
            return 0.4
        return 0.0
    except Exception:
        return 0.0

def rank_matches(df, q_text, q_loc, q_date, q_rgb):
    if df.empty:
        return df
    ts = text_sim(q_text, df)
    is_ = img_sim(q_rgb, df)
    ls = np.array([loc_score(row["location"], q_loc) for _, row in df.iterrows()])
    ds = np.array([date_score(row["date"], q_date) for _, row in df.iterrows()])
    alpha, beta, gamma, delta = 0.6, 0.15, 0.15, 0.1
    hybrid = alpha * ts + beta * is_ + gamma * ls + delta * ds
    df = df.copy()
    df["text_sim"] = ts
    df["img_sim"] = is_
    df["loc_score"] = ls
    df["date_score"] = ds
    df["hybrid_score"] = hybrid
    return df.sort_values("hybrid_score", ascending=False)

# --------- style ---------

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

st.title("Campus Lost & Found with AutoMatch")


menu = st.sidebar.radio("Navigation", ["Add Found Item", "Search Lost Item", "Feedback & Logs"])

# --------- Add page ---------

if menu == "Add Found Item":
    st.subheader("Add Found Item")
    df = load_items()

    c1, c2 = st.columns(2)
    with c1:
        desc = st.text_input("Found Item Description")
        loc = st.text_input("Location Found")
    with c2:
        date = st.date_input("Date Found")
        contact = st.text_input("Contact (Phone/Email)")

    img_file = st.file_uploader(
        "Upload Image of Found Item (optional)", type=["jpg", "jpeg", "png"], key="add_image"
    )

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
            st.success("Found item saved.")

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

# --------- Search page ---------

elif menu == "Search Lost Item":
    st.subheader("Search Lost Item")
    df = load_items()
    if df.empty:
        st.warning("No items stored yet. Please add found items first.")
    else:
        q_text = st.text_area(
            "Describe the lost item",
            placeholder="Example: Black Lenovo laptop bag with red zip",
        )
        c1, c2 = st.columns(2)
        with c1:
            q_loc = st.text_input(
                "Approximate location lost (optional)", placeholder="Library / CSE block / Canteen"
            )
        with c2:
            q_date = st.date_input("Approximate date lost (optional)")
        lost_img = st.file_uploader(
            "Upload image of the lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="lost_image",
        )
        top_k = st.slider("Number of matches to display", 1, 10, 5)

        if st.button("Find Matches"):
            if not q_text.strip():
                st.error("Please enter a description.")
            else:
                qr, qg, qb = extract_rgb(lost_img)
                q_rgb = (qr, qg, qb)
                q_date_str = str(q_date) if q_date else ""
                ranked = rank_matches(df, q_text, q_loc, q_date_str, q_rgb).head(top_k)

                if ranked.empty:
                    st.warning("No matches found.")
                else:
                    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
                        st.markdown("---")
                        st.markdown(f"Match {i}: Overall Score {row['hybrid_score']*100:.1f}%")
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

                        c1b, c2b = st.columns(2)
                        with c1b:
                            if st.button(f"Helpful (ID {row['id']})", key=f"good_{row['id']}"):
                                save_feedback(q_text, row["id"], row["hybrid_score"], True)
                                st.success("Feedback recorded.")
                        with c2b:
                            if st.button(f"Not useful (ID {row['id']})", key=f"bad_{row['id']}"):
                                save_feedback(q_text, row["id"], row["hybrid_score"], False)
                                st.warning("Marked as not useful.")

# --------- Feedback page ---------

else:
    st.subheader("Feedback & Logs")
    fb = load_feedback()
    if fb.empty:
        st.info("No feedback given yet.")
    else:
        total = len(fb)
        good = (fb["feedback"] == "good").sum()
        bad = (fb["feedback"] == "bad").sum()
        st.write(f"Total feedback entries: {total}")
        st.write(f"Marked helpful: {good}")
        st.write(f"Marked not useful: {bad}")
        if total > 0:
            st.write(f"Overall positive rate: {(good / total) * 100:.1f}%")
        st.markdown("### Raw Feedback")
        st.dataframe(fb.reset_index(drop=True), hide_index=True, use_container_width=True)

