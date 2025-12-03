import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- WordNet (optional) ----------
try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.download("wordnet", quiet=True)
    WORDNET = True
except:
    WORDNET = False

DATA_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"

COLUMNS = ["id", "description", "location", "date", "contact", "img_r", "img_g", "img_b"]

# ---------- Utilities ----------
def load_items():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0 if c.startswith("img_") else ""

    df = df[COLUMNS]
    df["id"] = df["id"].fillna(0).astype(int)
    return df


def save_items(df):
    df.to_csv(DATA_FILE, index=False)


def get_image_features(file):
    if file is None:
        return [0, 0, 0]
    try:
        img = Image.open(file).convert("RGB").resize((128, 128))
        arr = np.array(img)
        return [arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()]
    except:
        return [0, 0, 0]


def expand_text(text):
    if not WORDNET:
        return text
    words = text.split()
    extra = []
    for w in words:
        syns = wn.synsets(w)
        for s in syns[:1]:
            extra.append(s.lemmas()[0].name().replace("_", " "))
    return text + " " + " ".join(extra)


def hybrid_match(q_text, q_img, q_loc, q_date):
    df = load_items()
    if df.empty:
        return df

    # text similarity
    query_expanded = expand_text(q_text)
    texts = df["description"].astype(str).tolist() + [query_expanded]

    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(texts)
    text_scores = cosine_similarity(mat[-1], mat[:-1])[0]

    # image similarity
    img_feats = df[["img_r", "img_g", "img_b"]].values
    q_vec = np.array(q_img).reshape(1, -1)
    img_scores = cosine_similarity(q_vec, img_feats)[0]

    # location match (1 or 0)
    loc_scores = (df["location"].str.lower() == q_loc.lower()).astype(float)

    # date closeness
    ds = []
    for d in df["date"]:
        try:
            diff = abs((pd.to_datetime(q_date) - pd.to_datetime(d)).days)
            ds.append(max(0, 1 - diff/7))
        except:
            ds.append(0)
    date_scores = np.array(ds)

    # hybrid score
    final = 0.6*text_scores + 0.15*img_scores + 0.15*loc_scores + 0.1*date_scores

    df["text"] = text_scores
    df["img"] = img_scores
    df["loc"] = loc_scores
    df["date_s"] = date_scores
    df["score"] = final

    return df.sort_values("score", ascending=False)


def save_feedback(item_id, score, good):
    fb = pd.DataFrame([[item_id, score, "good" if good else "bad"]],
                      columns=["item_id", "score", "feedback"])
    if os.path.exists(FEEDBACK_FILE):
        old = pd.read_csv(FEEDBACK_FILE)
        fb = pd.concat([old, fb], ignore_index=True)
    fb.to_csv(FEEDBACK_FILE, index=False)


# ---------- UI ----------
st.title("🏛️Campus Lost & Found with AutoMatch)")

choice = st.sidebar.radio("Menu", ["➕ Add Found Item", "🔍 Search Lost Item", "📊 Feedback"])


# ADD FOUND ITEM
if choice == "➕ Add Found Item":
    desc = st.text_area("Description")
    loc = st.text_input("Location Found")
    date = st.date_input("Date Found")
    contact = st.text_input("Contact")
    img = st.file_uploader("Upload Image (optional)", type=["jpg","jpeg","png"])

    if st.button("Save"):
        df = load_items()
        new_id = 1 if df.empty else df["id"].max()+1
        r,g,b = get_image_features(img)

        new = {
            "id": new_id,
            "description": desc,
            "location": loc,
            "date": str(date),
            "contact": contact,
            "img_r": r, "img_g": g, "img_b": b
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        save_items(df)
        st.success("Item saved successfully!")

    st.write("### Stored Items")
    df = load_items()
    st.dataframe(df[["id","description","location","date","contact"]], hide_index=True)


# SEARCH LOST ITEM
elif choice == "🔍 Search Lost Item":
    q_text = st.text_area("Describe lost item")
    q_loc = st.text_input("Lost Location")
    q_date = st.date_input("Lost Date")
    q_img = st.file_uploader("Upload Image (optional)", type=["jpg","jpeg","png"])
    k = st.slider("Number of matches", 1, 10, 3)

    if st.button("Search"):
        q_img_feat = get_image_features(q_img)
        res = hybrid_match(q_text, q_img_feat, q_loc, q_date).head(k)

        for _, r in res.iterrows():
            st.markdown("### 🎯 Match Score: *{:.2f}%*".format(r["score"]*100))
            st.write("*Description:*", r["description"])
            st.write("*Location:*", r["location"])
            st.write("*Date Found:*", r["date"])
            st.write("*Contact:*", r["contact"])
            st.caption(
                f"Text: {r['text']*100:.1f}% | Image: {r['img']*100:.1f}% | Location: {r['loc']*100:.1f}% | Date: {r['date_s']*100:.1f}%"
            )

            c1,c2 = st.columns(2)
            if c1.button(f"👍 Helpful {r['id']}"):
                save_feedback(r["id"], r["score"], True)
            if c2.button(f"👎 Not useful {r['id']}"):
                save_feedback(r["id"], r["score"], False)


# FEEDBACK
else:
    if os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(fb,hide_index=True)
    else:
        st.info("No feedback yet.")

