import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import nltk
from nltk.corpus import wordnet

# -------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------
nltk.download("wordnet")

st.set_page_config(page_title="Campus Lost & Found – AutoMatch", layout="wide")

DATA_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"


# -------------------------------------------------
# LOAD & SAVE FUNCTIONS
# -------------------------------------------------
@st.cache_data
def load_items():
    try:
        df = pd.read_csv(DATA_FILE)
    except:
        df = pd.DataFrame(columns=["id", "description", "location", "date", "contact"])
    return df


def save_items(df):
    df.to_csv(DATA_FILE, index=False)


def save_feedback(row_id, score):
    fb = pd.DataFrame([[row_id, score]], columns=["item_id", "feedback"])
    try:
        old = pd.read_csv(FEEDBACK_FILE)
        fb = pd.concat([old, fb], ignore_index=True)
    except:
        pass
    fb.to_csv(FEEDBACK_FILE, index=False)


# -------------------------------------------------
# IMAGE FEATURE EXTRACTOR (MEAN RGB)
# -------------------------------------------------
def get_image_features(uploaded):
    if uploaded is None:
        return [0, 0, 0]

    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()
    return [r, g, b]


# -------------------------------------------------
# TEXT SYNONYM EXPANSION
# -------------------------------------------------
def expand_text(text):
    words = text.split()
    expanded = words.copy()

    for w in words:
        syns = wordnet.synsets(w)
        for s in syns[:1]:  # only take one synonym to avoid over-expansion
            lemma = s.lemmas()[0].name().replace("_", " ")
            expanded.append(lemma)

    return " ".join(expanded)


# -------------------------------------------------
# HYBRID MATCHING FUNCTION
# -------------------------------------------------
def match_items(query_text, query_img, query_loc, query_date, top_k=5):
    df = load_items()

    # Expand query text
    query_text_expanded = expand_text(query_text)

    # TF-IDF Text matching
    texts = df["description"].astype(str).tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts + [query_text_expanded])
    query_vec = tfidf_matrix[-1]
    text_sims = cosine_similarity(query_vec, tfidf_matrix[:-1])[0]

    # Image similarity
    q_r, q_g, q_b = query_img
    img_sims = []
    for _ in range(len(df)):
        img_sims.append(0)  # old items have no images → similarity = 0
    img_sims = np.array(img_sims)

    # Location match (binary)
    loc_sims = np.where(df["location"].str.lower() == query_loc.lower(), 1, 0)

    # Date score (closer = higher)
    date_sims = []
    for d in df["date"]:
        try:
            diff = abs(pd.to_datetime(query_date) - pd.to_datetime(d)).days
            date_sims.append(max(0, 1 - diff / 7))
        except:
            date_sims.append(0)
    date_sims = np.array(date_sims)

    # Hybrid final score
    final_score = (
        0.6 * text_sims +
        0.15 * img_sims +
        0.15 * loc_sims +
        0.1 * date_sims
    )

    df["text_score"] = text_sims
    df["img_score"] = img_sims
    df["loc_score"] = loc_sims
    df["date_score"] = date_sims
    df["final_score"] = final_score

    return df.sort_values("final_score", ascending=False).head(top_k)


# -------------------------------------------------
# SIDEBAR OPTIONS
# -------------------------------------------------
st.sidebar.title("Choose an action:")
mode = st.sidebar.radio("", ["➕ Add Found Item", "🔍 Search Lost Item", "📊 Feedback & Logs"])


# -------------------------------------------------
# ADD FOUND ITEM PAGE
# -------------------------------------------------
if mode == "➕ Add Found Item":
    st.title("➕ Add Found Item")

    desc = st.text_area("Found Item Description")
    loc = st.text_input("Location Found")
    date = st.date_input("Date Found")
    contact = st.text_input("Contact Number")
    img_upload = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])

    if st.button("Save Found Item"):
        df = load_items()
        new_id = 1 if df.empty else df["id"].max() + 1

        r, g, b = get_image_features(img_upload)

        new_row = {
            "id": new_id,
            "description": desc,
            "location": loc,
            "date": str(date),
            "contact": contact,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_items(df)

        st.success("Item saved successfully!")

    st.subheader("Current Stored Items")
    st.dataframe(load_items())


# -------------------------------------------------
# SEARCH LOST ITEM PAGE
# -------------------------------------------------
if mode == "🔍 Search Lost Item":
    st.title("🔍 Search Lost Item")

    q_text = st.text_area("Describe the lost item (text)")
    q_loc = st.text_input("Location")
    q_date = st.date_input("Date Lost")
    q_img = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
    top_k = st.slider("Number of matches to display", 1, 10, 5)

    if st.button("Find Matches"):
        q_img_feat = get_image_features(q_img)
        results = match_items(q_text, q_img_feat, q_loc, q_date, top_k)

        st.success("Matches Found:")

        for _, row in results.iterrows():
            st.markdown(f"### 🎯 Match Score: *{row['final_score']*100:.2f}%*")
            st.write(f"*Description:* {row['description']}")
            st.write(f"*Location:* {row['location']}")
            st.write(f"*Date Found:* {row['date']}")
            st.write(f"*Contact:* {row['contact']}")

            st.caption(
                f"Text: {row['text_score']*100:.1f}% | "
                f"Location: {row['loc_score']*100:.1f}% | "
                f"Date: {row['date_score']*100:.1f}% | "
                f"Image: {row['img_score']*100:.1f}%"
            )

            # Feedback buttons
            c1, c2 = st.columns(2)
            if c1.button(f"👍 Helpful (ID {row['id']})"):
                save_feedback(row["id"], 1)
                st.success("Thank you for your feedback!")
            if c2.button(f"👎 Not Useful (ID {row['id']})"):
                save_feedback(row["id"], 0)
                st.warning("Feedback submitted.")


# -------------------------------------------------
# FEEDBACK PAGE
# -------------------------------------------------
if mode == "📊 Feedback & Logs":
    st.title("📊 Feedback & Logs")

    try:
        fb = pd.read_csv(FEEDBACK_FILE)
        st.subheader("Feedback Summary")
        st.write(f"Total Feedback: {len(fb)}")
        st.write(f"👍 Helpful: {(fb['feedback']==1).sum()}")
        st.write(f"👎 Not Useful: {(fb['feedback']==0).sum()}")
        st.dataframe(fb)
    except:
        st.info("No feedback yet.")
