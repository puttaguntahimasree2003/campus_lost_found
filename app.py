import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

# ---------------------------
# Load & Save CSV
# ---------------------------
CSV_PATH = "items.csv"

def load_items():
    try:
        return pd.read_csv(CSV_PATH)
    except:
        return pd.DataFrame(columns=["id","description","location","date","contact","r","g","b"])

def save_items(df):
    df.to_csv(CSV_PATH, index=False)

# ---------------------------
# Extract image features
# ---------------------------
def extract_image_features(uploaded_img):
    if uploaded_img is None:
        return None, None, None
    img = Image.open(uploaded_img).convert("RGB")
    arr = np.array(img)
    return arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()  # R, G, B

# ---------------------------
# Text similarity
# ---------------------------
def compute_text_similarity(query, df):
    texts = df["description"].fillna("").tolist()
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts + [query])
    cosine = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    return cosine

# ---------------------------
# Image similarity
# ---------------------------
def compute_image_similarity(query_rgb, df):
    if query_rgb is None:
        return np.zeros(len(df))
    qr, qg, qb = query_rgb
    sims = []
    for _, row in df.iterrows():
        if pd.isna(row["r"]):
            sims.append(0)  
        else:
            fr, fg, fb = row["r"], row["g"], row["b"]
            dist = np.sqrt((qr-fr)*2 + (qg-fg)*2 + (qb-fb)*2)
            sim = max(0, 1 - dist/441.67)  
            sims.append(sim)
    return np.array(sims)

# ---------------------------
# UI STARTS HERE
# ---------------------------
st.set_page_config(page_title="Campus Lost & Found with AutoMatch", layout="wide")

st.title("🏛️ Campus Lost & Found with AutoMatch")

menu = st.sidebar.radio("Choose an action:", ["➕ Add Found Item", "🔍 Search Lost Item"])

df = load_items()

# --------------------------------------------------------
# ADD FOUND ITEM
# --------------------------------------------------------
if menu == "➕ Add Found Item":

    st.header("Add Found Item")

    desc = st.text_input("Found Item Description")
    loc = st.text_input("Location Found")
    date = st.date_input("Date Found")
    contact = st.text_input("Contact Number")

    uploaded_img = st.file_uploader("Upload Image (optional)", type=["jpg","jpeg","png"], key="add_img")

    # extract rgb
    if uploaded_img:
        r, g, b = extract_image_features(uploaded_img)
    else:
        r, g, b = None, None, None

    if st.button("Save Found Item"):
        new_id = 1 if df.empty else df["id"].max() + 1

        new_row = {
            "id": new_id,
            "description": desc,
            "location": loc,
            "date": str(date),
            "contact": contact,
            "r": r,
            "g": g,
            "b": b
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_items(df)
        st.success("Item added successfully! 🎉")

# --------------------------------------------------------
# SEARCH LOST ITEM
# --------------------------------------------------------
elif menu == "🔍 Search Lost Item":

    st.header("Search Lost Item")

    query = st.text_input("Describe what you lost")
    lost_img = st.file_uploader("Upload Lost Item Image (optional)", type=["jpg","jpeg","png"], key="search_img")

    if lost_img:
        qr, qg, qb = extract_image_features(lost_img)
        query_rgb = (qr, qg, qb)
    else:
        query_rgb = None

    top_k = st.slider("Number of matches to show:", 1, 10, 3)

    if st.button("Find Matches"):

        if df.empty:
            st.warning("No items in database!")
        else:
            text_sim = compute_text_similarity(query, df)
            img_sim = compute_image_similarity(query_rgb, df)

            final_score = (0.7 * text_sim) + (0.3 * img_sim)

            df["score"] = final_score
            results = df.sort_values("score", ascending=False).head(top_k)

            st.subheader("🔎 Best Matches")
            for _, row in results.iterrows():
                st.markdown(f"""
                *Match Score: {round(row['score']*100,2)}%*  
                *Description:* {row['description']}  
                *Location:* {row['location']}  
                *Date:* {row['date']}  
                *Contact:* {row['contact']}  
                """)


