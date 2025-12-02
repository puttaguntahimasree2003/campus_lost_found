import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FOUND_FILE = "items.csv"

# ---------------- DATA FUNCTIONS ----------------

def load_found_items():
    """Load found items from CSV or return empty DataFrame."""
    if os.path.exists(FOUND_FILE):
        return pd.read_csv(FOUND_FILE)
    else:
        return pd.DataFrame(columns=["id", "description", "location", "date", "contact"])


def save_found_item(description, location, date, contact):
    """Save a new found item to CSV."""
    df = load_found_items()
    new_id = len(df) + 1

    new_row = {
        "id": new_id,
        "description": description,
        "location": location,
        "date": str(date),
        "contact": contact,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(FOUND_FILE, index=False)


def find_matches(lost_description, top_k=5):
    """Match lost description against all found items."""
    df = load_found_items()
    if df.empty:
        return None

    descriptions = [lost_description] + df["description"].fillna("").tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    lost_vec = tfidf_matrix[0:1]
    found_vecs = tfidf_matrix[1:]

    sims = cosine_similarity(lost_vec, found_vecs).flatten()
    df["similarity"] = sims

    df = df.sort_values(by="similarity", ascending=False)
    return df.head(top_k)

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Campus Lost & Found - AutoMatch", layout="wide")
st.title("🎒 Campus Lost and Found with AutoMatch")
st.write("Matches lost item descriptions with found items using TF-IDF + cosine similarity.")

mode = st.sidebar.radio("Choose an action:", ["➕ Add Found Item", "🔍 Search Lost Item"])

# -------- ADD FOUND ITEM PAGE --------

if mode == "➕ Add Found Item":
    st.header("➕ Add Found Item")

    description = st.text_area("Found Item Description", placeholder="Example: Black Lenovo laptop bag with red zip")
    location = st.text_input("Location Found", placeholder="Library stairs")
    date = st.date_input("Date Found")
    contact = st.text_input("Contact (Phone/Email)", placeholder="9876543210")

    if st.button("Save Found Item"):
        if description.strip() == "":
            st.error("Description cannot be empty!")
        else:
            save_found_item(description, location, date, contact)
            st.success("✅ Found item saved!")

    st.subheader("📋 Current Found Items")
    df = load_found_items()
    if df.empty:
        st.info("No found items added yet.")
    else:
        df_display = df.copy()
        df_display["id"] = df_display["id"].astype(str)
        df_display = df_display[["id", "description", "location", "date", "contact"]]
        st.dataframe(df_display, hide_index=True, use_container_width=True)

# -------- SEARCH LOST ITEM PAGE --------

elif mode == "🔍 Search Lost Item":
    st.header("🔍 Search Lost Item")

    lost_description = st.text_area("Describe the lost item", placeholder="Black Lenovo laptop bag with red zip")
    top_k = st.slider("How many matches to show?", 1, 10, 5)

    if st.button("Find Matches"):
        if lost_description.strip() == "":
            st.error("Please describe your lost item.")
        else:
            matches = find_matches(lost_description, top_k)
            if matches is None or matches.empty:
                st.warning("No found items to match with!")
            else:
                st.success(f"Showing top {len(matches)} matches:")

                for _, row in matches.iterrows():
                    st.markdown("---")
                    st.markdown(f"### 🎯 Match Score: **{round(row['similarity']*100, 1)}%**")
                    st.write(f"**id:** {row['id']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**Date:** {row['date']}")

                    st.write(f"**Contact:** {row['contact']}")


