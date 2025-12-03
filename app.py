# app.py
# Campus Lost & Found – Persistent Version
# - items saved in items.csv (never reset)
# - images stored in /images folder
# - add / edit / delete items
# - feedback stored in feedback.csv
# - text AutoMatch using TF-IDF
# - search and feedback are separated

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# BASIC CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Campus Lost & Found", layout="wide")

ITEMS_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"
IMAGE_DIR = "images"

os.makedirs(IMAGE_DIR, exist_ok=True)


# ----------------------------------------------------
# HELPERS TO LOAD & SAVE DATA (PERSISTENT)
# ----------------------------------------------------
def load_items():
    """Load items.csv into session_state.items_df and ensure required columns exist."""
    if "items_df" in st.session_state:
        return

    if os.path.exists(ITEMS_FILE):
        df = pd.read_csv(ITEMS_FILE)
    else:
        df = pd.DataFrame(
            columns=["id", "description", "location", "date", "contact", "image"]
        )

    required_cols = ["id", "description", "location", "date", "contact", "image"]
    for col in required_cols:
        if col not in df.columns:
            if col == "id":
                df[col] = np.arange(1, len(df) + 1)
            else:
                df[col] = ""

    try:
        df["id"] = df["id"].astype(int)
    except Exception:
        pass

    st.session_state.items_df = df
    st.session_state.items_df.to_csv(ITEMS_FILE, index=False)


def load_feedback():
    """Load feedback.csv into session_state.feedback_df."""
    if "feedback_df" in st.session_state:
        return

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
    else:
        df = pd.DataFrame(columns=["time", "item_id", "rating", "comment"])

    st.session_state.feedback_df = df
    st.session_state.feedback_df.to_csv(FEEDBACK_FILE, index=False)


def save_items():
    st.session_state.items_df.to_csv(ITEMS_FILE, index=False)


def save_feedback():
    st.session_state.feedback_df.to_csv(FEEDBACK_FILE, index=False)


# ----------------------------------------------------
# INITIAL LOAD
# ----------------------------------------------------
load_items()
load_feedback()

items_df = st.session_state.items_df
feedback_df = st.session_state.feedback_df

# ----------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------
st.title("🏫 Campus Lost & Found – AutoMatch + Feedback (Persistent)")


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_add, tab_manage, tab_search, tab_feedback = st.tabs(
    ["➕ Add Item", "📝 View / Edit / Delete", "🔍 Search Items", "⭐ Feedback"]
)


# ----------------------------------------------------
# TAB 1: ADD ITEM
# ----------------------------------------------------
with tab_add:
    st.subheader("Add a new lost / found item")

    with st.form("add_item_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            description = st.text_input(
                "Item description*", placeholder="Red water bottle with scratches..."
            )
            location = st.text_input(
                "Location found*", placeholder="Girls Hostel"
            )
            date_found = st.date_input("Date*", value=datetime.today())
        with col2:
            contact = st.text_input(
                "Contact number*", placeholder="9876543210"
            )
            uploaded_image = st.file_uploader(
                "Item image (optional)", type=["png", "jpg", "jpeg"]
            )

        submitted = st.form_submit_button("Add item ✅")

    if submitted:
        if description.strip() == "" or location.strip() == "" or contact.strip() == "":
            st.error("Please fill description, location and contact.")
        else:
            image_path = ""
            if uploaded_image is not None:
                safe_name = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_image.name}"
                )
                image_path = os.path.join(IMAGE_DIR, safe_name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

            if items_df.empty:
                new_id = 1
            else:
                new_id = int(items_df["id"].max()) + 1

            new_row = {
                "id": new_id,
                "description": description.strip(),
                "location": location.strip(),
                "date": date_found.strftime("%Y-%m-%d"),
                "contact": contact.strip(),
                "image": image_path,
            }

            st.session_state.items_df = pd.concat(
                [st.session_state.items_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )
            items_df = st.session_state.items_df
            save_items()

            st.success(f"Item #{new_id} added and saved 🎉")


# ----------------------------------------------------
# TAB 2: VIEW / EDIT / DELETE
# ----------------------------------------------------
with tab_manage:
    st.subheader("All items (persistent)")

    items_df = st.session_state.items_df

    if items_df.empty:
        st.info("No items yet. Add some from the 'Add Item' tab.")
    else:
        # show only selected columns, hide index
        cols_to_show = ["id", "description", "location", "date", "contact"]
        existing_cols = [c for c in cols_to_show if c in items_df.columns]

        st.dataframe(
            items_df[existing_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Edit or Delete an item")

        ids = items_df["id"].tolist()
        selected_id = st.selectbox("Select item ID", ids)

        row = items_df[items_df["id"] == selected_id].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            new_desc = st.text_input(
                "Description", row["description"], key=f"edit_desc_{selected_id}"
            )
            new_loc = st.text_input(
                "Location", row["location"], key=f"edit_loc_{selected_id}"
            )
            new_date = st.text_input(
                "Date (YYYY-MM-DD)", row["date"], key=f"edit_date_{selected_id}"
            )
        with col2:
            new_contact = st.text_input(
                "Contact", row["contact"], key=f"edit_contact_{selected_id}"
            )

            st.write("Current image:")
            if (
                isinstance(row["image"], str)
                and row["image"] != ""
                and os.path.exists(row["image"])
            ):
                st.image(row["image"], width=150)
            else:
                st.caption("No image")

            new_image_upload = st.file_uploader(
                "Replace image (optional)",
                type=["png", "jpg", "jpeg"],
                key=f"edit_image_{selected_id}",
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Save changes", key=f"save_{selected_id}"):
                idx = items_df.index[items_df["id"] == selected_id][0]

                st.session_state.items_df.at[idx, "description"] = new_desc
                st.session_state.items_df.at[idx, "location"] = new_loc
                st.session_state.items_df.at[idx, "date"] = new_date
                st.session_state.items_df.at[idx, "contact"] = new_contact

                if new_image_upload is not None:
                    safe_name = (
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{new_image_upload.name}"
                    )
                    new_path = os.path.join(IMAGE_DIR, safe_name)
                    with open(new_path, "wb") as f:
                        f.write(new_image_upload.getbuffer())
                    st.session_state.items_df.at[idx, "image"] = new_path

                save_items()
                st.success("Item updated and saved ✅")
                st.rerun()

        with col_btn2:
            if st.button("🗑️ Delete item", key=f"delete_{selected_id}"):
                st.session_state.items_df = (
                    items_df[items_df["id"] != selected_id]
                    .reset_index(drop=True)
                )
                save_items()
                st.success(f"Item #{selected_id} deleted ❌")
                st.rerun()


# ----------------------------------------------------
# TAB 3: SEARCH ITEMS (NO FEEDBACK HERE)
# ----------------------------------------------------
with tab_search:
    st.subheader("Search items (AutoMatch)")

    items_df = st.session_state.items_df

    if items_df.empty:
        st.info("No items to search. Add items first.")
    else:
        query = st.text_input("Describe what you're looking for:", "")

        if query.strip() == "":
            st.info("Type something to search 😊")
        else:
            corpus = (
                items_df["description"].fillna("")
                + " "
                + items_df["location"].fillna("")
            ).tolist()
            vectorizer = TfidfVectorizer(stop_words="english")
            try:
                X = vectorizer.fit_transform(corpus)
                q_vec = vectorizer.transform([query])
                sims = cosine_similarity(q_vec, X)[0]
            except ValueError:
                sims = np.zeros(len(items_df))

            # use a temp DF so similarity does NOT go into main table
            tmp = items_df.copy()
            tmp["similarity"] = sims
            results = tmp.sort_values("similarity", ascending=False).head(10)

            if results["similarity"].max() == 0:
                st.warning("No strong matches found, but here are some items:")
            else:
                st.success("Here are your best matches:")

            for _, row in results.iterrows():
                with st.container():
                    st.markdown(
                        f"### 🔹 ID {int(row['id'])}: {row['description']}"
                    )
                    st.write(
                        f"**Location:** {row['location']}  |  **Date:** {row['date']}  |  **Contact:** {row['contact']}"
                    )
                    st.write(
                        f"Similarity score: {row['similarity'] * 100:.1f}%"
                    )

                    if (
                        isinstance(row["image"], str)
                        and row["image"] != ""
                        and os.path.exists(row["image"])
                    ):
                        st.image(row["image"], width=200)

                    st.markdown("---")


# ----------------------------------------------------
# TAB 4: FEEDBACK (SEPARATE)
# ----------------------------------------------------
with tab_feedback:
    st.subheader("Give feedback on suggestions")

    items_df = st.session_state.items_df
    feedback_df = st.session_state.feedback_df

    if items_df.empty:
        st.info("No items available yet to give feedback.")
    else:
        with st.form("feedback_form"):
            item_ids = items_df["id"].tolist()
            fb_item_id = st.selectbox("Select item ID", item_ids)
            fb_rating = st.radio("Was this suggestion helpful?", ["Yes", "No"])
            fb_comment = st.text_input(
                "Comment (optional)",
                placeholder="Why was this helpful / not helpful?",
            )
            fb_submit = st.form_submit_button("Submit feedback ✅")

        if fb_submit:
            new_fb = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "item_id": int(fb_item_id),
                "rating": fb_rating,
                "comment": fb_comment.strip(),
            }
            st.session_state.feedback_df = pd.concat(
                [st.session_state.feedback_df, pd.DataFrame([new_fb])],
                ignore_index=True,
            )
            save_feedback()
            st.success("Feedback saved, thank you 💛")

    st.markdown("### 📋 All feedback")
    if st.session_state.feedback_df.empty:
        st.caption("No feedback yet.")
    else:
        st.dataframe(
            st.session_state.feedback_df,
            use_container_width=True,
            hide_index=True,
        )
