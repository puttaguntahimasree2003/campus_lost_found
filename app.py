# app.py
# Campus Lost & Found – Persistent Version
# - items saved in items.csv (never reset)
# - images stored in /images folder
# - add / edit / delete items
# - feedback stored in feedback.csv
# - text AutoMatch using TF-IDF
# - Search tab has feedback + controls
# - Feedback tab only shows feedback table

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
        df = pd.DataFrame(columns=["time", "item_id", "rating", "comment", "query"])

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
    ["➕ Add Item", "📝 View / Edit / Delete", "🔍 Search & Feedback", "📋 All Feedback"]
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
# TAB 3: SEARCH + FEEDBACK (TOGETHER)
# ----------------------------------------------------
with tab_search:
    st.subheader("Search items and give feedback")

    items_df = st.session_state.items_df

    if items_df.empty:
        st.info("No items to search. Add items first.")
    else:
        # search controls
        query = st.text_input(
            "Describe what you're looking for (description keywords)*", ""
        )
        location_filter = st.text_input(
            "Location (optional)", placeholder="e.g. Girls Hostel"
        )

        max_results = min(20, len(items_df))
        top_k = st.slider(
            "How many suggestions do you want to see?",
            min_value=1,
            max_value=max_results,
            value=min(5, max_results),
        )

        if query.strip() == "":
            st.info("Type a description to search 😊")
        else:
            # optional location filter BEFORE similarity (so TF-IDF only on filtered set)
            df_search = items_df.copy()
            if location_filter.strip():
                mask = df_search["location"].str.contains(
                    location_filter.strip(), case=False, na=False
                )
                df_search = df_search[mask]

            if df_search.empty:
                st.warning("No items match this location filter.")
            else:
                corpus = (
                    df_search["description"].fillna("")
                    + " "
                    + df_search["location"].fillna("")
                ).tolist()
                vectorizer = TfidfVectorizer(stop_words="english")
                try:
                    X = vectorizer.fit_transform(corpus)
                    q_vec = vectorizer.transform([query])
                    sims = cosine_similarity(q_vec, X)[0]
                except ValueError:
                    sims = np.zeros(len(df_search))

                tmp = df_search.copy()
                tmp["similarity"] = sims
                results = (
                    tmp.sort_values("similarity", ascending=False)
                    .head(top_k)
                    .reset_index(drop=True)
                )

                if results["similarity"].max() == 0:
                    st.warning("No strong matches found, but here are some items:")
                else:
                    st.success("Here are your best matches:")

                for i, row in results.iterrows():
                    item_id = int(row["id"])
                    with st.container():
                        st.markdown(
                            f"### 🔹 ID {item_id}: {row['description']}"
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

                        # ---- Feedback section for this item (in Search tab) ----
                        st.markdown("**Feedback on this suggestion:**")
                        fb_col1, fb_col2 = st.columns([1, 3])
                        with fb_col1:
                            rating = st.radio(
                                f"Helpful? (ID {item_id})",
                                ["Yes", "No"],
                                key=f"fb_rating_{item_id}",
                            )
                        with fb_col2:
                            comment = st.text_input(
                                "Comment (optional)",
                                key=f"fb_comment_{item_id}",
                                placeholder="Why was this helpful / not helpful?",
                            )

                        if st.button(
                            "Submit feedback", key=f"fb_btn_{item_id}"
                        ):
                            new_fb = {
                                "time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "item_id": item_id,
                                "rating": rating,
                                "comment": comment.strip(),
                                "query": query.strip(),
                            }
                            st.session_state.feedback_df = pd.concat(
                                [
                                    st.session_state.feedback_df,
                                    pd.DataFrame([new_fb]),
                                ],
                                ignore_index=True,
                            )
                            save_feedback()
                            st.success("Feedback saved, thank you 💛")

                        st.markdown("---")


# ----------------------------------------------------
# TAB 4: FEEDBACK TABLE ONLY
# ----------------------------------------------------
with tab_feedback:
    st.subheader("All feedback given")

    feedback_df = st.session_state.feedback_df

    if feedback_df.empty:
        st.caption("No feedback yet.")
    else:
        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True,
        )

