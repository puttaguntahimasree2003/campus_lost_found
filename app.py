import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Campus Lost & Found", layout="wide")

ITEMS_FILE = "items.csv"
FEEDBACK_FILE = "feedback.csv"
IMAGE_DIR = "images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# HELPERS TO LOAD & SAVE DATA (PERSISTENT)

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
    """Load feedback.csv into session_state.feedback_df with columns:
       item_id, helpful, comment, time
    """
    if "feedback_df" in st.session_state:
        return

    required_cols = ["item_id", "helpful", "comment", "time"]

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
    else:
        df = pd.DataFrame(columns=required_cols)

    # ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # keep only required columns in correct order
    df = df[required_cols]

    st.session_state.feedback_df = df
    st.session_state.feedback_df.to_csv(FEEDBACK_FILE, index=False)


def save_items():
    st.session_state.items_df.to_csv(ITEMS_FILE, index=False)


def save_feedback():
    """Always save only the 4 feedback columns in correct order."""
    required_cols = ["item_id", "helpful", "comment", "time"]
    df = st.session_state.feedback_df
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    st.session_state.feedback_df = df[required_cols]
    st.session_state.feedback_df.to_csv(FEEDBACK_FILE, index=False)


# ---------------- IMAGE FEATURE HELPERS ----------------
def preprocess_image(img: Image.Image, size=(128, 128)) -> np.ndarray:
    """Convert image to a small grayscale, flattened & normalized vector."""
    img = img.convert("L").resize(size)  # grayscale
    arr = np.array(img, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def compute_image_similarity(query_vec: np.ndarray, image_path: str) -> float:
    """Cosine similarity between query image vector and stored image."""
    if (
        not isinstance(image_path, str)
        or image_path.strip() == ""
        or not os.path.exists(image_path)
    ):
        return 0.0
    try:
        img = Image.open(image_path)
        vec = preprocess_image(img)
        if np.linalg.norm(vec) == 0 or np.linalg.norm(query_vec) == 0:
            return 0.0
        return float(np.dot(query_vec, vec))  # cosine because both normalized
    except Exception:
        return 0.0


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
st.title("🏫 Campus Lost & Found with AutoMatch")


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_add, tab_manage, tab_search, tab_feedback = st.tabs(
    ["➕ Add Item", "📝 View / Edit / Delete", "🔍 Search Item", "📋 All Feedback"]
)


# ----------------------------------------------------
# TAB 1: ADD ITEM
# ----------------------------------------------------
with tab_add:
    st.subheader("Add a item you found")

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
    st.subheader("All items")

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
# TAB 3: SEARCH + FEEDBACK (TEXT + IMAGE SIMILARITY)
# ----------------------------------------------------
with tab_search:
    st.subheader("Search items and give feedback")

    items_df = st.session_state.items_df

    if items_df.empty:
        st.info("No items to search. Add items first.")
    else:
        # search controls
        query = st.text_input(
                "Describe what you're looking for*", placeholder="Red water bottle with scratches..."
            )
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

        # checkbox to show/hide images in search results
        show_images = st.checkbox("Show item images", value=False)

        # NEW: upload query image to get image similarity
        query_image_file = st.file_uploader(
            "Upload image to match (optional for image similarity)",
            type=["png", "jpg", "jpeg"],
            key="search_query_image",
        )

        query_img_vec = None
        if query_image_file is not None:
            try:
                q_img = Image.open(query_image_file)
                query_img_vec = preprocess_image(q_img)
            except Exception:
                st.warning("Could not read the uploaded image for similarity.")

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
                df_search = df_search.reset_index(drop=True)

                # ---------- TEXT SIMILARITY ----------
                corpus = (
                    df_search["description"].fillna("")
                    + " "
                    + df_search["location"].fillna("")
                ).tolist()
                vectorizer = TfidfVectorizer(stop_words="english")
                try:
                    X = vectorizer.fit_transform(corpus)
                    q_vec = vectorizer.transform([query])
                    text_sims = cosine_similarity(q_vec, X)[0]
                except ValueError:
                    text_sims = np.zeros(len(df_search))

                # ---------- IMAGE SIMILARITY ----------
                img_sims = np.zeros(len(df_search), dtype=float)
                if query_img_vec is not None:
                    for i_row, img_path in enumerate(df_search["image"].tolist()):
                        img_sims[i_row] = compute_image_similarity(
                            query_img_vec, img_path
                        )

                # ---------- COMBINED SCORE ----------
                tmp = df_search.copy()
                tmp["text_sim"] = text_sims
                tmp["img_sim"] = img_sims

                if query_img_vec is not None:
                    # average of text + image similarities
                    tmp["score"] = (tmp["text_sim"] + tmp["img_sim"]) / 2.0
                else:
                    tmp["score"] = tmp["text_sim"]

                results = (
                    tmp.sort_values("score", ascending=False)
                    .head(top_k)
                    .reset_index(drop=True)
                )

                if results["score"].max() == 0:
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
                        # show both similarities
                        st.write(
                            f"Text similarity: {row['text_sim'] * 100:.1f}%"
                            f"  |  Image similarity: {row['img_sim'] * 100:.1f}%"
                        )

                        # show image only if checkbox is ticked
                        if show_images:
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
                                "item_id": item_id,
                                "helpful": rating,  # Yes / No
                                "comment": comment.strip(),
                                "time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
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


# TAB 4: FEEDBACK TABLE ONLY

with tab_feedback:
    st.subheader("All feedback given")

    feedback_df = st.session_state.feedback_df

    if feedback_df.empty:
        st.caption("No feedback yet.")
    else:
        # show only item_id, helpful, comment, time in order
        cols = ["item_id", "helpful", "comment", "time"]
        existing = [c for c in cols if c in feedback_df.columns]
        st.dataframe(
            feedback_df[existing],
            use_container_width=True,
            hide_index=True,
        )




