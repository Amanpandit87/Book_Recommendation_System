import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl

# ✅ Must be first Streamlit command
st.set_page_config(page_title="📚 Book Recommendation", page_icon="📖", layout="wide")

# 🔁 Theme toggle (Dark/Light)
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 📦 Load model and data
Model = pkl.load(open("book_recommendation_model.pkl", "rb"))
pivot_book = pkl.load(open("pivot_book.pkl", "rb"))

@st.cache_data
def load_metadata():
    books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']]
    books.columns = ['ISBN', 'Title', 'Author', 'Year', 'Publisher', 'ImageURL']
    books.drop_duplicates(subset="Title", inplace=True)
    books.set_index("Title", inplace=True)
    return books

book_meta = load_metadata()

# 📊 Top trending books by number of ratings
@st.cache_data
def get_top_trending():
    ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip")
    ratings.columns = ["User-ID", "ISBN", "Book-Rating"]
    merged = ratings.merge(book_meta.reset_index(), on="ISBN")
    popular = merged.groupby("Title").count()["Book-Rating"].sort_values(ascending=False).head(10)
    return popular.index.tolist()

top_books = get_top_trending()

# 🔍 Recommend function
def recommend(book_name):
    try:
        book_id = np.where(pivot_book.index == book_name)[0][0]
        distance, suggestion = Model.kneighbors(pivot_book.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        recommendations = []
        for i in suggestion[0]:
            suggested_title = pivot_book.index[i]
            if suggested_title != book_name:
                meta = book_meta.loc[suggested_title] if suggested_title in book_meta.index else None
                if isinstance(meta, pd.DataFrame):
                    meta = meta.iloc[0]
                recommendations.append({
                    "title": suggested_title,
                    "author": meta["Author"] if meta is not None else "N/A",
                    "publisher": meta["Publisher"] if meta is not None else "N/A",
                    "year": meta["Year"] if meta is not None else "N/A",
                    "image": meta["ImageURL"] if meta is not None else None
                })
        return recommendations

    except Exception as e:
        return [{"title": f"Error: {e}"}]

# 🎯 Title
st.title("📚 Book Recommendation System")

# 🔝 Top trending
with st.expander("📈 Show Top 10 Trending Books"):
    st.markdown("Here are the most rated books:")
    for title in top_books:
        st.markdown(f"➡️ {title}")

# 🔍 Search by keyword
search_keyword = st.text_input("🔍 Search Book Title (Type & Select):")

book_matches = [title for title in pivot_book.index if search_keyword.lower() in title.lower()] if search_keyword else []
selected_book = st.selectbox("Choose a Book", book_matches if book_matches else pivot_book.index.tolist())

# 🔘 Button & output
if st.button("✨ Get Recommendations"):
    st.subheader("Recommendations for: " + selected_book)
    recs = recommend(selected_book)

    for rec in recs:
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                if rec["image"]:
                    st.image(rec["image"], width=100)
                else:
                    st.write("📕 No Image")
            with cols[1]:
                st.markdown(f"**📖 Title:** {rec['title']}")
                st.markdown(f"**✍️ Author:** {rec['author']}")
                st.markdown(f"**🏢 Publisher:** {rec['publisher']}")
                st.markdown(f"**📅 Year:** {rec['year']}")
            st.markdown("---")
