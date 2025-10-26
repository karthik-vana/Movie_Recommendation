import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommendation — Feature Text → Tokens")

DATA_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/refs/heads/main/Movies%20Recommendation.csv"

@st.cache_data(show_spinner=False)
def load_data(url):
    df = pd.read_csv(url)
    # fill NaNs for feature columns and create a combined text feature
    features = ['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']
    for f in features:
        if f not in df.columns:
            df[f] = ""
    df_features = df[features].fillna('')
    combined = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
    df = df.reset_index(drop=True)
    return df, combined

@st.cache_data(show_spinner=False)
def build_tfidf_and_similarity(combined_series):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(combined_series)
    sim = cosine_similarity(X)
    return tfidf, X, sim

df, combined = load_data(DATA_URL)
tfidf, X, similarity = build_tfidf_and_similarity(combined)

st.sidebar.header("Find recommendations")
movie_input = st.sidebar.text_input("Enter your favourite movie name")
top_option = st.sidebar.selectbox("Which list to show?", ["Top 10", "Top 30", "Both (Top 10 & Top 30)"])
reco_button = st.sidebar.button("Recommend")

st.markdown("### Dataset quick info")
st.write(f"Total movies loaded: {len(df)}")

def recommend_titles(movie_name, top_n=10):
    titles = df['Movie_Title'].astype(str).tolist()
    # get close matches; handle case-insensitively by using original titles list
    matches = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.0)
    if not matches:
        return None, f"No close match found for '{movie_name}'. Check spelling."
    close_match = matches[0]
    idx = df[df['Movie_Title'] == close_match].index
    if len(idx) == 0:
        return None, f"Matched title '{close_match}' not found in DataFrame index."
    idx = idx[0]
    sim_scores = list(enumerate(similarity[idx]))
    # sort by score (descending) and exclude the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in sim_scores:
        if i == idx:
            continue
        title = df.loc[i, 'Movie_Title']
        recommendations.append((title, float(score)))
        if len(recommendations) >= top_n:
            break
    return recommendations, None

if reco_button:
    if not movie_input:
        st.sidebar.error("Please type a movie name in the text box first.")
    else:
        with st.spinner("Finding recommendations..."):
            rec10, err10 = recommend_titles(movie_input, top_n=10)
            rec30, err30 = recommend_titles(movie_input, top_n=30)
        if err10 or err30:
            st.error(err10 or err30)
        else:
            show_both = (top_option == "Both (Top 10 & Top 30)")
            if top_option == "Top 10" or show_both:
                st.subheader("Top 10 Movies Suggested for you")
                for i, (title, score) in enumerate(rec10, start=1):
                    st.write(f"{i}. {title}  —  similarity: {score:.4f}")
            if top_option == "Top 30" or show_both:
                st.subheader("Top 30 Movies Suggested for you")
                for i, (title, score) in enumerate(rec30, start=1):
                    st.write(f"{i}. {title}  —  similarity: {score:.4f}")

# Helpful footer
st.markdown("---")
st.caption("Notes: This app uses TF-IDF on combined text features (genre, keywords, tagline, cast, director). "
           "If the dataset schema differs, the app will still try to proceed by filling missing columns.")