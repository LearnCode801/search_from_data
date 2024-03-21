import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
github_link = "[GitHub](https://github.com/LearnCode801)"
linkedin_link = "[LinkedIn](https://www.linkedin.com/in/muhammad-talha-806126234/)"

# st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(hide_st_style, unsafe_allow_html=True)

# Your Streamlit app content here

st.sidebar.markdown(github_link, unsafe_allow_html=True)
st.sidebar.markdown(linkedin_link, unsafe_allow_html=True)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def get_top_related_articles(user_input, df, vectorizer, tfidf_matrix, top_n):

    user_input_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_related_articles = df.iloc[top_indices]

    return top_related_articles

def main():
    st.write('## Title Quest: Discovering Articles through Title Similarity')
    st.write('### Whole Scrape Data') 
    df = pd.read_csv("medium.csv",encoding="latin1")
    df=df[['Title','Content','Headings','Author URL','Read Time','Date','Image URL']]
    df.dropna(subset = ['Title'], inplace=True)
    st.write(df)

    tfidf_matrix = vectorizer.transform(df['Title'])

    user_input = st.text_input("#### Enter your  Title/Topic realated string:")

    if user_input:
        try:
            col1, col2 = st.columns(2)

            with col1:
                top_n = st.number_input("Number of responses:", min_value=1, value=1, step=1)

            with col2:
                selected_columns = st.multiselect("Select columns to display:", df.columns, default=['Title', 'Content'])

            if not selected_columns:
                selected_columns = ['Title', 'Content']

            # Retrieve top related articles
            top_related_articles = get_top_related_articles(user_input, df, vectorizer, tfidf_matrix, top_n)

            st.write(f"### {top_n} - Top Related Articles")

            tab1, tab2 = st.tabs(["## Responce: JSON Format", "## Responce: Table Format"])
            data_dict = top_related_articles[selected_columns].to_dict(orient='records')
                
            tab1.write(data_dict)
            tab2.write(top_related_articles[selected_columns])


        except KeyError as e:
            st.error("Error: " + str(e))


if __name__ == "__main__":
    main()



