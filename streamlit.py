import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open('C:/Users/PMLS/Pictures/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def get_top_related_articles(user_input, df, vectorizer, tfidf_matrix, top_n):
    # Transform user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Compute cosine similarity between user input and article titles
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()

    # Get indices of top related articles
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Get top related articles
    top_related_articles = df.iloc[top_indices]

    return top_related_articles

def main():
    st.write('### Article Search Based on the "Title Similarity Search"')

    # File upload

    
    df = pd.read_csv("D:/6thSemester/Data Warehousing/Assignemts/Api python/medium.csv",encoding="latin1")
    df=df[['Title','Content','Headings','Author URL','Read Time','Date','Image URL']]
    df.dropna(subset = ['Title'], inplace=True)
    st.write(df)

    tfidf_matrix = vectorizer.transform(df['Title'])

    user_input = st.text_input("Enter your your topic realated string:")

    if user_input:
        try:
            top_n=3
            top_related_articles = get_top_related_articles(user_input, df, vectorizer, tfidf_matrix,top_n)
            st.write(f"### {top_n} - Top Related Articles")
            st.write(top_related_articles[['Title', 'Content', 'Author URL', 'Date', 'Image URL']])
        except KeyError as e:
            st.error("Error: " + str(e))

if __name__ == "__main__":
    main()















# import streamlit as st
# import pandas as pd
# import pickle
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity



# def processing1():
#     with open('C:/Users/PMLS/Pictures/tfidf_vectorizer.pkl', 'rb') as file:
#         tfidf_matrix = pickle.load(file)

#     return tfidf_matrix

# def get_top_related_articles(user_input, df, tfidf_matrix, vectorizer, top_n=1):
#     # Transform user input
#     user_input_tfidf = vectorizer.transform([user_input])

#     # Compute cosine similarity between user input and article titles
#     cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()

#     # Get indices of top related articles
#     top_indices = cosine_similarities.argsort()[-top_n:][::-1]

#     # Get top related articles
#     top_related_articles = df.iloc[top_indices]

#     return top_related_articles

# def main():
#     st.title('CSV File Viewer')

#     # File upload
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         # Read CSV file
#         df = pd.read_csv(uploaded_file)
#         df=df[['Title','Content','Headings','Author URL','Read Time','Date','Image URL']]
#         df.dropna(subset = ['Title'], inplace=True)
#         # Display DataFrame
#         st.write("### DataFrame")
#         st.write(df.head())
#         st.write("===================================")
#         user_input = input("Enter your query: ")
#         try:
#             tfidf_matrix=processing1()
#             top_related_articles = get_top_related_articles(user_input, df, tfidf_matrix, vectorizer)
#             print(top_related_articles[['Title', 'Content', 'Author URL', 'Date','Image URL']])
#         except KeyError as e:
#             print("Error: ", e)


# if __name__ == "__main__":
#     main()