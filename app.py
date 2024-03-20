from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open('C:/Users/PMLS/Pictures/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)



df = pd.read_csv("D:/6thSemester/Data Warehousing/Assignemts/Api python/medium.csv",encoding="latin1")
df=df[['Title','Content','Headings','Author URL','Read Time','Date','Image URL']]
df.dropna(subset = ['Title'], inplace=True)


tfidf_matrix = vectorizer.transform(df['Title'])

def get_top_related_articles(user_input, df, vectorizer, tfidf_matrix, top_n=1):
    user_input_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_related_articles = df.iloc[top_indices]

    return top_related_articles.to_dict(orient='records')


@app.route('/')
def wellcome():
    return "Wellcome to the Flask Api of Data Scraping"


@app.route('/get_related_articles', methods=['POST'])
def get_related_articles():
    try:
        user_input = request.json['query']
        top_related_articles = get_top_related_articles(user_input, df, vectorizer, tfidf_matrix)
        return jsonify(top_related_articles)
    except:
        return "No Data Existed"



if __name__ == '__main__':
    app.run(debug=True)


