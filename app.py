from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import logging

logging.basicConfig(filename='app_file_log.log')
# logging.debug('This message should go to the log file')
tweet_input =  st.container()
sentiment_analysis =  st.container()


data = pd.read_csv("src/twitter.csv")
#print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# svm = SVC(gamma=2)
# svm.fit(X_train,y_train)
# svm.score(X_test,y_test)


# def hate_speech_detection():
#
#     st.title("Cyberbullying  Detection")
#     user_text = st.text_area("Enter any Tweet: ")
#     if len(user_text) < 1:
#         st.write("  ")
#     else:
#         sample = user_text
#         data = cv.transform([sample]).toarray()
#         a = clf.predict(data)
#         st.title(a)
# hate_speech_detection()

with tweet_input:
    st.header('Is Your Text Considered Cyberbullying?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_area("Enter any Tweet: ")
    st.button("Predict", key=str)
    if len(user_text) < 1:
        st.write("  ")
    else:
        sample = user_text
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')

        # explaining VADER
        st.write(
            """*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')

        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        if sentiment_dict['compound'] >= 0.05:
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05:
            category = ("**Negative 🚫**")
        else:
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category)
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg'] * 100, "% Negative")
            st.write(sentiment_dict['neu'] * 100, "% Neutral")
            st.write(sentiment_dict['pos'] * 100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)
            logging.debug('This message should go to the log file')





# streamlit run app.py
# streamlit run streamlit_app.py --logger.level=debug 2>logs.txt

