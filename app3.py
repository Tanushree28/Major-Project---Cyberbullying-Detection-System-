# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
# creating page sections
site_header =  st.container()
business_context =  st.container()
data_desc =  st.container()
performance =  st.container()
tweet_input =  st.container()
model_results =  st.container()
sentiment_analysis =  st.container()
contact =  st.container()


def main():
    """Bully Detector"""
    st.title("Cyberbullying Detection System")
    # -----------------------------------------------Side Menu Bar ---------------------------------------
    menu = ["Home", "Bully Detector", "Report"]
    choice = st.sidebar.selectbox("Menu", menu)

    # -----------------------------------------------Home Section ---------------------------------------
    if choice == "Home":

        st.header('The Problem of Cyberbullying')
        st.info("""
        ***“Think before you click. If people do not know you personally and if they cannot see you as you type, what you post online can be taken out of context if you are not careful in the way your message is delivered.”*** - Germany Kent
                """)
        images = ['image/1.png', 'image/2.png', 'image/3.png', 'image/4.png', 'image/5.png']

        st.image(images, use_column_width=True)

        st.write("""
        In 2018, an [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5914259/) on The Indian Journal of Psychiatry written by Smith et al. defined Cyberbullying as an **“aggressive, intentional act carried out by a group or individual, using electronic forms of contact, repeatedly and over time against a victim who cannot easily defend himself or herself.”**
                         """)

        st.write("""
        ***Hate Speech*** is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.** 

        ***Foul or Offensive Language*** means **words, whether intended or not, that offend, intimidate, or otherwise cause emotional or psychological harm to the recipient and/or staff and includes content that incites hatred against, promotes discrimination of, or disparages an individual or group on the basis of their race, ethnic origin, religion (or lack thereof), disability, age, nationality, veteran status, sexual orientation, gender, gender identity, or other characteristic that is associated with systematic discrimination or marginalisation.**

        Usually, the difference between hate speech and offensive language comes down to subtle context or diction.
        """)

        st.image(Image.open('visualizations/word_venn.png'), width=400)

    # ---------------------------------------Bully Detector Section------------------------------------

    elif choice == "Bully Detector":
        data = pd.read_csv("src/twitter.csv")
        # print(data.head())

        data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
        # print(data.head())

        data = data[["tweet", "labels"]]
        # print(data.head())

        import re
        import nltk
        stemmer = nltk.SnowballStemmer("english")
        from nltk.corpus import stopwords
        import string
        stopword = set(stopwords.words('english'))

        def clean(text):
            text = str(text).lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = [word for word in text.split(' ') if word not in stopword]
            text = " ".join(text)
            text = [stemmer.stem(word) for word in text.split(' ')]
            text = " ".join(text)
            return text

        data["tweet"] = data["tweet"].apply(clean)
        # print(data.head())

        x = np.array(data["tweet"])
        y = np.array(data["labels"])

        cv = CountVectorizer()
        X = cv.fit_transform(x)  # Fit the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

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
            st.write(
                """*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
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

    # ----------------------------------------Email Section ------------------------------------------
    elif choice == "Report":
        st.subheader("Report Bullying")
        st.header(":mailbox: Get In Touch With Us!  ")

        contact_form = """
        <form action="https://formsubmit.co/tanu.nepal1@gmail.com" method="POST" enctype="multipart/form-data">
            <input type="hidden" name="_template" value="table">
            <input type="hidden" name="_captcha" value="false">
            <input type="hidden" name="_subject" value="Reporting Cyberbullying">
            <input type="text" name="Name" placeholder="Your name" required>
            <input type="email" name="Email" placeholder="Your Email" required>
            <textarea name="message" placeholder="Details of your problem"></textarea>
            <input type="file" name="Attachment" accept="image/png, image/jpeg, image/jpg">
            <button type="submit">Send</button>
        </form>
         """
        st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")


if __name__ == '__main__':
    main()

# # streamlit run app3.py