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
    # st.title("Cyberbullying Detection System")
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
        with business_context:
            st.header('The Problem of Content Moderation')
            st.write("""

                        **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.

                        Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.

                        """)

        with data_desc:
            understanding, venn = st.columns(2)
            with understanding:
                st.text('')
                st.write("""
                            The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.

                            The `.csv` file has **24,802 rows** where **6% of the tweets were labeled as "Hate Speech".**

                            Each tweet's label was voted on by crowdsource and determined by majority rules.
                            """)
            with venn:
                st.image(Image.open('visualizations/word_venn.png'), width=400)

        with performance:
            description, conf_matrix = st.columns(2)
            with description:
                st.header('Final Model Performance')
                st.write("""
                            These scores are indicative of the two major roadblocks of the project:
                            - The massive class imbalance of the dataset
                            - The model's inability to identify what constitutes as hate speech
                            """)
            with conf_matrix:
                st.image(Image.open('visualizations/normalized_log_reg_countvec_matrix.png'), width=400)

        with tweet_input:
            st.header('Is Your Tweet Considered Hate Speech?')
            st.write(
                """*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
            # user input here
            user_text = st.text_input('Enter Tweet')  # setting input as user_text
            st.button("Predict", key=str)

        with model_results:
            st.subheader('Prediction:')
            if user_text:
                # processing user_text
                # removing punctuation
                user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
                # tokenizing
                stop_words = set(stopwords.words('english'))
                tokens = nltk.word_tokenize(user_text)
                # removing stop words
                stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
                # taking root word
                lemmatizer = WordNetLemmatizer()
                lemmatized_output = []
                for word in stopwords_removed:
                    lemmatized_output.append(lemmatizer.lemmatize(word))

                # instantiating count vectorizor
                count = CountVectorizer(stop_words=stop_words)
                X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
                X_test = lemmatized_output
                X_train_count = count.fit_transform(X_train)
                X_test_count = count.transform(X_test)

                # loading in model
                final_model = pickle.load(open('pickle/final_SVM_count_model.pkl', 'rb'))

                # apply model to make predictions
                prediction = final_model.predict(X_test_count[0])
                st.title(prediction)

                if prediction == 0:
                    st.subheader('**Hate Speech**')
                if prediction == 1:
                    st.subheader('**Offensive Language**')
                elif prediction == 2:
                    st.subheader('**Neither Hate Speech Nor Offensive Language Maybe Gibberish too**')
                st.text('')

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
                    # logging.debug('This message should go to the log file')

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

# # streamlit run app2.py