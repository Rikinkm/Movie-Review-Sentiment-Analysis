import streamlit as st
import pickle

# Load the LR model and TF-IDF vectorizer from disk
filename_model = 'model_lr.pkl'
filename_tfidf = 'tfidf.pkl'
model = pickle.load(open(filename_model, 'rb'))
tfidf = pickle.load(open(filename_tfidf, 'rb'))

# Define Streamlit app
def app():
    # Title of the app
    st.title('Movie Review Sentiment Analysis')

    # Text input for user to enter review
    user_input = st.text_input("Enter your movie review here:")

    # Button to classify the sentiment of the entered review
    if st.button('Classify'):
        user_input_tfidf = tfidf.transform([user_input]).toarray()
        prediction = model.predict(user_input_tfidf)

        # Show the predicted sentiment
        if prediction[0] == 0:
            st.write('The predicted sentiment is: Negative')
        else:
            st.write('The predicted sentiment is: Positive')




# Run the app
if __name__ == '__main__':
    app()
