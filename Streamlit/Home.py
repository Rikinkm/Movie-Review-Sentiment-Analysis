import streamlit as st
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import re
from collections import Counter
import plotly.express as px

# Disable warning message related to Matplotlib's global figure object
st.set_option('deprecation.showPyplotGlobalUse', False)


def predict_sentiment(file_path_model, file_path_tfidf, file):
    # Load trained model and vectorizer
    loaded_model = pkl.load(open(file_path_model, 'rb'))
    loaded_tfidf = pkl.load(open(file_path_tfidf, 'rb'))

    # Load test data
    x_test_2 = pd.read_csv(file)['review']

    # Transform test data using vectorizer
    x_test1 = loaded_tfidf.transform(x_test_2)

    # Make predictions using trained model
    y_pred1 = loaded_model.predict(x_test1)

    # Mapping function to convert labels to sentiment
    def map_sentiment(label):
        if label == 0:
            return 'negative'
        else:
            return 'positive'

    # Apply mapping function to predicted labels
    y_pred_sentiment = [map_sentiment(label) for label in y_pred1]

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({'Review': x_test_2, 'Sentiment': y_pred_sentiment})

    # Add checkbox to show input file
    if st.checkbox('Show input'):
        st.write(x_test_2.head(10))

    # Add checkbox to show sample of 10 rows of data
    if st.checkbox('Show Output'):
        st.write(results_df.head(10))

    # Add button to show pie chart of positive and negative review counts
    if st.button('Show review counts', key='review_count_button'):
        pos_count = len(results_df[results_df['Sentiment'] == 'positive'])
        neg_count = len(results_df[results_df['Sentiment'] == 'negative'])
        fig, ax = plt.subplots(figsize=(9, 2))
        ax.pie([pos_count, neg_count], labels=['Positive', 'Negative'], autopct='%1.1f%%')
        ax.set_title('Review Counts')
        st.pyplot(fig)

    # Define data by copying the results DataFrame
    data = results_df.copy()

    # Add button to show average word count of positive reviews
    if st.button('Show average word count of reviews'):
        # Create a copy of the results DataFrame
        data = results_df.copy()

        # Add a column for the number of words in each review
        data['word_count'] = data['Review'].apply(lambda x: len(re.findall(r'\w+', x)))

        # Calculate the average word count for positive reviews
        pos_avg_word_count = data[data['Sentiment'] == 'positive']['word_count'].mean()

        # Display the result
        st.write('Average word count of positive reviews:', round(pos_avg_word_count, 2))

        # Calculate the average word count for negative reviews
        neg_avg_word_count = data[data['Sentiment'] == 'negative']['word_count'].mean()

        # Display the result
        st.write('Average word count of negative reviews:', round(neg_avg_word_count, 2))

        # Show histogram of word count for positive and negative reviews
        fig, ax = plt.subplots(1,2, figsize=(10, 6))
        ax[0].hist(data[data['Sentiment'] == 'positive']['word_count'], label='Positive', color='blue', rwidth=0.9)
        ax[0].legend(loc='upper right')
        ax[1].hist(data[data['Sentiment'] == 'negative']['word_count'], label='Negative', color='red', rwidth=0.9)
        ax[1].legend(loc='upper right')
        fig.suptitle('Number of words in review')
        st.pyplot(fig)




    # Define data by copying the results DataFrame
    data = results_df.copy()


st.title('Sentiment Analysis App')
file = st.file_uploader("Upload Yous Database", type="csv")
if file is not None:
    predict_sentiment('model_lr.pkl', 'tfidf.pkl', file)
