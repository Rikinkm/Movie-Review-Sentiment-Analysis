import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np

#title
st.title('Movie Review Sentiment Analysis')
#markdown
st.markdown('This application is all about Movie Review sentiment analysis of different movie. We can analyse reviews of the movie using this streamlit app.')


with st.expander('Upload Your File'):
    data = st.file_uploader('Upload file')

if data:
        df = pd.read_csv(data)
if st.checkbox("Show Data"):
    st.write(df.drop(columns=['sentiment'], axis=1).head(10))




