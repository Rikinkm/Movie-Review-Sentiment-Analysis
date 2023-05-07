import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from tqdm.auto import tqdm
import time



# Load the dataset
data = pd.read_csv('Mountain Analytics IMDB Dataset 1.csv')


data.head()

data.tail()

data.isna().any()#Detect missing values,gives Boolean value True for NA (not-a -number) values, and otherwise False.

data.isna().sum()#returns the number of missing values in each column.

data['review'].nunique()

data.shape

data.info()

data['review'].drop_duplicates(inplace = True)

data['review'].nunique()

data.shape

data

# Create a bar plot of the class distribution
class_counts = data['sentiment'].value_counts()
class_counts.plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiments')
plt.ylabel('Number of Reviews')
plt.show()

for i in range(5):#printing first 5 reviews
    print('Review: ',[i])
    print(data['review'].iloc[i],"\n")
    print('Sentiment: ',data['sentiment'].iloc[i],'\n\n')


from collections import Counter
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def no_of_words(text):#takes the text,split it into words and returns the count of words
    words=text.split()
    word_count=len(words)
    return word_count

data['word_count']=data['review'].apply(no_of_words)

data.head()

fig,ax=plt.subplots(1,2,figsize=(10,6))
ax[0].hist(data[data['sentiment']=='positive']['word_count'],label='Positive',color='blue',rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(data[data['sentiment']=='negative']['word_count'],label='Negative',color='red',rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle('Number of words in review')
plt.show()

fig,ax=plt.subplots(1,2,figsize=(10,6))
ax[0].hist(data[data['sentiment']=='positive']['review'].str.len(),label='Positive',color='blue',rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(data[data['sentiment']=='negative']['review'].str.len(),label='Negative',color='red',rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle('Length of reviews')
plt.show()

#converting target column into numerical format
data.sentiment.replace("positive",1,inplace=True)
data.sentiment.replace("negative",0,inplace=True)

data.head()


from nltk.tokenize import sent_tokenize, word_tokenize
def data_processing(text):
    text=text.lower()
    text=re.sub('<br />','',text)
    text=re.sub(r"https\S+|www\S+|http\S+",'',text,flags=re.MULTILINE)
    text=re.sub(r'\@w+|\#','',text)
    text=re.sub(r'[^\w\s]','',text)
    text_tokens=word_tokenize(text)
    filtered_text=[w for w in text_tokens if not w in stopwords]
    return " ".join(filtered_text)

data['review']=data['review'].apply(data_processing)

data.head()

duplicated_count=data.duplicated().sum()
print('Number of duplicate entries: ',duplicated_count)

#Removing duplicate entries
data=data.drop_duplicates('review')

#stemming
stemmer=PorterStemmer()
def stemming(data):
    text=[stemmer.stem(word) for word in data]
    return data

data.review=data['review'].apply(lambda x: stemming(x))

data['word_count']=data['review'].apply(no_of_words)

data.head()

for i in range(5):#printing first 5 pre-processed reviews
    print('Review: ',[i])
    print(data['review'].iloc[i],"\n")
    print('Sentiment: ',data['sentiment'].iloc[i],'\n\n')

pos_reviews=data[data.sentiment==1]
pos_reviews.head()

from collections import Counter
count=Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word]+=1
count.most_common(20)

pos_words=pd.DataFrame(count.most_common(20))
pos_words.columns=['Word','Count']
pos_words.head()

import plotly.express as px
px.bar(pos_words,x='Count',y='Word',title='Common words in positive reviews',color='Word')

neg_reviews=data[data.sentiment==0]
neg_reviews.head()

from collections import Counter
count=Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word]+=1
count.most_common(20)


neg_words=pd.DataFrame(count.most_common(20))
neg_words.columns=['Word','Count']
neg_words.head()


px.bar(neg_words,x='Count',y='Word',title='Common words in negative reviews',color='Word')


%%time 
#progress indicator

tqdm.pandas()

X=data['review']
Y=data['sentiment']

X

x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(X,Y,test_size=0.2,random_state=37)

x_test_1

Y

X.info()

vect=TfidfVectorizer()
X_tv=vect.fit_transform(X)

X_tv

x_train_tv,x_test_tv,y_train_tv,y_test_tv=train_test_split(X_tv,Y,test_size=0.2,random_state=37)

x_train_tv

pd.DataFrame.sparse.from_spmatrix(x_train_tv,columns=vect.get_feature_names())

pd.DataFrame.sparse.from_spmatrix(x_test_tv,columns=vect.get_feature_names())

print("Size of x_train:  ", (x_train_tv.shape))
print("Size of y_train:  ", (y_train_tv.shape))
print("Size of x_test:  ", (x_test_tv.shape))
print("Size of y_test:  ", (y_test_tv.shape))


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# train a Logistic Regression Model
logreg= LogisticRegression(max_iter = 1000)

logreg.fit(x_train_tv,y_train_tv)

# evaluate the classifier on the test set
y_pred_tv = logreg.predict(x_test_tv)

y_pred_tv

y_test_tv

acc_tv = accuracy_score(y_test_tv, y_pred_tv)
print("Accuracy:", acc_tv)

cm_tv = confusion_matrix(y_test_tv, y_pred_tv)
sns.heatmap(cm_tv, annot=True)

print("\n")
print(classification_report(y_test_tv,y_pred_tv))

from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
param_grid={"C": loguniform(1e-3,1e3),"class_weight":['balanced',None]}
clf=LogisticRegression()

random_search=RandomizedSearchCV(
    clf,
    param_grid,
    n_iter=25,
    verbose=1,
    n_jobs=-1,
    random_state=123,
    return_train_score=True,
    error_score='raise'
)
random_result=random_search.fit(x_train_tv,y_train_tv)

print(random_result.best_score_, random_result.best_params_)

# train a Multinomial Naive Baye's Model
mulNB= MultinomialNB()

mulNB.fit(x_train_tv,y_train_tv)

# evaluate the classifier on the test set
y_pred_tv = mulNB.predict(x_test_tv)

y_pred_tv

y_test_tv

acc_tv = accuracy_score(y_test_tv, y_pred_tv)
print("Accuracy:", acc_tv)

cm_tv = confusion_matrix(y_test_tv, y_pred_tv)
sns.heatmap(cm_tv, annot=True)

print("\n")
print(classification_report(y_test_tv,y_pred_tv))

from sklearn.ensemble import RandomForestClassifier

# train a Multinomial Naive Baye's Model
rand=RandomForestClassifier()

rand.fit(x_train_tv,y_train_tv)

# evaluate the classifier on the test set
y_pred_tv = rand.predict(x_test_tv)

y_pred_tv

y_test_tv

acc_tv = accuracy_score(y_test_tv, y_pred_tv)
print("Accuracy:", acc_tv)

cm_tv = confusion_matrix(y_test_tv, y_pred_tv)
sns.heatmap(cm_tv, annot=True)

countV=CountVectorizer()
X_cv=countV.fit_transform(X)

X_cv

x_train_cv,x_test_cv,y_train_cv,y_test_cv=train_test_split(X_cv,Y,test_size=0.2,random_state=37)

x_train_cv

pd.DataFrame.sparse.from_spmatrix(x_train_cv,columns=countV.get_feature_names())

pd.DataFrame.sparse.from_spmatrix(x_test_cv,columns=countV.get_feature_names())

print("Size of x_train:  ", (x_train_cv.shape))
print("Size of y_train:  ", (y_train_cv.shape))
print("Size of x_test:  ", (x_test_cv.shape))
print("Size of y_test:  ", (y_test_cv.shape))

logreg.fit(x_train_cv,y_train_cv)

# evaluate the classifier on the test set
y_pred_cv = logreg.predict(x_test_cv)

y_pred_cv

y_test_cv

acc_cv = accuracy_score(y_test_cv, y_pred_cv)
print("Accuracy:", acc_cv)

cm_cv = confusion_matrix(y_test_cv, y_pred_cv)
sns.heatmap(cm_cv, annot=True)

print("\n")
print(classification_report(y_test_cv,y_pred_cv))

random_search=RandomizedSearchCV(
    clf,
    param_grid,
    n_iter=25,
    verbose=1,
    n_jobs=-1,
    random_state=123,
    return_train_score=True,
    error_score='raise'
)
random_resultcv=random_search.fit(x_train_cv,y_train_cv)


print(random_resultcv.best_score_, random_resultcv.best_params_)

# train a Multinomial Naive Baye's Model
multiNB= MultinomialNB()

multiNB.fit(x_train_cv,y_train_cv)

# evaluate the classifier on the test set
y_pred_cv = multiNB.predict(x_test_cv)

y_pred_cv

y_test_cv

acc_cv = accuracy_score(y_test_cv, y_pred_cv)
print("Accuracy:", acc_cv)

cm_cv = confusion_matrix(y_test_cv, y_pred_cv)
sns.heatmap(cm_cv, annot=True)

print("\n")
print(classification_report(y_test_cv,y_pred_cv))

# train a Multinomial Naive Baye's Model
randcv=RandomForestClassifier()

randcv.fit(x_train_cv,y_train_cv)

# evaluate the classifier on the test set
y_pred_cv = randcv.predict(x_test_cv)

y_test_cv

acc_cv = accuracy_score(y_test_cv, y_pred_cv)
print("Accuracy:", acc_cv)

import pickle as pkl

file_path_model='model_lr.pkl'

file_path_tfidf = 'tfidf.pkl'

pkl.dump(logreg,open(file_path_model,'wb'))

pkl.dump(vect,open(file_path_tfidf,'wb'))

loaded_model=pkl.load(open('model_lr.pkl','rb'))
loaded_tfidf=pkl.load(open('tfidf.pkl','rb'))

x_test_1

x_test_2=pd.read_csv('sample37.csv')

x_test_2.head()

x_test_2.tail()

x_test_2.info()

x_test37= loaded_tfidf.transform(x_test_2)

y_pred37= loaded_model.predict(x_test37)

y_pred37.size

x_test= loaded_tfidf.transform(x_test_1)

y_pred= loaded_model.predict(x_test)

y_pred

y_pred.size

x_test_3=pd.read_csv('sample37.csv')

x_test_3['review']=x_test_3['review'].apply(data_processing)

x_test86= loaded_tfidf.transform(x_test_3)

y_pred86= loaded_model.predict(x_test86)

y_pred86

y_pred86.size



