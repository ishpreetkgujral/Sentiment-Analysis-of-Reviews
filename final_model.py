# necessary imports
import nltk
import re
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing
import random
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
stemmer=SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from googletrans import Translator
translator = Translator(service_urls=['translate.google.co.in'])



# dataset
raw_dataset=pd.read_csv("/content/Sentiment_Analysis_Test_Dataset1.csv",encoding = "ISO-8859-1")

# print(raw_dataset)
df=raw_dataset

review_column=df['Reviews']

length=len(review_column)-2
# print(len(review_column))

# helper function for translation of text from hindi to english
def translation(text):
    # time.sleep(1000)
    return translator.translate(text, src='hi', dest='en').text

# function to detect the language of the text and if hindi then translate it into english
def detect_language(text):
    updated_review=[]
    for i in range(1,len(text)):
        time.sleep(10)
        language=TextBlob(text[i]).detect_language()
        print(language)
        if language=='hi':
            updated_review.append(translation(text[i]))
        else:
            updated_review.append(text[i])
    return updated_review 




# helper function for spelling correction
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# function to correct the spelling of misspelled words in a review
def fixing(text):
    corpus=[]
    for i in range(1,len(text)):
        review=text[i]
        # review=reduce_lengthening(review)
        review_corrected=TextBlob(review).correct()
        corpus.append(review_corrected)
    return corpus


Cstopwords=set(stopwords.words('english')+list(punctuation))

lemma=WordNetLemmatizer()

# function to remove stopwords and perform lemmatization of the words in the review
def clean_review(review_column):
    review_corpus=[]
    for i in range(0,len(review_column)):
        review=review_column[i]
        #review=BeautifulSoup(review,'lxml').text
        review=re.sub('[^a-zA-Z]',' ',str(review))
        review=str(review).lower()
        review=word_tokenize(review)
        #review=[stemmer.stem(w) for w in review if w not in Cstopwords]
        review=[lemma.lemmatize(w) for w in review ]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus



# single_lang now contains the translated version of the reviews
single_lang=detect_language(review_column[1:])
# result contains the reviews after spelling correction
result=fixing(review_column[1:])
# review_corpus consists of the final cleaned dataset 
review_corpus=clean_review(result)

# print(df)
# print(len(df))
# print(len(review_column))
# print(review_corpus)
# df['clean_review']=review_corpus
# print(df['clean_review'])
# print(len(df['clean_review']))
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment=SentimentIntensityAnalyzer()
#
# # count=1
# # for sentences in df['clean_review']:
# #     ss=sentiment.polarity_scores(sentences)
# #     if ss['compound']>=0.05:
# #         df.set_value(count,"sentiment",1)
# #         count+=1
# #     elif ss['compound'] <= - 0.05:
# #         df.set_value(count, "sentiment", -1)
# #         count+= 1
# #     else:
# #         df.set_value(count, "sentiment", 0)
# #         count+= 1
# # print(df['sentiment'])


# function to find sentiment value of a review
def sentiment_value(paragraph):
    analyser = SentimentIntensityAnalyzer()
    result = analyser.polarity_scores(paragraph)
    score = result['compound']
    return round(score,1)
all_reviews=review_corpus
all_sent_values=[]
for i in range(0,length):
    all_sent_values.append(sentiment_value(all_reviews[i]))

# print(all_sent_values)


# Assigning sentiment values ranging from 1 to 5 to a review
SENTIMENT_VALUE = []
SENTIMENT = []
for i in range(0,length):
    sent = all_sent_values[i]
    if (sent<=1 and sent>=0.5):
        SENTIMENT.append('Positive')
        SENTIMENT_VALUE.append(1)
    elif (sent<0.5 and sent>0):
        SENTIMENT.append('Positive')
        SENTIMENT_VALUE.append(1)
    elif (sent==0):
        SENTIMENT.append('Neutral')
        SENTIMENT_VALUE.append(0)
    elif (sent<0 and sent>=-0.5):
        SENTIMENT.append('Negative')
        SENTIMENT_VALUE.append(-1)
    else:
        SENTIMENT.append('Negative')
        SENTIMENT_VALUE.append(-1)
#
#


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=length,min_df=5,ngram_range=(1,2))

X1=cv.fit_transform(review_corpus)
print(X1.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(min_df=5, max_df=0.95, max_features = length, ngram_range = ( 1, 2 ),
                              sublinear_tf = True)

tfidf=tfidf.fit(review_corpus)

X2=tfidf.transform(review_corpus)
print(X2.shape)

y=SENTIMENT_VALUE
# print(y.shape)

X=X2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))
print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))


# Logistic regression

from sklearn.linear_model import LogisticRegression as lr

model_lr=lr(random_state=0)

model_lr=lr(penalty='l2',C=1.0,random_state=0)
model_lr.fit(X_train,y_train)
# filename = 'finalized_model.sav'
# pickle.dump(model_lr, open(filename, 'wb'))
y_pred_lr=model_lr.predict(X_test)
# print(y_pred_lr)
print('accuracy for Logistic Regression :',accuracy_score(y_test,y_pred_lr))
# print('confusion matrix for Logistic Regression:\n',confusion_matrix(y_test,y_pred_lr))
# # print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_lr))
# # print('Precision score for Logistic Regression :',precision_score(y_test,y_pred_lr))
# # print('recall score for Logistic Regression :',recall_score(y_test,y_pred_lr))
# # print('AUC: ', roc_auc_score(y_test, y_pred_lr))


# Random Forest
# from sklearn.ensemble import RandomForestClassifier

# model_rf=RandomForestClassifier()
# model_rf.fit(X_train,y_train)
# y_pred_rf=model_rf.predict(X_test)
# print(y_pred_rf)
# print('accuracy for Random Forest Classifier :',accuracy_score(y_test,y_pred_rf))
# print('confusion matrix for Random Forest Classifier:\n',confusion_matrix(y_test,y_pred_rf))


# # Naive bayes
#
# # from sklearn.naive_bayes import MultinomialNB
# # model_nb=MultinomialNB()
# # model_nb.fit(X_train,y_train)
# # y_pred_nb=model_nb.predict(X_test)
# # print('accuracy for Naive Bayes Classifier :',accuracy_score(y_test,y_pred_nb))
# # print('confusion matrix for Naive Bayes Classifier:\n',confusion_matrix(y_test,y_pred_nb))
# # print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_nb))
# # print('Precision score for Logistic Regression :',precision_score(y_test,y_pred_nb))
# # print('recall score for Logistic Regression :',recall_score(y_test,y_pred_nb))
# # print('AUC: ', roc_auc_score(y_test, y_pred_nb))
predictions=[]
for y in y_pred_lr:
    x=[]
    x.append(y)
    predictions.append(x)
final=[]
for pred in predictions:
  if(pred==1):
    final.append("Positive")
  elif(pred==0):
    final.append("Neutral")
  else:
    final.append("Negative")

file = open('output.csv', 'w+', newline='')

# writing the data into the file
with file:
    write = csv.writer(file)
    write.writerows(final)
