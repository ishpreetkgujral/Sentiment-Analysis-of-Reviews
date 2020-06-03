# Sentiment-Analysis-of-Reviews
Sentiment Analysis of product reviews 
This project aims at producing sentiment values for the product reviews.
Algorithm-
1. Firstly after extracting all the reviews, they are converted to english language as we mostly observe the reviews to be in highlish language
2. The reviews returned are tokenized, stopwords removed, lemmatized and HTML tags removed using Beautiful Soup library.
3. The sentiment values are assigned using Vader tool 
4. Using Naive Bayes, Logistic Regression and Random Forest one by one we check the accuracy of the sentiment scores.
5. A csv containing the result is generated at the end as per the reviews.

Result-
Logistic Regression gives the best accuracy score.
