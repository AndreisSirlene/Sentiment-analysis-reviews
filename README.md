# Amazon reviews - Sentiment Analysis

> “A fool's brain digests philosophy into madness, science into superstition and art into pedantry. Hence a university education. ” George Bernard Shaw


# :pushpin: Table of Contents

* [Introduction](#memo-introduction)
* [Preprocessing and Cleaning dataset](#rocket-preprocessing-and-cleaning-dataset)
* [Story Generation and Visualization from reviews](#festivus-story-generation-and-visualization-from-reviews)
* [Text reviews](#runner-text-reviews)
* [Extracting Features from Cleaned reviews](#worker-extracting-features-from-cleaned-reviews)
* [Model Building: Sentiment Analysis](#closedbook-model-building-sentiment-analysis)
* [Group Project](#tada-group-project)


# :memo: Introduction

This project proposed the analyse of consumer behaviour in order to assist a business to build an effective and targeted marketing strategy. 

To do this we will build predictive models on data sets compiled from e-Commerce giants, Amazon & Walmart datasets.

•***Build a Sentiment Analysis model*** to predict the effect on sales in relation to customer reviews.

•***Build a Market Basket Model on the Amazon dataset***. This will enable the enterprise to predict consumer behaviour by suggesting complimentary goods to purchase. 

•***Analyse the conversion rates*** in this dataset also with a view to building a model to increase these.


Examine customer sensitivity to price by ***building a linear regression model on the Walmart*** dataset.

The retail industry has taken a 180 degree turn with the rise in online shopping. In 2019, retail e-commerce sales worldwide amounted to 3.53 trillion US dollars and e-retail revenues are projected to grow to 6.54 trillion US dollars by 2022.

It was predicted that in 2020 the global e-commerce market exceed 4 trillion dollars, and one in every four online consumers purchases from stores once a week according to Invespcro (2020) report.

# :rocket: Preprocessing and Cleaning dataset

**Importing Libraries**

* Visualization libraries

Pandas, Seaborn, Matplotlib.pyplot, Plotly.express as px

* NLTK libraries

nltk, re, Wordcloud, PorterStemmer, TfidfVectorizer, Stopwords, Word_tokenize, TextBlob

* Machine Learning libraries

sklearn, SVC, LabelEncoder, StandardScaler, Preprocessing import normalize, ExtraTreesClassifier, GridSearchCV

* Machine Learning Models

LogisticRegression, DecisionTreeClassifier, BernoulliNB, KNeighborsClassifier, OneVsRestClassifier

model_selection import train_test_split, label_binarize

* Other Libraries

Counter, SMOTE, CountVectorizer


⌛️ Dataset features

uniq_id, product_name, manufacturer, price, number_available_in_stock, number_of_reviews, number_of_answered_questions, average_review_rating,
amazon_category_and_sub_category, customers_who_bought_this_also_bought, description, product_information, product_description, items_customers_buy_after_viewing_this_item, customer_questions_and_answers, customer_reviews, sellers 


# :festivus: Story Generation and Visualization from reviews


By go further in the exploratory data analysis on texts we are try to understand what features contributes to the sentiment category.

Prior analysis assumptions:

* Higher the rate the sentiment becomes positive

* There are be many positive sentiment reviews which lead to bias

* These assumptions will be verified with our plots also we will do text analysis

# :runner: Text reviews

### Review Text Ponctuation and creat stop words
NLKT stop words contains words like not, hasn't, would'nt which actually conveys a negative sentiment. If we remove that it will end up contradicting the target variable(sentiment). So I have curated the stop words which doesn't have any negative sentiment or any negative alternatives.


### Creating additional features for text analysis.

Create polarity, review length and word count

Polarity: By using Textblob for figuring out the rate of sentiment between [-1,1] where -1 is negative and 1 is positive

Review length: length of the review which includes each letters and spaces

Word length: It measures how many words are in the customer review column

# :worker: Extracting Features from Cleaned reviews

Before we build the model for our sentiment analysis, it is required to convert the review texts into vector formation as computer cannot understand words and their sentiment. In this project, we are going to use TF-TDF method to convert the texts.

**Encoding target variable-sentiment**

### Stemming the reviews
Stemming is a method of deriving root word from the inflected word. Here we extract the customer reviews and convert the words to its root word.

There is another technique knows as Lemmatization where it converts the words into root words which has a semantic meaning.

### Handling Imbalance target using feature-SMOTE
We noticed that we got a lot of positive sentiments compared to negative and neutral. So it is crucial to balanced the classes in such situation. SMOTE(Synthetic Minority Oversampling Technique)is used to balance out the imbalanced dataset problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them.

# 😝: Model Building: Sentiment Analysis

Sentiment Analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. 
Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before.It is quite hard for a human to go through each single line and identify the emotion being the user experience. With machine learning models nowadays we can automatically analyzing customer feedback, from product reviews and survey responses to social media conversations for example, which allows to tailor products and services to meet customer needs.


# :tada: Group Project

CCT COLLEGE DUBLIN

**Higher Diploma in Science in Data Analytics for Business**

Members: 

MONIQUE DIAZ

RODRIGO MACHADO

SARAH PARSONS-LAPPIN

SIRLENE ANDREIS

Under Supervision of: GRAHAM GLANVILLE & MARK MORRISSEY

Released in March 2021.

This project is under the [MIT license](https://github.com/AndreisSirlene/Sentiment-reviews-AWS/blob/master/LICENSE).

Made with love by [Sirlene Andreis](https://github.com/AndreisSirlene) 💚🚀
