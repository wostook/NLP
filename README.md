# LDA topic modelling and sentiment analysis of scraped Twitter data
![image](https://github.com/wostook/NLP/blob/master/data/WordCloud.JPG)
In a project I scraped arund 13000 tweets with #conspiracytheory tag, posted from the 1st of July till the 10th of August 2020. I processed text and extracted 7 main topics, gaining insight about the most discussed issues and the words that frequently appear within each topic. Additionally I ran sentiment analysis on the tweets, trying to get know how people feel about topics they write about. 

## Motivation 

Motivation was twofold. One thing was making use of text mining and computing power to get insight from thousands of tweets without need to read them physically. It can be obviously a powerful tool with many potential uses in marketing, journalism, social media monitoring, recommending machines, etc..

Another thing was to figure out what are the most common topics which are likely to fall under the ‘conspiracy’ headline. With a huge information overload on the Internet and the set of inborn cognitive biases, one should be able to spot a topic that is likely to be distorted by conspiracies and fake news. It seems it’s becoming increasingly valuable skill nowadays, from both personal and collective perspective–to filter out what matters and is most likely to be truth. 

## Tools and pipeline 

In a nutshell, I scraped data using Twitterscaper API, processed text with NTLK and Gensim APIs, explored data and built a corpus. For topic modelling I used Latent Dirichlet Allocation model which is unsupervised machine learning algorithm that helps discover hidden semantic structures in set of documents. It represent documents as mixture of topics with certain probabilities of occurrence of words, where both topic per document model and word per topic are modelled as Dirichlet distributions. To visualize and interpret topics and words, I used pyLDAvis. Sentiment analysis was run with VADER- a lexicon and rule based API tool. 

I used the following pipeline:

1.	Scrape data from twitter
2.	EDA
3.	Text pre-processing:
 - Convert into lower case
 - Remove numbers and special characters 
 - Tokenize and remove short words
 - Tag POS and lemmatize 
 - Filter stop words
 - Make n-grams
4.	Calculate word frequency and make wordclouds
5.	Create dictionary, BOW and transform into TF-IDF
6.	Run LDA for set of numbers of topics, calculate coherence scores
7.	Choose and run best model
8.	Run VADER sentiment analysis 
9.	Visualize topics and sentiment  

## Prerequisites

The project was done with [Python 3.7.4](https://www.python.org/downloads/release/python-374/) and the following libraries:

- [TwitterScraper](https://github.com/taspinar/twitterscraper)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [vaderSentiment](https://pypi.org/project/vaderSentiment/)
- [Wordcloud](https://amueller.github.io/word_cloud/)
- [NLTK](http://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)

## Data and Code

[Twitter Data](./data/Tweets_conspiracytheory)

Code provided in `LDA_sentiment_twitter.ipynb`

## Interactive visualization

[Link to pyLDAvis topic visualization](https://htmlpreview.github.io/?https://github.com/wostook/NLP/blob/master/data/pyLDAvis.html)
