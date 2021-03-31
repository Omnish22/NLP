import nltk 
# nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt 
# print(nltk.__version__)
# nltk.download('stopwords')
import re # for regular expression
import string # for string operation
from nltk.corpus import stopwords  # for the stopwords
from nltk.stem import PorterStemmer # for stemming of words
from nltk.tokenize import TweetTokenizer  # for tekenize the strings
import random


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
print("POSITIVE TWEETS")
print(len(all_positive_tweets))
print(type(all_positive_tweets))
print(type(all_positive_tweets[0]))

all_negative_tweets = twitter_samples.strings('negative_tweets.json')
print("\nNEGATIVE TWEETS")
print(len(all_negative_tweets))
print(type(all_negative_tweets))
print(type(all_negative_tweets[0]))


# def plot_pie(sizes,labels):
#     fig = plt.figure(figsize=(10,10))
#     plt.pie(sizes,labels = labels,autopct='%1.1f%%',shadow=True,startangle=90)
#     plt.axis('equal')
#     plt.show()

# labels = 'Positive','Negative'
# plot_pie([len(all_positive_tweets),len(all_negative_tweets)],labels)


print(all_positive_tweets[random.randint(0,5000)])
print(all_negative_tweets[random.randint(0,5000)])

# remove all RT  , Hyperlinks and #
tweet = all_positive_tweets[2277]
print(tweet)
tweet2= re.sub(r'^RT[\s]+','',tweet)
tweet2 = re.sub(r'https?:\/\/.*[\r\n]*','',tweet2)
tweet2 = re.sub('#','',tweet2)
print("REMOVE RT HYPERLINKS #")
print(tweet2)

# tokenize the words and convert them into lower case
tokenizer = TweetTokenizer(preserve_case=True,strip_handles=True,reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)
print('TOKENIZER')
print(tweet_tokens)

# to remove stopwords and punctuation from vocabulary
stopwords_english = stopwords.words('english')
print("STOP WORDS")
print(stopwords_english)

print("\n PUNTUATION")
print(string.punctuation)

clean_tweet=[]
for word in tweet_tokens:
    if (word not in stopwords_english and word not in string.punctuation):
        clean_tweet.append(word)

print("\nRemove stop words and punctuation from tweets:")
print(clean_tweet)

stemmer = PorterStemmer()
tweet_stem = []
for word in clean_tweet:
    stemmed_word = stemmer.stem(word)
    tweet_stem.append(stemmed_word)

print("STEMMED WORDS ARE")
print(tweet_stem)
