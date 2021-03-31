import re 
import nltk
import string 
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
        process1_tweet= re.sub(r'^RT','',tweet)
        process2_tweet= re.sub(r'https?:\/\/.*[\r\n]*','',process1_tweet)
        process3_tweet = re.sub(r'#','',process2_tweet)

        tokenizer = TweetTokenizer(preserve_case=True,strip_handles=True,reduce_len=True)
        tokenized_tweet = tokenizer.tokenize(process3_tweet)

        stopwords_english = stopwords.words('english')
        punctuation = string.punctuation
        clean_tweet = []
        for word in tokenized_tweet:
            if (word not in stopwords_english) and (word not in punctuation):
                clean_tweet.append(word)
            
        stemmer = PorterStemmer()
        tweet_stem = []
        for word in clean_tweet:
            tweet_stem.append(stemmer.stem(word))
        
        return tweet_stem



def build_freqs(tweets,ys):
    # {
    #     ('followfriday',1): 2  #--> followfriday appears 2 times in positive comment
    #     ('yippi',1): 3   # ---> yippi appears 3 times in positive comment
    #      tweets: a list of tweets
    #       ys: an m x 1 array with the sentiment label of each tweet
    #       (either 0 or 1)
    # }
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    # y is a m*1 matrix convertted to list of m length cotaining 
    # 1 for positive and 0 for negative
    for y,tweet in zip(yslist,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = positive_tweets+negative_tweets
print(len(tweets))


labels = np.append( np.ones((len(positive_tweets))) , np.zeros(len((negative_tweets))) )
print(labels.shape,'is shape of labels out of which 5000 are 1 and 5000 are 0')

freqs = build_freqs(tweets,labels)
print(len(freqs))

# print(freqs)

# TABLE OF WORD COUNTS
data = list() 

t1 = tweets[2277]
t1 = process_tweet(t1)
for word in t1:
    pos = 0
    neg = 0
    if (word,1) in freqs:
        pos = freqs[(word,1)]
    if (word,0) in freqs:
        neg = freqs[(word,0)]
    
    data.append([word,pos,neg])

print(data)