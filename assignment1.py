import re
import nltk
import string 
import numpy as np 
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer



all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")


def process_tweet(tweet):
    process1_tweet = re.sub(r'^RT','',tweet)
    process2_tweet = re.sub(r'https?:\/\/.*','',process1_tweet)
    process3_tweet = re.sub(r'#','',process2_tweet)
    process3_tweet = re.sub(r'[^a-zA-Z0-9]@[a-zA-Z0-9]*','',process3_tweet)
    process4_tweet = re.sub(r'@','',process3_tweet)

    tokenizer = TweetTokenizer(preserve_case=True,reduce_len=True,strip_handles=True)
    tweet_tokenizer = tokenizer.tokenize(process4_tweet)

    stopwords_english= stopwords.words("english")
    punctuation = string.punctuation
    clean_tweet = []
    for i in tweet_tokenizer:
        if (i not in stopwords_english) and (i not in punctuation):
            clean_tweet.append(i)
    
    stemmer = PorterStemmer()
    tweet_stem = []
    for i in clean_tweet:
        tweet_stem.append(stemmer.stem(i))

    return tweet_stem

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs={}
    for y,tweet in zip(yslist,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair]=1
    
    return freqs


def sigmoid(z):
    h = 1/(1+np.exp(-z))
    return h


train_positive = all_positive_tweets[:4000]
test_positive = all_positive_tweets[4000:]

train_negative = all_negative_tweets[:4000]
test_negative = all_negative_tweets[4000:]

train_x = train_positive + train_negative
test_x = test_positive + test_negative

train_y = np.append(np.ones((len(train_positive),1)),np.zeros((len(train_negative),1)), axis=0)
test_y = np.append(np.ones((len(test_positive),1)),np.zeros((len(test_negative),1)), axis=0)

def gradientDescent(x,y,theta,alpha,num_iters):
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x,theta)
        h = sigmoid(z)
        j = (-1/m)*(y.T @ np.log(h) + (1-y).T @ np.log(1-h))
        theta = theta - (alpha/m)*(x.T @ (h-y))
    j= float(j)
    return j , theta 


freqs = build_freqs(train_x,train_y)


def extract_features(tweet,freqs):
    ''' 
        tweet contain single tweet and freqs contain all the 
        frequncy  '''
    word_list = process_tweet(tweet)
    x = np.zeros((1,3))
    x[0,0]=1 # setting bias term equal to one
    for word in word_list:
        x[0,1] += freqs.get((word,1.0),0)
        x[0,2] += freqs.get((word,0.0),0)
    return x 

temp1 = extract_features(train_x[0],freqs)
# print(temp1)
tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
# print(tmp2)

X = np.zeros((len(train_x),3))
for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i],freqs)

Y = train_y 
J,theta = gradientDescent(X,Y,np.zeros((3,1)),1e-9,1500)
print(J,theta.shape)


def predict_tweet(tweet,freqs,theta):
    x = extract_features(tweet,freqs)
    y = sigmoid(np.dot(x,theta))
    return y 

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 
                'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))


def test_logistic_regression(test_x,test_y,freqs,theta):
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet,freqs,theta)
        if y_pred>0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    return accuracy

tmp_accuracy = test_logistic_regression(test_x,test_y,freqs,theta)
print(tmp_accuracy)


t1 = 'amazing amazing awesome'
print(predict_tweet(t1,freqs,theta))