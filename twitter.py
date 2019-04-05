import pyrebase ##library for database
import tweepy ##library for twitter api
import sys ##the sys library
# import pandas ##pandas dataframe library
import json ##json library
import datetime ##datetime library used to handle the created at attribute of tweets
from textblob import TextBlob ##textblob library used to getting sentiments of the tweet
import re ##regex library for cleaning the tweet
# import requests ##to request polarity from nltk server
import emoji ##to detect emojis in a tweet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer ##utlimate sentiment analyzer
analyser = SentimentIntensityAnalyzer() ##object initialization of vader sentiment analyzer
import got3 as got##get old tweets api
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from twython import Twython


with open('dictt.txt', 'r') as f:wordlist = [line.split(None, 1)[0] for line in f]

##vader sentiment
##handles slangs, emojis, marginal---- the ultimate 
def vader_sentiment(sentence):return float(analyser.polarity_scores(sentence)['compound'])

##function to extract emojis
def extract_emojis(str):return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

##function to insert datetime as json serializable object
def datetime_handler(x):
    if isinstance(x, datetime.datetime):return x.isoformat()
    raise TypeError("Unknown type")

## function to clean tweet and to pre process it before computing its sentiments
def clean_tweet(tweet): return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

## funtion to return the tweet sentiment using textblob library
def get_tweet_sentiment_textblob(tweet): return float(TextBlob(tweet) .sentiment.polarity)

###### driver code ##########
##initializing the database
config = {
    "apiKey": "AIzaSyBTCYOHXBWKSo5d_2rem9gGsPyIBMvweDc",
    "authDomain": "twitter-analysis-61655.firebaseapp.com",
    "databaseURL": "https://twitter-analysis-61655.firebaseio.com/",
    "storageBucket": "twitter-analysis-61655.appspot.com",
    "serviceAccount": "my_service.json"
}
firebase=pyrebase.initialize_app(config)
db=firebase.database()
print("database initialized")

## get all the keys from the twitter console
api_key="GhyNN1YJi3WNLYZ5nSOwVGBLl"
api_secret="aOoNZLrE2ytiwsN4ywk3FjNW9B6oDiskURcHwqJkToHSNom8YP"
access_token="239037296-XaDR0KimOVz6nD3EFO8scSeyQ8ypuG3BrpURQM0l"
access_token_secret="Zf6nvUFOVHO4kZdmKXS2a5Ln9F05iDuMT815AKOZcnKMr"

## create the auth and O-auth objects
auth = tweepy.AppAuthHandler(api_key, api_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
twitter = Twython(api_key, api_secret)
print("twitter initialized")

movies=["#birdbox","#venom","#ragnarok","#deadpool2","#infinitywar","#alita","#aquaman","#captainmarvel"] ##filtering the retweets of tweets
movies2=["birdbox","venom","ragnarok","deadpool2","infinitywar","alita","aquaman","captainmarvel"] ## just to store the keys of database
movie_imdb_rating=[67,68,79,78,85,76,72,71]
movie_release_date=["2018-11-12","2018-10-3","2017-11-3","2018-5-18","2018-4-27","2019-2-14","2018-12-21","2019-03-08"]

dictt={}
# print("gathering tweets..........")

# # iterating through the movie list
# for i in range(len(movies)):
#     dictt[movies2[i]]={}
#     xxx=0
#     tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movies[i]).setSince(movie_release_date[i]).setMaxTweets(3000).setLang(Lang="en")
#     c = got.manager.TweetManager.getTweets(tweetCriteria)
#     for tweet in range(len(c)):
#         tweet_user_location=c[tweet].geo
#         if(tweet_user_location==''):tweet_user_location="N/A"
#         tweet_text=c[tweet].text
#         tweet_text_cleaned=clean_tweet(tweet_text)
#         sentiment_textblob=get_tweet_sentiment_textblob(tweet_text_cleaned)
#         sentiment_vader=vader_sentiment(tweet_text_cleaned)
#         data={}
#         data["tweet_text_original"]=tweet_text
#         data["tweet_text_cleaned"]=tweet_text_cleaned
#         data["tweet_mentions"]=c[tweet].mentions
#         data["tweet_hashtags"]=c[tweet].hashtags
#         data["tweet_user_name"]=c[tweet].username
#         data["tweet_user_location"]=tweet_user_location
#         data["sentiment_textblob"]=sentiment_textblob
#         data["sentiment_vader"]=sentiment_vader
#         dictt[movies2[i]][xxx]=data
#         xxx=xxx+1
#         # print(movies2[i],"->",xxx)
#     print(movies2[i])

# # print(dict)
# db.child("new_new1").set(dictt)
# print("Done......")






##begin analysis out here 
best_sentiment=[]
user_names=[]
dict_user_names={}
print("Running.........")
for i in movies2:
    sentiment_textblob=[]
    sentiment_vader=[]
    sentiment=[]
    dict_user_names[i]={}
    
    ll=db.child("new_new1").child(i).get().val()
    for movie_data in ll:            
            sentiment_textblob.append(movie_data['sentiment_textblob'])
            sentiment_vader.append(movie_data['sentiment_vader'])
            for user_name in movie_data["tweet_user_name"].split(" "):
                if(user_name not in user_names):
                    user_names.append(movie_data["tweet_user_name"])
                sub_dict=dict_user_names[i]
                if(user_name not in sub_dict):
                    sub_dict[user_name]=([max(movie_data["sentiment_textblob"], movie_data["sentiment_vader"])], [])
                else:
                    sub_dict[user_name][0].append(max(movie_data["sentiment_textblob"], movie_data["sentiment_vader"]))
        
    textblob=sum(sentiment_textblob)/len(sentiment_textblob)*100+50
    vader=sum(sentiment_vader)/len(sentiment_vader)*100+50

    best_sentiment.append(float(max(textblob,vader)))
    
    print("Movie ", i, " sentiment - textblob: ",textblob, "%"," collected out of: ",len(sentiment_textblob)," #tweets")
    print("Movie ", i, " sentiment - vader: ",vader, "%"," collected out of: ",len(sentiment_vader)," #tweets")
    # print(len(dict_user_names[i]))
    print("\n")  

## visualization using graph
barWidth = 0.28
r1 = np.arange(len(movie_imdb_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_imdb_rating))], movies2)
plt.legend()
plt.show()

# print(len(user_names))
for i in movies2:
    edges=[]
    users=list(dict_user_names[i].keys())
    for y in range(0,len(users)):
        user_A=users[y]
        for x in range(y+1,len(users)):
            user_B=users[x]
            check=api.show_friendship(source_screen_name=user_A,target_screen_name=user_B)
            follows=check[0].following
            followed_back=check[0].followed_by
            if(follows):
                dict_user_names[user_A][1].append(user_B)
                edges.append((user_A,user_B))
            if(followed_back):
                dict_user_names[user_B][1].append(user_A)
                edges.append((user_B,user_A))
                