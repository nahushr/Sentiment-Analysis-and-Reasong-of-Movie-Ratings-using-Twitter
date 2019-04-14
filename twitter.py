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
import networkx as nx
with open('dictt.txt', 'r') as f:wordlist = [line.split(None, 1)[0] for line in f]
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

def display_topics(model, feature_names, no_top_words):
    topic_list=[]
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx), end='')
        topic_list.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topic_list


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
# db.child("new_movie_dataset").remove()
# sys.exit(1)
print("database initialized")

## get all the keys from the twitter console
api_key="GhyNN1YJi3WNLYZ5nSOwVGBLl"
api_secret="aOoNZLrE2ytiwsN4ywk3FjNW9B6oDiskURcHwqJkToHSNom8YP"
access_token="239037296-XaDR0KimOVz6nD3EFO8scSeyQ8ypuG3BrpURQM0l"
access_token_secret="Zf6nvUFOVHO4kZdmKXS2a5Ln9F05iDuMT815AKOZcnKMr"

## create the auth and O-auth objects
auth = tweepy.AppAuthHandler(api_key, api_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
print("twitter initialized")

movies=["#birdbox","#venom","#ragnarok","#deadpool2","#infinitywar","#alita","#aquaman","#captainmarvel","#dumbo","#shazam","#logan"] ##filtering the retweets of tweets
movies2=["birdbox","venom","ragnarok","deadpool2","infinitywar","alita","aquaman","captainmarvel","dumbo","shazam","logan"] ## just to store the keys of database
movie_imdb_rating=[67,68,79,78,85,76,72,71,67,77,81]
movie_release_date=["2018-11-12","2018-10-3","2017-11-3","2018-5-18","2018-4-27","2019-2-14","2018-12-21","2019-03-08","2019-03-29","2019-04-05","2019-03-03"]

dictt={}
print("gathering tweets..........")

# iterating through the movie list
for i in range(len(movies)):
    dictt[movies2[i]]={}
    xxx=0
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movies[i]).setSince(movie_release_date[i]).setMaxTweets(3000).setLang(Lang="en")
    c = got.manager.TweetManager.getTweets(tweetCriteria)
    for tweet in range(len(c)):
        tweet_user_location=c[tweet].geo
        if(tweet_user_location==''):tweet_user_location="N/A"
        tweet_text=c[tweet].text
        tweet_text_cleaned=clean_tweet(tweet_text)
        sentiment_textblob=get_tweet_sentiment_textblob(tweet_text_cleaned)
        sentiment_vader=vader_sentiment(tweet_text_cleaned)
        data={}
        data["tweet_text_original"]=tweet_text
        data["tweet_text_cleaned"]=tweet_text_cleaned
        data["tweet_mentions"]=c[tweet].mentions
        data["tweet_hashtags"]=c[tweet].hashtags
        data["tweet_user_name"]=c[tweet].username
        data["tweet_user_location"]=tweet_user_location
        data["sentiment_textblob"]=sentiment_textblob
        data["sentiment_vader"]=sentiment_vader
        dictt[movies2[i]][xxx]=data
        xxx=xxx+1
        # print(movies2[i],"->",xxx)
    print(movies2[i],"->",xxx)

# print(dict)
db.child("new_new1").set(dictt)
print("Done......")






##begin analysis out here 
dict_user_movie={}
dumb_users=[]
for i in movies2:
    ll=db.child("new_new1").child(i).get().val()
    for movie_data in ll:
         for user_name in movie_data["tweet_user_name"].split(" "):
                if user_name not in dict_user_movie:
                    if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                        dict_user_movie[user_name]=[]
                        dict_user_movie[user_name].append((i,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):
                        dumb_users.append(user_name)
                        
                else:
                    if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                        dict_user_movie[user_name].append((i,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):
                        dumb_users.append(user_name)
final_dict={}
for key,value in dict_user_movie.items():
    temp_dict={}
    for v in value:
        if v[0] not in temp_dict:
            temp_dict[v[0]]=([v[1]],[movie_imdb_rating[movies2.index(v[0])]])
        else:
            temp_dict[v[0]][0].append(v[1])
    final_dict[key]=temp_dict
super_users=[]
for key, value in final_dict.items():
    cc=0
    for key2,value2 in value.items():
        actual_rating=value2[1][0]
        for ratings in value2[0]:
            tweet_rating=(ratings/2)*100+50
            if(tweet_rating>actual_rating-8 and tweet_rating<actual_rating+8):cc+=1
    if(cc>=3):super_users.append(key)

print("Super_users",len(super_users))
print("Dumb_users:",len(dumb_users))

best_sentiment=[]
user_names=[]
print("Running.........")
dict_movies={}
dict_neg={}
dict_neu={}
dict_pos={}
no_features = 1000
no_topics = 10
no_top_words = 10
topics_all={}
topics_neg={}
topics_neu={}
topics_pos={}
for i in movies2:
    sentiment_textblob=[]
    sentiment_vader=[]
    node_neg=[]
    node_neu=[]
    node_pos=[]
    edges_neg=[]
    edges_neu=[]
    edges_pos=[]
    G_neg=nx.Graph()
    G_neu=nx.Graph()
    G_pos=nx.Graph()
    dict_movies[i]=[]
    dict_neg[i]=[]
    dict_neu[i]=[]
    dict_pos[i]=[]
    ll=db.child("new_new1").child(i).get().val()
    for movie_data in ll:
            dict_movies[i].append(movie_data["tweet_text_cleaned"])
            if(movie_data["sentiment_vader"]<0):dict_neg[i].append(movie_data["tweet_text_cleaned"])
            elif(movie_data["sentiment_vader"]==0):dict_neu[i].append(movie_data["tweet_text_cleaned"])
            else:dict_pos[i].append(movie_data["tweet_text_cleaned"])  

            for user_name in movie_data["tweet_user_name"].split(" "):
                if(user_name in super_users):              
                    sentiment_textblob.append(movie_data['sentiment_textblob']+0.2)
                    sentiment_vader.append(movie_data['sentiment_vader']+0.2)
                elif(user_name in dumb_users):
                    sentiment_textblob.append(movie_data['sentiment_textblob']-0.05)
                    sentiment_vader.append(movie_data['sentiment_vader']-0.05)
                else:
                    sentiment_textblob.append(movie_data['sentiment_textblob'])
                    sentiment_vader.append(movie_data['sentiment_vader'])


            if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])==0):
                userA=movie_data["tweet_user_name"].split(" ")[0]
                for user_name in movie_data["tweet_user_name"].split(" "):
                    node_neu.append((user_name,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    for user_name2 in movie_data["tweet_user_name"].split(" "):
                        if( ((user_name,user_name2) not in edges_neu) and ((user_name2,user_name) not in edges_neu) ):
                            edges_neu.append((user_name,user_name2))
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):
                userA=movie_data["tweet_user_name"].split(" ")[0]
                for user_name in movie_data["tweet_user_name"].split(" "):
                    node_neg.append((user_name,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    for user_name2 in movie_data["tweet_user_name"].split(" "):
                        if( ((user_name,user_name2) not in edges_neg) and ((user_name2,user_name) not in edges_neg) ):
                            edges_neg.append((user_name,user_name2))
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                userA=movie_data["tweet_user_name"].split(" ")[0]
                for user_name in movie_data["tweet_user_name"].split(" "):
                    node_pos.append((user_name,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    for user_name2 in movie_data["tweet_user_name"].split(" "):
                        if( ((user_name,user_name2) not in edges_pos) and ((user_name2,user_name) not in edges_pos) ):
                            edges_pos.append((user_name,user_name2))
    
            

    # for ks in range(0,len(node_neg)):
    #     userA=node_neg[ks][0]
    #     for ks2 in range(ks+1,len(node_neg)):
    #         userB=node_neg[ks2][0]
    #         if( ((userA,userB) or (userB,userA)) in edges_neg and userA==userB):pass
    #         else:
    #             check=api.show_friendship(source_screen_name=userA,target_screen_name=userB)
    #             follows=check[0].following
    #             followed_back=check[0].followed_by
    #             if(follows or followed_back):
    #                 edges_neg.append((userA,userB))
    
    # for ks in range(0,len(node_neu)):
    #     userA=node_neu[ks][0]
    #     for ks2 in range(ks+1,len(node_neu)):
    #         userB=node_neu[ks2][0]
    #         if( ((userA,userB) or (userB,userA)) in edges_neu and userA==userB):pass
    #         else:
    #             check=api.show_friendship(source_screen_name=userA,target_screen_name=userB)
    #             follows=check[0].following
    #             followed_back=check[0].followed_by
    #             if(follows or followed_back):
    #                 edges_neu.append((userA,userB))
    
    # for ks in range(0,len(node_pos)):
    #     userA=node_pos[ks][0]
    #     for ks2 in range(ks+1,len(node_pos)):
    #         userB=node_pos[ks2][0]
    #         if( ((userA,userB) or (userB,userA)) in edges_pos and userA==userB):pass
    #         else:
    #             check=api.show_friendship(source_screen_name=userA,target_screen_name=userB)
    #             follows=check[0].following
    #             followed_back=check[0].followed_by
    #             if(follows or followed_back):
    #                 edges_pos.append((userA,userB))

    G_neg.add_nodes_from(node_neg)
    G_neu.add_nodes_from(node_neu)
    G_pos.add_nodes_from(node_pos)

    G_neg.add_edges_from(edges_neg)
    G_neu.add_edges_from(edges_neu)
    G_pos.add_edges_from(edges_pos)

    path1="gephi/"+i+"_neg.gexf"
    path2="gephi/"+i+"_neu.gexf"
    path3="gephi/"+i+"_pos.gexf"

    nx.write_gexf(G_neg, path1)
    nx.write_gexf(G_neu, path2)
    nx.write_gexf(G_pos, path3)

                

        
    print(i,len(node_neg),len(node_neu),len(node_pos),len(edges_neg),len(edges_neu),len(edges_pos))
    textblob=sum(sentiment_textblob)/len(sentiment_textblob)*100+50
    vader=sum(sentiment_vader)/len(sentiment_vader)*100+50

    best_sentiment.append(float(max(textblob,vader)))
    
    print("Movie ", i, " sentiment - textblob: ",textblob, "%"," collected out of: ",len(sentiment_textblob)," #tweets")
    print("Movie ", i, " sentiment - vader: ",vader, "%"," collected out of: ",len(sentiment_vader)," #tweets")
    print(movie_imdb_rating)
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

print(">>>>>>>>> Topic Modelling on all tweets <<<<<<<<<<<")
for i in movies2:
    topics_all[i]=[]
    print("****** ",i," *******")
    documents=dict_movies[i]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    topics_all[i]=display_topics(lda, tf_feature_names, no_top_words)

print(">>>>>>>>> Topic Modelling on negative tweets tweets <<<<<<<<<<<")
for i in movies2:
    topics_neg[i]=[]
    print("****** ",i," *******")
    documents=dict_neg[i]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    topics_neg[i]=display_topics(lda, tf_feature_names, no_top_words)

print(">>>>>>>>> Topic Modelling on neutral tweets <<<<<<<<<<<")
for i in movies2:
    topics_neu[i]=[]
    print("****** ",i," *******")
    documents=dict_neu[i]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    topics_neu[i]=display_topics(lda, tf_feature_names, no_top_words)

print(">>>>>>>>> Topic Modelling on positive tweets <<<<<<<<<<<")
for i in movies2:
    topics_pos[i]=[]
    print("****** ",i," *******")
    documents=dict_pos[i]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    topics_pos[i]=display_topics(lda, tf_feature_names, no_top_words)

pickle_out = open("topics_all","wb")
pickle.dump(topics_all, pickle_out)
pickle_out.close()

pickle_out = open("topics_neg","wb")
pickle.dump(topics_neg, pickle_out)
pickle_out.close()

pickle_out = open("topics_neu","wb")
pickle.dump(topics_neu, pickle_out)
pickle_out.close()

pickle_out = open("topics_pos","wb")
pickle.dump(topics_pos, pickle_out)
pickle_out.close()

with open('best_sentiment.txt', 'w') as f:
    for item in best_sentiment:
        f.write("%s\n" % item)

with open('movie_imdb_rating.txt', 'w') as f:
    for item in movie_imdb_rating:
        f.write("%s\n" % item)



