#!/usr/bin/env python
# coding: utf-8

# Downloading all the dependencies:<br/>
# pyrebase -> the database where we are storing all our data. It is a mongoDB database hosted on google cloud <br/>
# tweepy -> the tweepy library is used for accessing all the twitter data and other functionalities from the official twitter developer api <br/>
# json -> the database which we use gives us a json formatted to data to handle such data we have used the json library<br/>
# datetime -> a datetime package to convert the string formatted date to actual date<br/>
# textblob -> a naive bayes based machine learning package to calculate the sentiment analysis of the tweets<br/>
# re -> a regular expression package to clean the data before passing it into the textblob library<br/>
# vadersentiment -> a lexicon based sentiment analysis package<br/>
# numpy -> to handle mathematical calculations<br/>
# matplotlib -> for data visualization<br/>
# pandas -> a data framework package to handle the data<br/>
# networkx -> to from graph networks and produce a gephi file for data visualization<br/>
# sklearn -> to use the topic modelling libraries built in sklearn<br/>
# pickle -> package to handle dictionary data and store the dictionary data in a file on local machine<br/>
# matplotlib_venn -> used to plot venn diagrams for the topics in topic modelling<br/>

# In[1]:


#get_ipython().system('pip install pyrebase')
#get_ipython().system('pip install tweepy')
#get_ipython().system('pip install json')
#get_ipython().system('pip install datetime')
#get_ipython().system('pip install textblob')
#get_ipython().system('pip install re')
#get_ipython().system('pip install vadersentiment')
#get_ipython().system('pip install numpy')
#get_ipython().system('pip install matplotlib.pyplot')
#get_ipython().system('pip install pandas')
#get_ipython().system('pip install networkx')
#get_ipython().system('pip install sklearn')
#get_ipython().system('pip install pickle')
#get_ipython().system('pip install matplotlib_venn')
import nltk
#nltk.download('movie_reviews')
#get_ipython().system('python -m textblob.download_corpora')
#get_ipython().system('pip install pyquery')


# In[2]:


import pyrebase ##library for database
import tweepy ##library for twitter api
import json ##json library
import datetime ##datetime library used to handle the created at attribute of tweets
from textblob import TextBlob ##textblob library used to getting sentiments of the tweet
from textblob.sentiments import NaiveBayesAnalyzer
import re ##regex library for cleaning the tweet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer ##utlimate sentiment analyzer
analyser = SentimentIntensityAnalyzer() ##object initialization of vader sentiment analyzer
# import got3 as got   ##get old tweets api ##uncomment this line to gather tweets using got3
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
import operator


# Function to display all the topics in topic modelling

# In[3]:


##https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d
def display_topics(model, feature_names, no_top_words):
    topic_list=[]
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx), end='')
        topic_list.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topic_list
##https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d


# Sentiment analysis using vadersentiment:<br/>
# -> this package produces the best results on our data<br/>
# -> this is a lexicon based sentiment analysis package<br/>
# -> the polarity range is from -1 to 1 where 0 is the neutral and -1 is the most negative<br/>
# -> we do not need to remove emojis from the text vadersentiment handles emoji polarity as well <br/>
# -> we do not need to remove slangs from the text vadersentiment handles that too<br/>
# -> vadersentiment also handles punctuations like more exclamation marks more is the importance of the sentence so more exlamation marks with a negative sentence reduces the polarity of the sentence and vice versa<br/>
# -> other features includes handling conjuctions, preceding Tri-gram and degree modifiers.<br/>
# (idea referenced from [1])<br/>

# In[4]:


##https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
def vader_sentiment(sentence):return float(analyser.polarity_scores(sentence)['compound'])
##https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f


# In[5]:


def datetime_handler(x):
    if isinstance(x, datetime.datetime):return x.isoformat()
    raise TypeError("Unknown type")


# The textblob package does not handle emojis or links or usernames thus we need to clean the data before passing it into the textblob library. So before passing it into the textblob library we need to remove the links and the user mentions from the tweeta

# In[6]:


##https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
def clean_tweet(tweet): return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())
def get_tweet_sentiment_textblob(tweet): return float(TextBlob(tweet).sentiment.polarity)
def get_tweet_sentiment_textblob_naive_bayes_classifer(tweet):
    blob = TextBlob(tweet, analyzer=NaiveBayesAnalyzer())
    return (max(blob.sentiment.p_pos, blob.sentiment.p_neg))
##https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/


# For this project we are using a firebase database which is a mongoDB formatted database hosted on google cloud. the main purpose of using this database over any other database or even a traditional text file or excel file is because all the data can be centralized at one place so we dont need to pass over the excel files again and again to other team members.
# Also the data insertion and selection of almost 24,000 tweets is handled effectively in this over traditional SQL databases.

# In[7]:


config = {
    "apiKey": "AIzaSyBTCYOHXBWKSo5d_2rem9gGsPyIBMvweDc",
    "authDomain": "twitter-analysis-61655.firebaseapp.com",
    "databaseURL": "https://twitter-analysis-61655.firebaseio.com/",
    "storageBucket": "twitter-analysis-61655.appspot.com",
    "serviceAccount": "my_service.json"
} ## account details
firebase=pyrebase.initialize_app(config)
db=firebase.database() ## initialize the database
print("database initialized")


# In[8]:


api_key="GhyNN1YJi3WNLYZ5nSOwVGBLl"
api_secret="aOoNZLrE2ytiwsN4ywk3FjNW9B6oDiskURcHwqJkToHSNom8YP"
access_token="239037296-XaDR0KimOVz6nD3EFO8scSeyQ8ypuG3BrpURQM0l"
access_token_secret="Zf6nvUFOVHO4kZdmKXS2a5Ln9F05iDuMT815AKOZcnKMr"
auth = tweepy.AppAuthHandler(api_key, api_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True) ##initialize the twitter developers api
print("twitter initialized")


# We are using a '#' followed by the movie name to collect our twitter data.
# For collecting the twitter data we are using the GOT3 package. Get old tweets for python3 which is a package used for collecting old tweets and some of the variables associated with that tweet.
# Along with that we also have our ground-truth from movie rating websites such as: IMDB, Rotten tomatoes,
# Movie release date is noted as well.<br/>
# We are only collecting tweets after the movie release date because that would give us the correct estimate of how the true movie reviews were after people actually watched it and avoided tweets that were tweeted before even the movie was released as those do not contribute to the actual review of a movie.

# In[9]:


movies=["#birdbox","#venom","#ragnarok","#deadpool2","#infinitywar","#alita","#aquaman","#captainmarvel","#dumbo","#shazam","#logan"] ##filtering the retweets of tweets
movies2=["birdbox","venom","ragnarok","deadpool2","infinitywar","alita","aquaman","captainmarvel","dumbo","shazam","logan"] ## just to store the keys of database
movie_imdb_rating=[67,68,79,78,85,76,72,71,67,77,81]
movie_rotten_tomatoes_rating=[67,81,87,85,91,94,76,56,57,88,90]
movie_metratic_rating=[67,35,74,66,68,54,55,64,51,71,77]
movie_moviefone_rating=[67,63,77,81,86,88,71,36,58,80,84]
movie_rating=[]
for i in range(len(movies)):
    rating=movie_imdb_rating[i]+movie_rotten_tomatoes_rating[i]+movie_moviefone_rating[i]+movie_metratic_rating[i]
    rating=rating/4
    movie_rating.append(rating)
movie_release_date=["2018-11-12","2018-10-3","2017-11-3","2018-5-18","2018-4-27","2019-2-14","2018-12-21","2019-03-08","2019-03-29","2019-04-05","2019-03-03"]


# In[ ]:


##https://github.com/Jefferson-Henrique/GetOldTweets-python/tree/master/got3 (tweet collection referenced)
# db.child("new_movie_dataset").remove() ## to delete the table from the database
dictt={}
print("gathering tweets..........")
# iterating through the movie list
for i in range(len(movies)):
    dictt[movies2[i]],xxx={},0
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movies[i]).setSince(movie_release_date[i]).setMaxTweets(3000).setLang(Lang="en")
    c = got.manager.TweetManager.getTweets(tweetCriteria)
    for tweet in range(len(c)):
        tweet_user_location=c[tweet].geo
        if(tweet_user_location==''):tweet_user_location="N/A" ## if the location is not available just store N/A in the database
        tweet_text=c[tweet].text
        tweet_text_cleaned=clean_tweet(tweet_text)
        sentiment_textblob=get_tweet_sentiment_textblob(tweet_text_cleaned)
        sentiment_vader=vader_sentiment(tweet_text_cleaned)
        sentiment_textblob_naive=get_tweet_sentiment_textblob_naive_bayes_classifer(tweet_text_cleaned)
        data={}
        data["tweet_text_original"]=tweet_text
        data["tweet_text_cleaned"]=tweet_text_cleaned
        data["tweet_mentions"]=c[tweet].mentions
        data["tweet_hashtags"]=c[tweet].hashtags
        data["tweet_user_name"]=c[tweet].username
        data["tweet_user_location"]=tweet_user_location
        data["sentiment_textblob"]=sentiment_textblob
        data["sentiment_vader"]=sentiment_vader
        data["sentiment_textblob_naive"]=sentiment_textblob_naive
        dictt[movies2[i]][xxx]=data
        xxx=xxx+1
    print(movies2[i],"->",xxx)
db.child("new_new1").set(dictt) ##pushing all the data in the database at once in the head node new_new_1
print("Tweet Collecting Done!......") 


# In[11]:


print("sentiment analysis before doing anything")
best_sentiment,best_sentiment2,best_sentiment3=[],[],[]
for i in movies2:
    sentiment_vader,sentiment_textblob,sentiment_texblob_naive=[],[],[]
    ll=db.child("new_new1").child(i).get().val()
    for movie_data in ll:
        for xx in movie_data["tweet_user_name"].split(" "):
            sentiment_vader.append(movie_data["sentiment_vader"])
            sentiment_textblob.append(movie_data["sentiment_textblob"])
            sentiment_texblob_naive.append(movie_data["sentiment_textblob_naive"])
    vader=sum(sentiment_vader)/len(sentiment_vader)*100+50
    textblob=sum(sentiment_textblob)/len(sentiment_textblob)*100+50
    textblob_naive=sum(sentiment_texblob_naive)/len(sentiment_texblob_naive)*100
    best_sentiment.append(float(textblob))
    best_sentiment2.append(float(vader))
    best_sentiment3.append(float(textblob_naive))
    print("Movie ", i, " sentiment - textblob: ",textblob, "%"," collected out of: ",len(sentiment_textblob)," #tweets")
    print("Movie ", i, " sentiment - textblob naive: ",textblob_naive, "%"," collected out of: ",len(sentiment_texblob_naive)," #tweets")
    print("Movie ", i, " sentiment - vader: ",vader, "%"," collected out of: ",len(sentiment_vader)," #tweets")
#     print(movie_rating)
    print("\n")  


# In[12]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("lexicon textblob")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment[i]
    print(movies2[i], " -> ",error)

## referencing needed


# In[13]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("Machine learning based Rating textblob")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment3, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment3[i]
    print(movies2[i], " -> ",error)

## referencing needed


# In[14]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("lexicon based rating")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment2, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
# plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment2[i]
    print(movies2[i], " -> ",error)


# To improve our results what we are doing is we are classifying users to two types super_users or dumb_users. 
# <b>Super Users:</b> These are the users who give the most accurate review of the movie and their sentiment score should be given extra score
# <b>Poor Users:</b> These are the users who give the least accurate review of the movie and their sentiment score should be given a lower score than calculated.
# Condition for becoming a super user: should have more than 5 tweets atlest for all the movies and minimum 5 movies should be within the range of (+8, ground_truth,-8)
# If the user rating is far off from the ground truth he/she is added in the dumb_user list such as falling far off by a score of .6 or more in the polarity score.   

# In[15]:


dict_user_movie,dumb_users,final_dict,super_users={},[],{},[]
for i in movies2:
    ll=db.child("new_new1").child(i).get().val()
    for movie_data in ll:
         for user_name in movie_data["tweet_user_name"].split(" "):
                if user_name not in dict_user_movie:
                    if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                        dict_user_movie[user_name]=[]
                        dict_user_movie[user_name].append((i,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):dumb_users.append(user_name)
                       
                else:
                    if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):dict_user_movie[user_name].append((i,max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])))
                    elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):dumb_users.append(user_name)
for key,value in dict_user_movie.items():
    temp_dict={}
    for v in value:
        if v[0] not in temp_dict:temp_dict[v[0]]=([v[1]],[movie_rating[movies2.index(v[0])]])
        else:temp_dict[v[0]][0].append(v[1])
    final_dict[key]=temp_dict
for key, value in final_dict.items():
    cc=0
    for key2,value2 in value.items():
        actual_rating=value2[1][0]
        for ratings in value2[0]:
            tweet_rating=(ratings/2)*100+50
            if(tweet_rating>actual_rating-10 and tweet_rating<actual_rating+10):cc+=1
    if(cc>=3):super_users.append(key)
print("Super_users",len(super_users))
print("Dumb_users:",len(dumb_users))


# So after getting the list of super users and dumb users we are increasing the sentiment of each super_user by 0.2 and decreasing the sentiment of dumb user by 0.05
# Now the next thing what we are doing is we are connecting all the users who have a common connection between them. For example if user A follows user B or if user B follows user A then connect them
# Also connect all the users who shared the same tweet.
# This gives us 3 networks for each movie: negative network, neutral network and positive network. From these graphs we can conclude that how one tweet from a highly connected user affect movie reviews as people tend to re-tweet the same tweet or get influenced by reading someone else's tweet in their network.

# In[16]:


print("processing tweets..........")
best_sentiment2,best_sentiment3=[],[]
dict_movies,best_sentiment,user_names,dict_neg,dict_neu,dict_pos,topics_all,topics_neg,topics_neu,topics_pos,no_features,no_topics,no_top_words={},[],[],{},{},{},{},{},{},{},1000,10,10
for i in movies2:
    sentiment_textblob,sentiment_vader,sentiment_textblob_naive=[],[],[]
    nodes_neg,nodes_neu,nodes_pos,nodes_all=[],[],[],[]
    edges_neg,edges_neu,edges_pos,edges_all=[],[],[],[]
    G_neg,G_neu,G_pos,G_all=nx.Graph(),nx.Graph(),nx.Graph(),nx.Graph()
    dict_movies[i],dict_neg[i],dict_neu[i],dict_pos[i]=[],[],[],[]
    ll=db.child("new_new1").child(i).get().val() ## getting the data from the database
    for movie_data in ll:
            dict_movies[i].append(movie_data["tweet_text_cleaned"])
            if(movie_data["sentiment_vader"]<0):dict_neg[i].append(movie_data["tweet_text_cleaned"])
            elif(movie_data["sentiment_vader"]==0):dict_neu[i].append(movie_data["tweet_text_cleaned"])
            else:dict_pos[i].append(movie_data["tweet_text_cleaned"])  
 
            ##sentiment score processing
            for user_name in movie_data["tweet_user_name"].split(" "):
                if(user_name in super_users):              
                    sentiment_textblob.append(movie_data['sentiment_textblob']+0.5)
                    sentiment_vader.append(movie_data['sentiment_vader']+0.5)
                    sentiment_textblob_naive.append(movie_data['sentiment_textblob_naive']+0.5)
                elif(user_name in dumb_users):
                    sentiment_textblob.append(movie_data['sentiment_textblob']-0.03)
                    sentiment_vader.append(movie_data['sentiment_vader']-0.03)
                    sentiment_textblob_naive.append(movie_data['sentiment_textblob_naive']-0.03)
                else:
                    sentiment_textblob.append(movie_data['sentiment_textblob'])
                    sentiment_vader.append(movie_data['sentiment_vader'])
                    sentiment_textblob_naive.append(movie_data['sentiment_textblob_naive'])
 
            ##adding edges
            if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])==0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):
                    for jj in range(ii+1,len(user_names)):edges_neu.append((user_names[ii],user_names[jj]))
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):
                    for jj in range(ii+1,len(user_names)):edges_neg.append((user_names[ii],user_names[jj]))
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):
                    for jj in range(ii+1,len(user_names)):
                        edges_pos.append((user_names[ii],user_names[jj]))
            user_names=movie_data["tweet_user_name"].split(" ")
            for ii in range(len(user_names)):
                for jj in range(ii+1,len(user_names)):edges_all.append((user_names[ii],user_names[jj]))
                        
            ##adding nodes:
            if(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])==0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):nodes_neu.append(user_names[ii])
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])<0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):nodes_neg.append(user_names[ii])
            elif(max(movie_data["sentiment_textblob"],movie_data["sentiment_vader"])>0):
                user_names=movie_data["tweet_user_name"].split(" ")
                for ii in range(len(user_names)):nodes_pos.append(user_names[ii])
            user_names=movie_data["tweet_user_name"].split(" ")
            for ii in range(len(user_names)):
                nodes_all.append(user_names[ii])
                
    G_neg.add_nodes_from(nodes_neg)
    G_neu.add_nodes_from(nodes_neu)
    G_pos.add_nodes_from(nodes_pos)
    G_all.add_nodes_from(nodes_all)
 
    G_neg.add_edges_from(edges_neg)
    G_neu.add_edges_from(edges_neu)
    G_pos.add_edges_from(edges_pos)
    G_all.add_edges_from(edges_all)
    
    ##https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
    print(">>>>>>>Degree centrality<<<<<<<<")
    print(max(nx.degree_centrality(G_neg).items(), key=operator.itemgetter(1))[0],max(nx.degree_centrality(G_neg).items(), key=operator.itemgetter(1))[1])
    print(max(nx.degree_centrality(G_neu).items(), key=operator.itemgetter(1))[0],max(nx.degree_centrality(G_neu).items(), key=operator.itemgetter(1))[1])
    print(max(nx.degree_centrality(G_pos).items(), key=operator.itemgetter(1))[0],max(nx.degree_centrality(G_pos).items(), key=operator.itemgetter(1))[1])
    print(max(nx.degree_centrality(G_all).items(), key=operator.itemgetter(1))[0],max(nx.degree_centrality(G_all).items(), key=operator.itemgetter(1))[1])
    print()
    
    print(">>>>>>>Eigen vector centrality<<<<<<<")
    print(max(nx.eigenvector_centrality(G_neg,max_iter=1000).items(), key=operator.itemgetter(1))[0],max(nx.eigenvector_centrality(G_neg,max_iter=1000).items(), key=operator.itemgetter(1))[1])
    print(max(nx.eigenvector_centrality(G_neu,max_iter=1000).items(), key=operator.itemgetter(1))[0],max(nx.eigenvector_centrality(G_neu,max_iter=1000).items(), key=operator.itemgetter(1))[1])
    print(max(nx.eigenvector_centrality(G_pos,max_iter=1000).items(), key=operator.itemgetter(1))[0],max(nx.eigenvector_centrality(G_pos,max_iter=1000).items(), key=operator.itemgetter(1))[1])
    print(max(nx.eigenvector_centrality(G_all,max_iter=1000).items(), key=operator.itemgetter(1))[0],max(nx.eigenvector_centrality(G_all,max_iter=1000).items(), key=operator.itemgetter(1))[1])
    print()
    ##https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html

 
    path1="gephi/"+i+"_neg.gexf"
    path2="gephi/"+i+"_neu.gexf"
    path3="gephi/"+i+"_pos.gexf"
    path4="gephi/"+i+"_all.gexf"
 
    nx.write_gexf(G_neg, path1)
    nx.write_gexf(G_neu, path2)
    nx.write_gexf(G_pos, path3)
    nx.write_gexf(G_all, path4)
 
#     print(i,len(node_neg),len(node_neu),len(node_pos),len(edges_neg),len(edges_neu),len(edges_pos))
    textblob=sum(sentiment_textblob)/len(sentiment_textblob)*100+50
    vader=sum(sentiment_vader)/len(sentiment_vader)*100+50
    textblob_naive=sum(sentiment_textblob_naive)/len(sentiment_textblob_naive)*100
    best_sentiment.append(float(textblob))
    best_sentiment2.append(float(vader))
    best_sentiment3.append(float(textblob_naive))
    print("Movie ", i, " sentiment - textblob: ",textblob, "%"," collected out of: ",len(sentiment_textblob)," #tweets")
    print("Movie ", i, " sentiment - textblob naive: ",textblob_naive, "%"," collected out of: ",len(sentiment_textblob_naive)," #tweets")
    print("Movie ", i, " sentiment - vader: ",vader, "%"," collected out of: ",len(sentiment_vader)," #tweets")
#     print(movie_rating)
    print("\n")  


# Histogram visualization of each movie compared with various professional movie review sites vs the ratings that we got.

# In[17]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("lexicon textblob")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment[i]
    print(movies2[i], " -> ",error)

## referencing needed


# In[18]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("Machine learning based Rating textblob")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment3, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment3[i]
    print(movies2[i], " -> ",error)

## referencing needed


# In[19]:


##https://python-graph-gallery.com/11-grouped-barplot/(reference to create a double bar chart)
print("lexicon based rating")
barWidth = 0.28
r1 = np.arange(len(movie_rating))
r2 = [x + barWidth for x in r1]
plt.bar(r1, movie_imdb_rating, color='red', width=barWidth, edgecolor='white', label='IMDB Rating')
plt.bar(r2, best_sentiment2, color='black', width=barWidth, edgecolor='white', label='Our Rating')
plt.xlabel('movie name', fontweight='bold')
plt.ylabel('Rating', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(movie_rating))], movies2)
# plt.legend()
plt.show()

print("error rate for each movie")
for i in range(len(movie_rating)):
    error=movie_imdb_rating[i]-best_sentiment2[i]
    print(movies2[i], " -> ",error)


# The Next thing what we are doing is topic modelling. the topic modelling is carried on 4 different datasets for each movie
# 1. we are gathering all the positive tweets of one particular movies and performing lda topic modelling on it with 10 topics and 10 top words
# 2. similarly we are doing the same on all the neutral and negtive tweets for each movie.
# 3. we are using the sklearns topic modelling algorithm to carry out our lda topic modelling
# Next what we are doing is getting all the words from positive, negative, and neutral sentiment scores and constructing a venn diagram for each movie which states if there are any common topic words for people who gave bad, good and neutral movie ratings. from this we can get insights about if there is a debated topic between all the set of users. This can help us filter out further more data using these topics about a particular movie.

# In[21]:


##https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847ddef topic_modelling(documents):
def topic_modelling(documents):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    return lda, tf_feature_names, no_top_words
##https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d
for i in movies2:
    print("****** ",i," *******")
    topics_all[i]=[]
    topics_neg[i]=[]
    topics_neu[i]=[]
    topics_pos[i]=[]
    
    print(">>>>>>>>> Topic Modelling on all tweets <<<<<<<<<<<")
    documents=dict_movies[i]
    lda,tf_feature_names, no_top_words=topic_modelling(documents)
    topics_all[i]=display_topics(lda, tf_feature_names, no_top_words)
    
    print(">>>>>>>>> Topic Modelling on negative tweets tweets <<<<<<<<<<<")
    documents=dict_neg[i]
    lda,tf_feature_names, no_top_words=topic_modelling(documents)
    topics_neg[i]=display_topics(lda, tf_feature_names, no_top_words)
    
    print(">>>>>>>>> Topic Modelling on neutral tweets <<<<<<<<<<<")
    documents=dict_neu[i]
    lda,tf_feature_names, no_top_words=topic_modelling(documents)
    topics_neu[i]=display_topics(lda, tf_feature_names, no_top_words)
    
    print(">>>>>>>>> Topic Modelling on positive tweets <<<<<<<<<<<")
    documents=dict_pos[i]
    lda,tf_feature_names, no_top_words=topic_modelling(documents)
    topics_pos[i]=display_topics(lda, tf_feature_names, no_top_words)


# venn diagram construction

# In[22]:


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
negg,poss,neuu=[],[],[]
for i in movies2:
    neg=[]
    neu=[]
    pos=[]
    for j in topics_neg[i]:
        words=j.split(" ")
        for k in words:
            neg.append(k)
    for j in topics_neu[i]:
        words=j.split(" ")
        for k in words:
            neu.append(k)
    for j in topics_pos[i]:
        words=j.split(" ")
        for k in words:
            pos.append(k)
    negg.append(neg)
    poss.append(pos)
    neuu.append(neu)


# In[23]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[0]).intersection(set(poss[0])).intersection(set(neuu[0])))
print()
print(set(negg[0]))
print()
print(set(poss[0]))
venn3([set(negg[0]), set(neuu[0]), set(poss[0])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[0])


# In[24]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[1]).intersection(set(poss[1])).intersection(set(neuu[1])))
print()
print(set(negg[1]))
print()
print(set(poss[1]))
venn3([set(negg[1]), set(neuu[1]), set(poss[1])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[1])


# In[25]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[2]).intersection(set(poss[2])).intersection(set(neuu[2])))
print()
print(set(negg[2]))
print()
print(set(poss[2]))
venn3([set(negg[2]), set(neuu[2]), set(poss[2])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[2])


# In[26]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[3]).intersection(set(poss[3])).intersection(set(neuu[3])))
print()
print(set(negg[3]))
print()
print(set(poss[3]))
venn3([set(negg[3]), set(neuu[3]), set(poss[3])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[3])


# In[27]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[4]).intersection(set(poss[4])).intersection(set(neuu[4])))
print()
print(set(negg[4]))
print()
print(set(poss[4]))
venn3([set(negg[4]), set(neuu[4]), set(poss[4])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[4])


# In[28]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[5]).intersection(set(poss[5])).intersection(set(neuu[5])))
print()
print(set(negg[5]))
print()
print(set(poss[5]))
venn3([set(negg[5]), set(neuu[5]), set(poss[5])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[5])


# In[29]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[6]).intersection(set(poss[6])).intersection(set(neuu[6])))
print()
print(set(negg[6]))
print()
print(set(poss[6]))
venn3([set(negg[6]), set(neuu[6]), set(poss[6])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[6])


# In[30]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[7]).intersection(set(poss[7])).intersection(set(neuu[7])))
print()
print(set(negg[7]))
print()
print(set(poss[7]))
venn3([set(negg[7]), set(neuu[7]), set(poss[7])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[7])


# In[31]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[8]).intersection(set(poss[8])).intersection(set(neuu[8])))
print()
print(set(negg[8]))
print()
print(set(poss[8]))
venn3([set(negg[8]), set(neuu[8]), set(poss[8])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[8])


# In[32]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[9]).intersection(set(poss[9])).intersection(set(neuu[9])))
print()
print(set(negg[9]))
print()
print(set(poss[9]))
venn3([set(negg[9]), set(neuu[9]), set(poss[9])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[9])


# In[33]:


## venn diaragm syntax referenced from https://www.badgrammargoodsyntax.com/compbio/2017/10/29/compbio-012-making-venn-diagrams-the-right-way-using-python
print(set(negg[10]).intersection(set(poss[10])).intersection(set(neuu[10])))
print()
print(set(negg[10]))
print()
print(set(poss[10]))
venn3([set(negg[10]), set(neuu[10]), set(poss[10])], set_labels = ('Negative','Neutral','Positive'))
plt.title(movies[10])


# After considering the topics of infinity war and removing those topics the ratings are:

# In[34]:


best_sentiment,best_sentiment2,best_sentiment3=[],[],[]
sentiment_vader,sentiment_textblob,sentiment_textblob_naive=[],[],[]
ll=db.child("new_new1").child("infinitywar").get().val()
for movie_data in ll:
    for ii in movie_data["tweet_user_name"].split(" "):
        tweet_text=movie_data["tweet_text_original"]
        tweet_text=tweet_text.lower()
        if( ("avengersendgame" in tweet_text)
           or ("endgame" in tweet_text)):break
        else:
            if(ii in super_users):
                sentiment_vader.append(movie_data["sentiment_vader"]+0.9) ##bias doubled as the number of tweets halved
                sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]+0.9)
            elif(ii in dumb_users):
                sentiment_vader.append(movie_data["sentiment_vader"]-0.03)
                sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]-0.03)
            else:
                sentiment_vader.append(movie_data["sentiment_vader"])
                sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"])
vader=sum(sentiment_vader)/len(sentiment_vader)*100+50
textblob_naive=sum(sentiment_textblob_naive)/len(sentiment_textblob_naive)*100

print("Machine Learning",textblob_naive,"||",len(sentiment_textblob_naive))
print("Lexicon based", vader,"||",len(sentiment_vader))


# We tried the same thing for alita but didnt work as effectively as the accuracy is still way too high

# In[35]:


dictt={}
best_sentiment,best_sentiment2,best_sentiment3=[],[],[]
sentiment_vader,sentiment_textblob,sentiment_textblob_naive=[],[],[]
ll=db.child("new_new1").child("alita").get().val()
for movie_data in ll:
        user_names=movie_data["tweet_user_name"].split(" ")
        for ii in user_names:
            tweet_text=movie_data["tweet_text_original"]
            tweet_text=tweet_text.lower()
            if( ("alitachallenge" in tweet_text) ):pass
            else:
                if(ii in super_users ):
                    if(ii not in dictt):dictt[ii]=1
                    else:dictt[ii]=dictt[ii]+1
                    sentiment_vader.append(movie_data["sentiment_vader"]+0.5)
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]+0.5)
                elif(ii in dumb_users):
                    sentiment_vader.append(movie_data["sentiment_vader"]-0.03)
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]-0.03)
                else:
                    sentiment_vader.append(movie_data["sentiment_vader"])
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"])
vader=sum(sentiment_vader)/len(sentiment_vader)*100+50
textblob_naive=sum(sentiment_textblob_naive)/len(sentiment_textblob_naive)*100

print("Machine Learning",textblob_naive,"||",len(sentiment_textblob_naive))
print("Lexicon based",vader,"||",len(sentiment_vader))


# However after network analysis we removed some of the nodes from the network and found this

# In[36]:


sorted_x = sorted(dictt.items(), key=lambda kv: kv[1], reverse=True)
not_to_include=[]
for i in range(0,23):
    not_to_include.append(sorted_x[i][0])
print("list of nodes that were removed")
print(not_to_include)

best_sentiment,best_sentiment2,best_sentiment3=[],[],[]
sentiment_vader,sentiment_textblob,sentiment_textblob_naive=[],[],[]
ll=db.child("new_new1").child("alita").get().val()
for movie_data in ll:
        user_names=movie_data["tweet_user_name"].split(" ")    
        for ii in user_names:
            flag=0
            for xx in not_to_include:
                if(xx in user_names):flag=1
            if(flag==1):break
            tweet_text=movie_data["tweet_text_original"]
            tweet_text=tweet_text.lower()
            if( ("alitachallenge" in tweet_text) ):pass
            else:
                if(ii in super_users ):
        
                    if(ii in dictt):dictt[ii]=1
                    else:dictt[ii]+1
                    sentiment_vader.append(movie_data["sentiment_vader"]+0.5)
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]+0.5)
                elif(ii in dumb_users):
                    sentiment_vader.append(movie_data["sentiment_vader"]-0.03)
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"]-0.03)
                else:
                    sentiment_vader.append(movie_data["sentiment_vader"])
                    sentiment_textblob_naive.append(movie_data["sentiment_textblob_naive"])
vader=sum(sentiment_vader)/len(sentiment_vader)*100+50
textblob_naive=sum(sentiment_textblob_naive)/len(sentiment_textblob_naive)*100

print("Machine Learning",textblob_naive,"||",len(sentiment_textblob_naive))
print("Lexicon based",vader,"||",len(sentiment_vader))


# Reference: <br/>
# [1] https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
