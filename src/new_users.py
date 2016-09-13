from twitter import *
import oauth2
from TwitterAPI import TwitterAPI, TwitterError
import json
from pymongo import MongoClient
import configparser
import math
import os
import re
import time

#This file uses Streaming API to find new users on Twitter


def main():
    config = configparser.ConfigParser()
    config.read("twitter.cfg")
    consumer_key1 = config.get('twitter', 'consumer_key')
    consumer_secret1 = config.get('twitter', 'consumer_secret')
    access_token1 = config.get('twitter', 'access_token')
    access_token_secret1 = config.get('twitter', 'access_token_secret')
    
    oauth = OAuth(token = access_token1, token_secret = access_token_secret1, consumer_secret = consumer_secret1, consumer_key = consumer_key1)
    
    
    t = Twitter(auth = oauth)
    timeline = t.statuses.home_timeline()
    print(timeline)
    
    
    stream = TwitterStream(auth=oauth)
    iterator = stream.statuses.sample()
    for tweet in iterator:
        #tweet = list(tweet)
        print(tweet)
        break
    print(tweet)
    
    
    api = TwitterAPI(consumer_key1, consumer_secret1, access_token1, access_token_secret1)
    timeline = api.request("statuses/sample")
    #timeline = api.request("statuses/user_timeline", {'id': '10836'})
    timeline = list(timeline)
    print(timeline)
    
    

if __name__ == '__main__':
    main()