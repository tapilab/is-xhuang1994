from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
import configparser
import math
import os
import re
import time

#This file reads user ids from honeypot dataset, collects all the latest information of active users, and store it in Mongodb


config = configparser.ConfigParser()
config.read("twitter.cfg")
consumer_key = 'VV6muIO8HtYyQHy7BubcHFHgk'
consumer_secret = 'irUrQmS14T12aZRoUMZPjuyU95yzb6p2Oo4PVIC3hjjS9VFYvt'
access_token = '2811894554-t6vBpEcXT8lpPDZm034iBcn4uY9jBIiAiPJgyEV'
access_token_secret = 'KO0trs36hOcIKFb1kpIl8vGuZ9V2uNbFuce0575JUnrZ0'

api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
timeline = api.request("users/lookup", {'user_id': '20015831'})
timeline = [r for r in timeline]