from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
import configparser
import math
import os
import re
import time

#This file reads bot-users from Mongodb, analyze their followers and friends, and find new potential bots on Twitter


#Read users from mongodb and return users based on some threshold
def read_users(mongodb, threshold):
    db = mongodb['new_data']
    unprotected_bots = [r['id'] for r in db['a_bots'].find({'protected': False})]
    pass


#Get follower&friend lists from Twitter api
def get_relation(u_bots, twitterapi):
    #Every user id is mapped with 2 lists -- first is friends list, second is followers list
    relation = {r:[[], []] for r in u_bots}
    for id in u_bots:
        while True:
            try:
                friends = [r['ids'] for r in api.request("friends/ids", {'user_id': id})][0]
                followers = [r['ids'] for r in api.request("followers/ids", {'user_id': id})][0]
                relation[id] = [friends, followers]
                break
            #Error code 429 means the requests have reached the rate limit
            except TwitterError.TwitterRequestError as tre:
                if tre.status_code == 429:
                    print("z z z ...")
                    time.sleep(60)
                    continue
                else:
                    print("Unexpected error raised")
                    s = tre.__str__()
                    print(s, "\n")
                    continue
            except Exception as e:
                print("Unexpected error raised")
                s = e.__str__()
                print(s, "\n")
                continue
    return relation
    
    
