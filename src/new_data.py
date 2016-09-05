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
consumer_key = config.get('twitter', 'consumer_key')
consumer_secret = config.get('twitter', 'consumer_secret')
access_token = config.get('twitter', 'access_token')
access_token_secret = config.get('twitter', 'access_token_secret')


#Read users from honeypot dataset
def read_users(filename):
    all_users = []
    with open(filename, 'r') as f:
        for line in f:
            all_users.append(re.split("[\t\n]", line)[0])
    return all_users

#Get all active users with user id, name, screen name, followers count, friends count, protection state, description, statuses count,
#which reside in the returned "user" object
def get_users(all_users, api):
    active_users = []
    #users/lookup request can return up to 100 active users per request
    for i in range(math.ceil(len(all_users)/100)):
        if i % 100 == 0:
            print("Get Users: Iterated %d times" % i)
        while True:
            try:
                users = api.request("users/lookup", {'user_id': all_users[i*100:i*100+99]})
                users = [r for r in users]
                for user in users:
                    this_user = {'id': user['id_str'], 'name': user['name'], 'screen_name': user['screen_name'], 'followers_count': user['followers_count'], \
                                 'friends_count': user['friends_count'], 'protected': user['protected'], 'description': user['description'], 'statuses_count': user['statuses_count']}
                    active_users.append(this_user)
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
    return active_users

#Get timeline for each user in the range. This will take about 40 hours in total, so I take a slice of 300 elements of the list each time, which takes about 25 minutes in average.
#There is 31702 active users in the database, so the number "end" is up to 106
def get_timeline(unprotected_users, api, start, end):
    user_ids = [r['id'] for r in unprotected_users][start*300:end*300]
    user_timeline = []
    i = 0
    for id in user_ids:
        if i % 150 == 0:
            print("Get Timeline: Iterated %d times" % i)
        i += 1
        while True:
            try:
                this_user = {'id': id, 'timeline': []}
                timeline = api.request("statuses/user_timeline", {'user_id': id, 'count': 200})
                timeline = [r for r in timeline]
                for tweet in timeline:
                    this_tweet = {'id': tweet['id_str'], 'created_at': tweet['created_at'], 'urls': [r['expanded_url'] for r in tweet['entities']['urls']], \
                                  'hashtags': tweet['entities']['hashtags'], 'user_mentions': tweet['entities']['user_mentions'], 'text': tweet['text'], \
                                  'is_rt': False if tweet['in_reply_to_user_id_str'] == None else True}
                    this_user['timeline'].append(this_tweet)
                user_timeline.append(this_user)
                break
            except TwitterError.TwitterRequestError as tre:
                if tre.status_code == 429:
                    print("z z z ...")
                    time.sleep(60)
                    continue
                else:
                    print("Unexpected error raised with id =", id)
                    s = tre.__str__()
                    print(s, "\n")
                    continue
            except Exception as e:
                print("Unexpected error raised with id =", id)
                s = e.__str__()
                print(s, "\n")
                continue
    return user_timeline

def main():
    api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    mClient = MongoClient()
    db = mClient['new_data']
    print("Twitter API connection and Mongodb connection ready")
    
    """
    all_bots = read_users("old_data" + os.path.sep + "bots.txt")
    print("Data read from old_data" + os.path.sep + "bots.txt")
    active_bots = get_users(all_bots, api)
    print("Active bots fetched with Twitter API")
    inactive_bots = [{'id': r} for r in all_bots if r not in [u['id'] for u in active_bots]]
    a_bots, i_bots = db['a_bots'], db['i_bots']
    a_bots.insert_many(active_bots)
    i_bots.insert_many(inactive_bots)
    print("Basic information of each active bot written to database")
    
    all_humans = read_users("old_data" + os.path.sep + "humans.txt")
    print("Data read from old_data" + os.path.sep + "humans.txt")
    active_humans = get_users(all_humans, api)
    print("Active humans fetched with Twitter API")
    inactive_humans = [{'id': r} for r in all_humans if r not in [u['id'] for u in active_humans]]
    a_humans, i_humans = db['a_humans'], db['i_humans']
    a_humans.insert_many(active_humans)
    i_humans.insert_many(inactive_humans)
    print("Basic information of each active human written to database")
    """
    
    unprotected_bots = [r for r in db['a_bots'].find({'protected': False})]
    unprotected_humans = [r for r in db['a_humans'].find({'protected': False})]
    bots_timeline = get_timeline(unprotected_bots, api, 6, 7)
    humans_timeline = get_timeline(unprotected_humans, api, 6, 7)
    print("Timeline fetched for each unprotected user")

    b_timeline, h_timeline = db['a_bots_timeline'], db['a_humans_timeline']
    b_timeline.insert_many(bots_timeline)
    h_timeline.insert_many(humans_timeline)
    print("Timeline of each unprotected user written to database")
    
    mClient.close()
    
    
    
    
if __name__ == '__main__':
    main()