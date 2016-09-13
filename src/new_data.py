from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
from collections import Counter
from sklearn.metrics import jaccard_similarity_score
import configparser
import string
import math
import os
import re
import time

#This file reads user ids from honeypot dataset, collects all the latest information of active users, and store it in Mongodb


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
                                 'friends_count': user['friends_count'], 'protected': user['protected'], 'description': user['description'], \
                                 'statuses_count': user['statuses_count'], 'profile_img': user['profile_image_url'], 'goe_enabled': user['geo_enabled']}
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
def get_timeline(unprotected_user, api):
    id = unprotected_user['id']
    weekdays = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    for e in [1]:
        while True:
            try:
                this_user = {'id': id, 'timeline': []}
                timeline = api.request("statuses/user_timeline", {'user_id': id, 'count': 200})
                timeline = [r for r in timeline]
                for tweet in timeline:
                    this_tweet = {'id': tweet['id_str'], 'created_at': tweet['created_at'], 'urls': [r['expanded_url'] for r in tweet['entities']['urls']], \
                                  'hashtags': [r['text'] for r in tweet['entities']['hashtags']], 'mentions': [r['id_str'] for r in tweet['entities']['user_mentions']], \
                                  'text': tweet['text'], 'source': tweet['source'], 'rt_count': tweet['retweet_count'], 'is_rt': True if tweet['text'][0:2] == "RT" else False, \
                                  'coordinated': True if tweet['coordinates'] != None else False, 'is_reply': False if tweet['in_reply_to_user_id_str'] == None else True}
                    weekday = weekdays[tweet['created_at'][0:3]]
                    this_tweet['weekday'] = weekday
                    end_of_entities = [r['indices'][1] for r in tweet['entities']['urls']] + [[r['indices'][1] for r in tweet['entities']['hashtags']]]
                    if len(tweet['text']) in end_of_entities or tweet['text'][-1] in string.punctuation:
                        this_tweet['end_with_phu'] = True
                    else:
                        this_tweet['end_with_phu'] = False
                    this_user['timeline'].append(this_tweet)
                break
            except TwitterError.TwitterRequestError as tre:
                if tre.status_code == 429:
                    print("z z z ...")
                    time.sleep(180)
                    continue
                elif tre.status_code == 401:
                    print('id %s is probably protected' % id)
                    break
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
    return this_user

#calculate Jaccard similarity between texts
def calc_sim(text1, text2):
    t1_1 = [r for r in re.split('\s+|\W+|http:\\\\|https:\\\\|#', text1) if r != '']
    t2_1 = [r for r in re.split('\s+|\W+|http:\\\\|https:\\\\|#', text2) if r != '']
    conj = set([r for r in t1_1 if r in t2_1])
    disj = set(t1_1 + t2_1)
    sim_1 = len(conj) / len(disj)
    
    t1_2, t2_2 = [], []
    for i in list(range(len(t1_1)-1)):
        t1_2.append((t1_1[i], t1_1[i+1]))
    for i in list(range(len(t2_1)-1)):
        t2_2.append((t2_1[i], t2_1[i+1]))
    sim_2 = len(set([r for r in t1_2 if r in t2_2])) / len(set(t1_2 + t2_2))
    
    t1_3, t2_3 = [], []
    for i in list(range(len(t1_1)-2)):
        t1_3.append((t1_1[i], t1_1[i+1], t1_1[i+2]))
    for i in list(range(len(t2_1)-2)):
        t2_3.append((t2_1[i], t2_1[i+1], t2_1[i+2]))
    sim_3 = len(set([r for r in t1_3 if r in t2_3])) / len(set(t1_3 + t2_3))
    return [sim_1, sim_2, sim_3]
    

def main():

    config = configparser.ConfigParser()
    config.read("twitter.cfg")
    consumer_key = config.get('twitter', 'consumer_key')
    consumer_secret = config.get('twitter', 'consumer_secret')
    access_token = config.get('twitter', 'access_token')
    access_token_secret = config.get('twitter', 'access_token_secret')
    
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
    """
    unprotected_bots = sorted([r for r in db['a_bots'].find({'protected': False})], key = lambda x: x['id'])
    unprotected_humans = sorted([r for r in db['a_humans'].find({'protected': False})], key = lambda x: x['id'])
    
    for d in list(range(98, 108):
        bots_timeline = get_timeline(unprotected_bots, api, d, d+1)
        humans_timeline = get_timeline(unprotected_humans, api, d, d+1)
        print("Timeline fetched for each unprotected user")
    
        for i in list(range(len(bots_timeline))):
            db['a_bots'].update_one({'id': bots_timeline[i]['id']}, {'$set': {'timeline': bots_timeline[i]['timeline']}}, False)
        for i in list(range(len(humans_timeline))):
            db['a_humans'].update_one({'id': humans_timeline[i]['id']}, {'$set': {'timeline': humans_timeline[i]['timeline']}}, False)
        
        print("Timeline of each unprotected user written to database")
    """
    """
    while True:
        bot = db['a_bots'].find_one({'protected': False, 'timeline': {'$exists': 0}})
        #human = db['a_humans'].find_one({'protected': False, 'timeline': {'$exists': 0}})
        bt = get_timeline(bot, api)
        #ht = get_timeline(human, api)
        db['a_bots'].update_one({'id': bt['id']}, {'$set': {'timeline': bt['timeline']}}, False)
        #db['a_humans'].update_one({'id': ht['id']}, {'$set': {'timeline': ht['timeline']}}, False)
    """
    while True:
        user = db['bots'].find_one({'tweets_sim': {'$exists': 0}}, {'id': 1, 'timeline': 1})
        if len(user['timeline']) == 0:
            sim = [0, 0, 0]
        else:
            tweets = [r['text'] for r in user['timeline']]
            for i in list(range(len(tweets)-1)):
                for j in list(range(i+1, len(tweets))):
                    calc_sim(tweets[i], tweets[j])
    
    mClient.close()
    
    
    
    
if __name__ == '__main__':
    main()