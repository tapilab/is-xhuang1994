from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
import configparser
import numpy as np
import string
import math
import os
import re
import time

#This file reads user ids from MongoDB -> old_data, collects the latest information of active users, and store it in Mongodb -> new_data


#Read user ids from collection
def read_users(collection):
    all_users = [r['id'] for r in collection.find({}, {'id': 1})]
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
                users = api.request("users/lookup", {'user_id': all_users[i*100:i*100+100]})
                users = [r for r in users]
                for user in users:
                    this_user = {'id': user['id_str'], 
                                 'name': user['name'], 
                                 'screen_name': user['screen_name'], 
                                 'followers_count': user['followers_count'], 
                                 'friends_count': user['friends_count'], 
                                 'protected': user['protected'], 
                                 'description': user['description'], 
                                 'lang': user['lang'], 
                                 'statuses_count': user['statuses_count'], 
                                 'profile_img': user['profile_image_url'], 
                                 'goe_enabled': user['geo_enabled'], 
                                 'active': True}
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

#Get timeline for a user, write info into collection
def get_timeline(user_id, api, collection):
    weekdays = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    while True:
        try:
            this_user = {'id': user_id, 'timeline': []}
            timeline = api.request("statuses/user_timeline", {'user_id': user_id, 'count': 200})
            timeline = [r for r in timeline]
            for tweet in timeline:
                this_tweet = {'id': tweet['id_str'], 
                              'created_at': tweet['created_at'], 
                              'urls': [r['expanded_url'] for r in tweet['entities']['urls']], 
                              'hashtags': [r['text'] for r in tweet['entities']['hashtags']], 
                              'mentions': [r['id_str'] for r in tweet['entities']['user_mentions']], 
                              'text': tweet['text'], 
                              'source': tweet['source'], 
                              'rt_count': tweet['retweet_count'], 
                              'is_rt': True if tweet['text'][0:2] == "RT" else False, 
                              'coordinated': True if tweet['coordinates'] != None else False, 
                              'is_reply': False if tweet['in_reply_to_user_id_str'] == None else True
                              'weekday': weekdays[tweet['created_at'][0:3]]}
                              
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
                print('Request failed: id %s is probably protected' % user_id)
                break
            else:
                print("Unexpected error raised with id =", user_id)
                s = tre.__str__()
                print(s, "\n")
                continue
        except Exception as e:
            print("Unexpected error raised with id =", user_id)
            s = e.__str__()
            print(s, "\n")
            continue
    collection.update({'id': bot['id']}, {'$set': {'timeline': bt['timeline']}}, False, False)


#Calculate Jaccard Similarity between texts
def calc_sim(text1, text2):
    t1_1 = [r for r in re.split('\s+|\W+|(http)://\S+|(http)s://\S+|(#)\S+|(@)\S+', text1) if r != '' and r != None]
    t2_1 = [r for r in re.split('\s+|\W+|(http)://\S+|(http)s://\S+|(#)\S+|(@)\S+', text2) if r != '' and r != None]
    
    conj = set([r for r in t1_1 if r in t2_1])
    disj = set(t1_1 + t2_1)
    sim_1 = len(conj) / len(disj) if len(disj) != 0 else 0
    
    t1_2, t2_2 = [], []
    for i in list(range(len(t1_1)-1)):
        t1_2.append((t1_1[i], t1_1[i+1]))
    for i in list(range(len(t2_1)-1)):
        t2_2.append((t2_1[i], t2_1[i+1]))
    sim_2 = len(set([r for r in t1_2 if r in t2_2])) / len(set(t1_2 + t2_2)) if len(set(t1_2 + t2_2)) > 0 else 0
    
    t1_3, t2_3 = [], []
    for i in list(range(len(t1_1)-2)):
        t1_3.append((t1_1[i], t1_1[i+1], t1_1[i+2]))
    for i in list(range(len(t2_1)-2)):
        t2_3.append((t2_1[i], t2_1[i+1], t2_1[i+2]))
    sim_3 = len(set([r for r in t1_3 if r in t2_3])) / len(set(t1_3 + t2_3)) if len(set(t1_3 + t2_3)) > 0 else 0
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
    new_data = mClient['new_data']
    old_data = mClient['old_data']
    print("Twitter API connection and Mongodb connection ready")
    
    
    for collection_name in ['bots', 'humans']:
        users_old = old_data[collection_name]
        all_users = read_users(users_old)
        active_users_new = get_users(all_users, api)
        inactive_users = [r for r in all_users if r not in [u['id'] for u in active_users_new]]
        print("Active %s fetched with Twitter API" % collection_name)
        
        users_new = new_data[collection_name]
        for user in active_users_new:
            users_new.update({'id': user['id']}, {'$set': user})
        for user_id in inactive_bots:
            users_new.update({'id': user_id}, {'$set': {'active': False}})
        print("Basic information of %s written to database" % collection_name)
    
    
    for collection_name in ['bots', 'humans', 'new_users']:
        while True:
            user = new_data[collection_name].find_one({'protected': False, 'timeline': {'$exists': 0}}, {'id': 1})
            if user == None:
                break
            get_timeline(user['id'], api, new_data[collection_name])
        print("Info about timeline of each %s written to database" % collection_name)
    
    
    for collection_name in ['bots', 'humans', 'new_users']:
        while True:
            user = db['new_users'].find_one({'tweets_sim': {'$exists': 0}, 'timeline': {'$exists': 1}}, {'id': 1, 'timeline': 1})
            if user == None:
                break
            tweets = [r['text'] for r in user['timeline'] if r['is_reply'] == False and r['is_rt'] == False]
            if len(tweets) < 2:
                sim = [0, 0, 0]
            else:
                sim = np.array([0, 0, 0])
                for i in list(range(len(tweets)-1)):
                    for j in list(range(i+1, len(tweets))):
                        sim = sim + np.array(calc_sim(tweets[i], tweets[j]))
                sim = sim / ((len(tweets)+1)*n/2)
            db['new_users'].update_one({'id': user['id']}, {'$set': {'tweets_sim': list(sim)}})
    
    mClient.close()



if __name__ == '__main__':
    main()