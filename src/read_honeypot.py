from pymongo import MongoClient
import re
import string
import numpy as np
from Lib import statistics as stat
import datetime as dt
from urllib.request import urlretrieve
import zipfile
import os

#This file reads data from honeypot dataset, process and write into MongoDB -> old_data

num_decimal = 4

#Read basic info for each user, write into the collection of MongoDB
def read_info(filename, collection):
    with open(filename, 'r') as f:
        for line in f:
            tokens = [r for r in re.split("[\t\n]", line) if r != '']
            #Parse info to numbers, take out the dates
            info = [tokens[0]] + [int(r) for r in tokens[3:]]
            
            collection.update({'id': info[0]}, {'$set': {'friends_count': info[1], 
                                                         'followers_count': info[2], 
                                                         'statuses_count': info[3], 
                                                         'len_screen_name': info[4], 
                                                         'len_profile': info[5]}}, 
                                               True, False)
    #Create index with id for the collection
    collection.create_index([('id', 1)])


#Read data of a series of # followings for each user, calculate stdev and change rate and write into collection
def read_followings(filename, collection):
    with open(filename, 'r') as f:
        for line in f:
            tokens = [r for r in re.split("[\D]", line) if r != ""]
            user_id = tokens[0]
            followings = [int(r) for r in tokens[1:]]
            #Calculate standard deviation of the data
            sd = round(stat.pstdev(followings), num_decimal)
            #Calcalate standard deviation of differences of the data
            sdd = round(stat.pstdev(list(np.array(followings[1:]) - np.array(followings[:-1]))), num_decimal)
            #Calculate lag one autocorrelation of the data
            avg = np.mean(followings)
            numerator = sum((np.array(followings[1:]) - avg) * (np.array(followings[:-1]) - avg))
            denominator = sum((np.array(followings) - avg) ** 2)
            lac = round(numerator/denominator, num_decimal) if denominator != 0 else 0
            
            collection.update({'id': user_id}, {'$set': 
                                                 {'friends': 
                                                   {'sd': sd,
                                                    'sdd': sdd,
                                                    'lac': lac}}},
                                               True, False)


#Read tweets posted by each user, extract useful info and write into collection
def read_tweets(filename, collection):
    with open(filename, encoding = 'utf-8', mode = 'r') as f:
        for line in f:
            tokens = [r for r in re.split("[\t\n]", line) if r != ""]
            user_id, tweet_id, tweet, created_at = tokens
            
            entities = [r for r in re.findall('https://\S+|http://\S+|@\S+|#\S+', tweet) if r != '']
            urls = [r for r in entities if 'http' in r]
            mentions = [r for r in entities if '@' in r]
            hashtags = [r for r in entities if '#' in r]
            #See if the tweet ends with a punctuation or hashtag or url, which might indicate that it's auto-generated
            tweet_tokens = [r for r in re.split('(http)s://\S+|(http)://\S+|(@)\S+|(#)\S+|\s+|\W+', tweet)]
            end_with_phu = tweet_tokens[-1] in ['http', '#'] or tweet[-1] in string.punctuation
            #Get the day of the week it is posted
            post_date = re.split("[-\s]", created_at)
            post_date = dt.date(int(post_date[0]), int(post_date[1]), int(post_date[2]))
            weekday_post = post_date.weekday()
            
            collection.update({'id': user_id}, {'$addToSet': 
                                                 {'timeline': 
                                                   {'id': tweet_id,
                                                    'text': tweet,
                                                    'created_at': created_at,
                                                    'urls': urls,
                                                    'mentions': mentions,
                                                    'hashtags': hashtags,
                                                    'end_with_phu': end_with_phu,
                                                    'weekday': weekday_post}}},
                                               True, False)


#Deleting ambiguous users who are duplicated in both polluters and legitimate users (44 found), return user ids
#Both inputs should be collections
def del_dup(bots, humans):
    bot_ids = [r['id'] for r in bots.find({}, {'id': 1})]
    human_ids = [r['id'] for r in humans.find({}, {'id': 1})]
    duplicates = [r for r in bot_ids if r in human_ids]
    for id in duplicates:
        bots.delete_one({'id': id})
        humans.delete_one({'id': id})
    return [len(duplicates), duplicates]


def main():
    if not os.path.exists('social_honeypot_icwsm_2011'):
        if not os.path.exists('social_honeypot_icwsm_2011.zip'):
            print("Downloading honeypot dataset")
            urlretrieve('http://infolab.tamu.edu/static/users/kyumin/social_honeypot_icwsm_2011.zip', 'social_honeypot_icwsm_2011.zip')
        zip = zipfile.ZipFile('social_honeypot_icwsm_2011.zip')
        zip.extractall(path = 'social_honeypot_icwsm_2011')
        zip.close()
        print("Dataset ready")
    
    mClient = MongoClient()
    bots = mClient['old_data']['bots']
    humans = mClient['old_data']['humans']
    
    read_info('social_honeypot_icwsm_2011' + os.path.sep + 'content_polluters.txt', bots)
    read_info('social_honeypot_icwsm_2011' + os.path.sep + 'legitimate_users.txt', humans)
    print('Basic info read and written to database')
    
    num_dup, dup = del_dup(bots, humans)
    print('%d duplicated users found and deleted' % num_dup)
    print(dup.__str__())
    
    read_followings('social_honeypot_icwsm_2011' + os.path.sep + 'content_polluters_followings.txt', bots)
    read_followings('social_honeypot_icwsm_2011' + os.path.sep + 'legitimate_users_followings.txt', humans)
    print('Numbers of friends read, stdev and change rate written to database')
    
    read_tweets('social_honeypot_icwsm_2011' + os.path.sep + 'content_polluters_tweets.txt', bots)
    read_tweets('social_honeypot_icwsm_2011' + os.path.sep + 'legitimate_users_tweets.txt', humans)
    
    mClient.close()


if __name__ == '__main__':
    main()

