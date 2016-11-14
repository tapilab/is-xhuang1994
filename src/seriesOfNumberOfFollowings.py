from new_data import read_users, get_users
from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
import configparser
import numpy as np
import string
import math
import time
import datetime


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
    print("Twitter API connection and Mongodb connection ready")
    '''
    while True:
        t = time.localtime()
        if t[4] < 2:
            time_str = "%d\\%d %d:%d" % (t[1], t[2], t[3], t[4])
            print("\nstarting collection process @ %s" % time_str)
            for collection_name in ['bots', 'humans', 'new_users']:
                all_users = read_users(new_data[collection_name])
                relations_count = [[r['id'], r['followers_count'], r['friends_count']] for r in get_users(all_users, api)]
                t = time.localtime()
                for r in relations_count:
                    new_data[collection_name].update({'id': r[0]}, {'$push': {'followers_count_record_new': [r[1], t[1], t[2], t[3], t[4]], 'friends_count_record_new': [r[2], t[1], t[2], t[3], t[4]]}}, False, False)
                
                time_str = "%d\\%d %d:%d" % (t[1], t[2], t[3], t[4])
                print('new_data.%s successfully updated @ %s' % (collection_name, time_str))
        else:
            time.sleep(110)
            continue
    '''
    new_users_record = new_data['new_users'].find({'friends_count_record_new': {'$exists': 1}}, {'id': 1, 'friends_count_record_new': 1, 'followers_count_record_new': 1})
    new_users_record = [r for r in new_users_record]
    
    for u in new_users_record:
        friends_record = u['friends_count_record_new']
        followers_record = u['followers_count_record_new']
        '''
        friends_change, followers_change = [], []
        for i in range(len(friends_record)-1):
            n, m, d, h, mn = friends_record[i]
            n1, m1, d1, h1, mn1 = friends_record[i+1]
            n_diff = n1-n
            t_diff = datetime.datetime(2016, m1, d1, h1, mn1) - datetime.datetime(2016, m, d, h, mn)
            t_diff = t_diff.days*86400 + t_diff.seconds
            friends_change.append(n_diff/t_diff)
        for i in range(len(followers_record)-1):
            n, m, d, h, mn = followers_record[i]
            n1, m1, d1, h1, mn1 = followers_record[i+1]
            n_diff = n1-n
            t_diff = datetime.datetime(2016, m1, d1, h1, mn1) - datetime.datetime(2016, m, d, h, mn)
            t_diff = t_diff.days*86400 + t_diff.seconds
            followers_change.append(n_diff/t_diff)
        new_data['new_users'].update({'id': u['id']}, {'$set': {'friends_change_rate': np.std(friends_change), 'followers_change_rate': np.std(followers_change)}}, False, False)
        '''
        
        friends_std = np.std([r[0] for r in friends_record])
        followers_std = np.std([r[0] for r in followers_record])
        ratio_std = np.std([(friends_record[i][0]+1)/(followers_record[i][0]+1) for i in range(len(friends_record))])
        new_data['new_users'].update({'id': u['id']}, {'$set': {'friends_std': friends_std, 'followers_std': followers_std, 'ratio_std': ratio_std}}, False, False)

if __name__ == '__main__':
    main()