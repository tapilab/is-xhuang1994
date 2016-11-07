from new_data import read_users, get_users
from TwitterAPI import TwitterAPI, TwitterError
from pymongo import MongoClient
import configparser
import numpy as np
import string
import math
import time


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
    
    while True:
        t = time.localtime()
        if t[4] < 2:
            time_str = "%d\\%d %d:%d" % (t[1], t[2], t[3], t[4])
            print("\nstarting collection process @ %s" % time_str)
            for collection_name in ['bots', 'humans', 'new_users']:
                all_users = read_users(new_data[collection_name])
                relations_count = [[r['id'], r['followers_count'], r['friends_count']] for r in get_users(all_users, api)]
                for r in relations_count:
                    new_data[collection_name].update({'id': r[0]}, {'$push': {'followers_count_record': r[1], 'friends_count_record': r[2]}}, False, False)
                t = time.localtime()
                time_str = "%d\\%d %d:%d" % (t[1], t[2], t[3], t[4])
                print('new_data.%s successfully updated @ %s' % (collection_name, time_str))
        else:
            time.sleep(110)
            continue


if __name__ == '__main__':
    main()