import new_data
from TwitterAPI import TwitterAPI
from pymongo import MongoClient
import configparser

#This file uses Streaming API to find new users on Twitter

#Find more users with Streaming API
def new_users(api, count, collection):
    stream = api.request("statuses/sample")
    user_ids = []
    for tweet in stream:
        if len(user_ids) >= count:
            break
        if not 'delete' in tweet and 
           tweet['user']['lang'] == 'en' and
           tweet['lang'] == 'en' and 
           not tweet['user']['id_str'] in user_ids:
            user_ids.append(tweet['user']['id_str'])
            collection.insert_one({'id': tweet['user']['id_str']})
    
    user_ids = [r['id'] for r in collection.find({}, {'id': 1})]
    users = new_data.get_users(user_ids, api)
    for user in users:
        collection.update({'id': user['id']}, user)
    
    while True:
        user = collection.find_one({'timeline': {'$exists': 0}})
        if user == None:
            break
        user_timeline = new_data.get_timeline(user['id'], api)
        collection.update_one({'id': user['id']}, {'$set': {'timeline': user_timeline['timeline']}})


def main():
    
    config = configparser.ConfigParser()
    config.read("twitter.cfg")
    consumer_key1 = config.get('twitter', 'consumer_key')
    consumer_secret1 = config.get('twitter', 'consumer_secret')
    access_token1 = config.get('twitter', 'access_token')
    access_token_secret1 = config.get('twitter', 'access_token_secret')
    
    mClient = MongoClient()
    db = mClient['new_data']
    api = TwitterAPI(consumer_key1, consumer_secret1, access_token1, access_token_secret1)
    print("Twitter API connection and Mongodb connection ready")
    
    new_users(api, 1100, db['new_users'])
        
    mClient.close()
    
    
if __name__ == '__main__':
    main()