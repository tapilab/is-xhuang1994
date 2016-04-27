from TwitterAPI import TwitterAPI
import time
import json
import re
import urllib.parse
from TwitterAPI import TwitterError
import configparser


config = configparser.ConfigParser()
config.read("twitter.cfg")
consumer_key = config.get('twitter', 'consumer_key')
consumer_secret = config.get('twitter', 'consumer_secret')
access_token = config.get('twitter', 'access_token')
access_token_secret = config.get('twitter', 'access_token_secret')

api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
print("Established Twitter connection.")


"""
#This is correct!!!
try:
    user_tweets = api.request("users/lookup", {'user_id': '5884032'})
    user_tweets = [r for r in user_tweets]
except TwitterError.TwitterRequestError as e:
    print(dir(e))
    print(type(e))
    print(e.status_code)
    print(e.__str__())

#print(len(user_tweets))
#url = user_tweets[len(user_tweets)-1]['text']
#print(url)
"""

polluters = []
freader_1 = open("polluters.txt", 'r')
for line in freader_1:
    tokens = re.split("[\t\n]", line)[0]
    polluters.append(tokens)
freader_1.close()
print("user ids read from polluters.txt")

legitimate_users = []
freader_2 = open("legitimate_users.txt", 'r')
for line in freader_2:
    tokens = re.split("[\t\n]", line)[0]
    legitimate_users.append(tokens)
freader_2.close()
print("user ids read from legitimate_users.txt")


fwitter_0 = open('errors.txt', 'w')
fwriter_1 = open('domains_polluters.txt', encoding = 'utf-8', mode = 'w')
for user in polluters:
    while True:
        try:
            user_tweets = api.request("statuses/user_timeline", {'user_id': user, 'count': 200})
            user_tweets = [r for r in user_tweets]
            break
        except TwitterError.TwitterRequestError as tre:
            if tre.status_code == 429:
                print("z z z ...")
                time.sleep(60)
                continue
            else:
                print("Unexpected error raised")
                s = user + '\n' + tre.__str__()
                fwitter_0.write(s)
                break
        except Exception as e:
            print("Unexpected error raised")
            s = user + '\n' + tre.__str__()
            fwitter_0.write(s)
            break
    s = '\n' + user + '\n'
    fwriter_1.write(s)
    for tweet in user_tweets:
        urls = tweet['entities']['urls']
        for url in urls:
            url = url['expanded_url']
            parsed_url = urllib.parse.urlparse(url)
            domain = '{url.scheme}://{url.netloc}/'.format(url=parsed_url)
            fwriter_1.write(domain + '\n')
            fwriter_1.flush()
fwriter_1.close()
fwitter_0.close()

