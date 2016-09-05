from TwitterAPI import TwitterAPI
import time
import json
import re
import urllib.parse
from TwitterAPI import TwitterError
import configparser
import math


config = configparser.ConfigParser()
config.read("twitter.cfg")
consumer_key = config.get('twitter', 'consumer_key')
consumer_secret = config.get('twitter', 'consumer_secret')
access_token = config.get('twitter', 'access_token')
access_token_secret = config.get('twitter', 'access_token_secret')

api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
print("Twitter connection ready.")

"""
#This is correct!!!
#try:
users = api.request("users/lookup", {'user_id': "10193122"})
users = [r for r in users]
#except TwitterError.TwitterRequestError as e:
    #print(dir(e))
    #print(type(e))
    #print(e.status_code)
    #print(e.__str__())
print(users)
print(users[0]["id_str"])
print(users[0]["protected"])
"""

bots = []
freader_1 = open("bots.txt", 'r')
for line in freader_1:
    tokens = re.split("[\t\n]", line)[0]
    bots.append(tokens)
freader_1.close()
print("user ids read from bots.txt")

humans = []
freader_2 = open("humans.txt", 'r')
for line in freader_2:
    tokens = re.split("[\t\n]", line)[0]
    humans.append(tokens)
freader_2.close()
print("user ids read from humans.txt")


active_users = []
protected_users = []
for i in range(math.ceil(len(bots)/50)):
    while True:
        try:
            users = api.request("users/lookup", {'user_id': bots[i*50:i*50+50]})
            users = [r for r in users]
            break
        except TwitterError.TwitterRequestError as tre:
            if tre.status_code == 429:
                print(tre.__str__())
                print("current i = ", i)
                print("z z z ...")
                time.sleep(60)
                continue
            else:
                print("Unexpected error raised")
                s = tre.__str__()
                print(s, "\n\n")
                print("current i = ", i)
                break
        except Exception as e:
            print("Unexpected error raised")
            s = e.__str__()
            print(s, "\n\n")
            print("current i = ", i)
            break
    for user in users:
        active_users.append(user["id_str"])
        if user["protected"]:
            protected_users.append(user["id_str"])

inactive_users = [i for i in bots if not i in active_users]
with open("inactive_bots.txt", 'w') as f:
    for u in inactive_users:
        s = u + "\n"
        f.write(s)
with open("protected_bots.txt", 'w') as f:
    for u in protected_users:
        s = u + "\n"
        f.write(s)
