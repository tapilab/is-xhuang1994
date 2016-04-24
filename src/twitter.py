from TwitterAPI import TwitterAPI
import time
import json
import re


consumer_key = "639XljWu0rfTyCQ3xjuaFdsxZ"
consumer_secret = "GP1dhzQDz6zS7XG2isBqyMQhFfymSbIO1JVvfELptG2viwxMFl"
access_token = "2811894554-16Gded9BKwjA1CbFDcl76K95aQiJp6t6uHSmrmj"
access_token_secret = "hFlfZ30Po4j4is6vNKy2lr6nlW1W0ZRfv75s6i4HegYR3"

api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
print("Established Twitter connection.")


user_tweets = api.request("statuses/user_timeline", user_id = '6301')
s = [r for r in user_tweets]
for a in s:
    print(a['text'])
    print("\n")


"""
polluters = []
reader_1 = open("polluters.txt", 'r')
for line in reader_1:
    tokens = [r for r in re.split("[\t\n]", line)[0]]
    polluters.append(tokens)
print("user ids read from polluters.txt")

legitimate_users = []
reader_2 = open("legitimate_users.txt", 'r')
for line in reader_2:
    tokens = [r for r in re.split("[\t\n]", line)[:23]]
    legitimate_users.append(tokens)
print("user ids read from legitimate_users.txt")

for user in polluters:
    user_tweets = api.request("statuses/user_timeline", '1091571')
    tweets = [r for r in user_tweets]
    print(tweets)
"""