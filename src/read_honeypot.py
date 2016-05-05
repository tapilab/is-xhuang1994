import re
import numpy as np
from Lib import statistics as stat
import datetime as dt
from urllib.request import urlretrieve
import zipfile
import os

#This file reads data from honeypot dataset, process and write the data into bots.txt and humans.txt

num_decimal = 4
#Reading basic info for each user into container
def read_info(filename, container):
    with open(filename, 'r') as f:
        for line in f:
            tokens = re.split("[\t\n]", line)
            #Parse strings to numbers, take out the dates
            parsed_tokens = [tokens[0]] + [int(r) for r in tokens[3:8]]
            #Remove length of screen name from data, which is found to have negative effect on classification accuracy
            parsed_tokens = parsed_tokens[:3] + parsed_tokens[4:]
            #Add # followings / # followers
            parsed_tokens.append(round((parsed_tokens[1]/parsed_tokens[2] if parsed_tokens[2] else 0), num_decimal))
            container.append(parsed_tokens)

#Reading data of a series of # followings for each user
def read_followings(filename, container):
    with open(filename, 'r') as f:
        i = 0
        for line in f:
            tokens = [int(r) for r in re.split("[\D]", line) if r != ""][1:]
            #Calculate standard deviation of the data
            sd = round(stat.pstdev(tokens), num_decimal)
            #Calcalate standard deviation of differences of the data
            sdd = round(stat.pstdev(list(np.array(tokens[1:]) - np.array(tokens[:-1]))), num_decimal)
            #Calculate lag one autocorrelation of the data
            avg = np.mean(tokens)
            numerator = sum((np.array(tokens[1:]) - avg) * (np.array(tokens[:-1]) - avg))
            denominator = sum((np.array(tokens) - avg) ** 2)
            lac = round(numerator/denominator, num_decimal) if denominator != 0 else 0
            container[i] += [sd, sdd, lac]
            i += 1

#Reading tweets posted by each user
def read_tweets(filename, container):
    curr_userID = ""
    curr_tweet_count = 0
    #Features contained in tweets (current user)
    urls, _at_s, hashtags, weekday_post = [], [], [], []
    #Index for container
    i = 0
    with open(filename, encoding = 'utf-8', mode = 'r') as f:
        while True:
            line = f.readline()
            tokens = [r for r in re.split("[\t\n]", line) if r != ""]
            if not line or tokens[0] != curr_userID and curr_userID:
                #New user found / eof reached
                num_tweets_weekday = [weekday_post.count(i) for i in range(7)]
                ratio_tweets_weekday = [round(x, num_decimal) for x in list(np.array(num_tweets_weekday) / len(weekday_post))]
                curr_user = num_tweets_weekday + ratio_tweets_weekday
                for feature in (urls, _at_s, hashtags):
                    curr_user.append(round(len(feature) / curr_tweet_count, num_decimal))
                    curr_user.append(round(len(set(feature)) / curr_tweet_count, num_decimal))
                while curr_userID != container[i][0]:
                    i += 1
                container[i] += curr_user
                if not line:
                    break
                #Reset current user info containers
                curr_tweet_count = 0
                urls, _at_s, hashtags, weekday_post = [], [], [], []
            #Post date of the tweet
            curr_userID = tokens[0]
            curr_tweet_count += 1
            urls += re.findall('http[\S]+', tokens[2])
            _at_s += re.findall('@[\S]+', tokens[2])
            hashtags += re.findall('#[\S]+', tokens[2])
            post_date = re.split("[-\s]", tokens[3])
            post_date = dt.date(int(post_date[0]), int(post_date[1]), int(post_date[2]))
            weekday_post.append(post_date.weekday())

#Deleting ambiguous users who are in both polluters and legitimate users (44 found)
#The user ids are found in ascending order
def del_amb(bots, humans):
    i, j, count = (0, 0, 0)
    while i < len(bots) and j < len(humans):
        if bots[i][0] == humans[j][0]:
            bots.pop(i)
            humans.pop(j)
            count += 1
        elif int(bots[i][0]) < int(humans[j][0]):
            i += 1
        else:
            j += 1
    return count

#Add 0's for missing values (some users have no tweets recorded)
def add0(container):
    length = 29
    for i in range(len(container)):
        container[i] += [0]*(length - len(container[i]))

#Write data into text files
def write_user(filename, container):
    with open(filename, 'w') as f:
        for inst in container:
            f.write("\t".join([str(x) for x in inst]))
            f.write("\n")

def main():
    if not os.path.exists('social_honeypot_icwsm_2011'):
        if not os.path.exists('social_honeypot_icwsm_2011.zip'):
            print("downloading data")
            urlretrieve('http://infolab.tamu.edu/static/users/kyumin/social_honeypot_icwsm_2011.zip', 'social_honeypot_icwsm_2011.zip')
        zip = zipfile.ZipFile('social_honeypot_icwsm_2011.zip')
        zip.extractall(path = 'social_honeypot_icwsm_2011')
        zip.close()
        print("data ready")
       
    bots, humans = [], []
    read_info('social_honeypot_icwsm_2011\content_polluters.txt', bots)
    print("data read from social_honeypot_icwsm_2011\content_polluters.txt")
    read_info('social_honeypot_icwsm_2011\legitimate_users.txt', humans)
    print("data read from social_honeypot_icwsm_2011\legitimate_users.txt")
    
    read_followings('social_honeypot_icwsm_2011\content_polluters_followings.txt', bots)
    print("data read from social_honeypot_icwsm_2011\content_polluters_followings.txt")
    read_followings('social_honeypot_icwsm_2011\legitimate_users_followings.txt', humans)
    print("data read from social_honeypot_icwsm_2011\legitimate_users_followings.txt")
    
    read_tweets('social_honeypot_icwsm_2011\content_polluters_tweets.txt', bots)
    print("data read from social_honeypot_icwsm_2011\content_polluters_tweets.txt")
    read_tweets('social_honeypot_icwsm_2011\legitimate_users_tweets.txt', humans)
    print("data read from social_honeypot_icwsm_2011\legitimate_users_tweets.txt")
    
    count = del_amb(bots, humans)
    print("%d mislabeled users deleted!" % count)
    add0(bots)
    add0(humans)
    print("added 0's for missing values")
    
    write_user('bots.txt', bots)
    print("data written to bots.txt")
    write_user('humans.txt', humans)
    print("data written to humans.txt")


if __name__ == '__main__':
    main()


#some index numbers of specific information for future use
userID = 0
numberOfFollowings = 1
numberOfFollowers = 2
numberOfTweets = 3
lengthOfDescriptionInUserProfile = 4
ratio_followings_followers = 5
standard_deviation_follwings = 6
standard_deviation_diff_follwings = 7
lag1_autocorrelation = 8
number_tweets_Monday = 9
#omited index numbers for number of tweets posted each day Tuesday - Saturday
number_tweets_Sunday = 15
ratio_tweets_Monday = 16
#omited index numbers for ratio of tweets posted each day Tuesday - Saturday
ratio_tweets_Sunday = 22
ratio_urls_tweets = 23
ratio_unique_urls_tweets = 24
ratio_at_tweets = 25
ratio_unique_at_tweets = 26
ratio_hashtags_tweets = 27
ratio_unique_hashtags_tweets = 28