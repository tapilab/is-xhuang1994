import re
import numpy as np
from Lib import statistics as stat
import datetime as dt

#This file reads data from honeypot dataset, process and write the data into polluters.txt and legitimate_users.txt


#read the text file of polluters information
reader_1 = open("social_honeypot_icwsm_2011\content_polluters.txt", 'r')

#convert the content read into usable dataset
polluters = []
for line in reader_1:
    tokens = re.split("[\t\n]", line)
    #parse strings to numbers, take out the dates
    parsed_tokens = [int(r) for r in ([tokens[0]] + tokens[3:8])]
    polluters.append(parsed_tokens)
reader_1.close()
print("data read from social_honeypot_icwsm_2011\content_polluters.txt")

#read the file of legitimate users information, and convert it into usable dataset
reader_2 = open("social_honeypot_icwsm_2011\legitimate_users.txt", 'r')
legitimate_users = []
for line in reader_2:
    tokens = re.split("[\t\n]", line)
    parsed_tokens = [int(r) for r in ([tokens[0]] + tokens[3:8])]
    legitimate_users.append(parsed_tokens)
reader_2.close()
print("data read from social_honeypot_icwsm_2011\legitimate_users.txt")

#read text file containing series of number of followings for each bot user
reader_3 = open("social_honeypot_icwsm_2011\content_polluters_followings.txt", 'r')

#convert the contend read into usable dataset
i = 0
for line in reader_3:
    tokens = [int(r) for r in re.split("[\D]", line) if r != ""][1:]
    #calculate standard deviation of the differences of the data
    sd = stat.pstdev(tokens)
    fd = []
    j = 1
    while j < len(tokens):
        fd.append(tokens[j] - tokens[j-1])
        j += 1
    sdd = stat.pstdev(fd)
    #calculate lag one autocorrelation
    #ac = np.correlate(tokens[1:len(tokens)], tokens[:len(tokens) - 1])  """I can't find what I need from output from this function as a single number"""
    avg = np.mean(tokens)
    j = 0
    numerator = 0
    denominator = (tokens[0] - avg) ** 2
    while j < len(tokens) - 1:
        numerator += (tokens[j] - avg) * (tokens[j+1] - avg)
        denominator += (tokens[j+1] - avg) ** 2
        j += 1
    if denominator == 0:
        ac = 0
    else:
        ac = numerator / denominator
    polluters[i].append(sd)
    polluters[i].append(sdd)
    polluters[i].append(ac)
    i += 1
reader_3.close()
print("data read from social_honeypot_icwsm_2011\content_polluters_followings.txt")

#read text file containing series of number of followings for each human user
reader_4 = open("social_honeypot_icwsm_2011\legitimate_users_followings.txt", 'r')

#convert the contend read into usable dataset
i = 0
for line in reader_4:
    tokens = [int(r) for r in re.split("[\D]", line) if r != ""][1:]
    #calculate standard deviation of the differences of the data
    sd = stat.pstdev(tokens)
    fd = []
    j = 1
    while j < len(tokens):
        fd.append(tokens[j] - tokens[j-1])
        j += 1
    sdd = stat.pstdev(fd)
    #calculate lag one autocorrelation
    avg = np.mean(tokens)
    j = 0
    numerator = 0
    denominator = (tokens[0] - avg) ** 2
    while j < len(tokens) - 1:
        numerator += (tokens[j] - avg) * (tokens[j+1] - avg)
        denominator += (tokens[j+1] - avg) ** 2
        j += 1
    if denominator == 0:
        ac = 0
    else:
        ac = numerator / denominator
    legitimate_users[i].append(sd)
    legitimate_users[i].append(sdd)
    legitimate_users[i].append(ac)
    i += 1
reader_4.close()
print("data read from social_honeypot_icwsm_2011\legitimate_users_followings.txt")

#read the text file of tweets posted by each bot user
reader_5 = open("social_honeypot_icwsm_2011\content_polluters_tweets.txt", encoding = 'utf-8', mode = 'r')

urls_polluters = []         #collection of urls from polluters
#convert the contend read into usable dataset
prev_userID = 0
curr_userID = 0
curr_tweetIDs = []          #collection of ids of tweets posted by current user
curr_tweets = []            #collection of tweets posted by current user
curr_tweets_created_at = [] #create dates of tweets
curr_tweets_weekday = []    #weekdays of create dates of tweets
urls_tweets = []             #collection of urls from tweets of current user
at_tweets = []               #collection of @ from tweets of current user
hashtags_tweets = []         #collection of hashtags from tweets of current user
polluter_ids = [s[0] for s in polluters]
curr_line = reader_5.readline()
while 1:
    prev_userID = curr_userID
    if curr_line != '':
        tokens = [r for r in re.split("[\t\n]", curr_line) if r != ""]
        #the date the tweet was posted
        date_info = re.split("[-\s]", tokens[3])
        post_date = dt.date(int(date_info[0]), int(date_info[1]), int(date_info[2]))

        curr_userID = int(tokens[0])
    #new user found / end of the file
    if prev_userID != 0 and curr_userID != prev_userID or curr_line == '':
        index = polluter_ids.index(prev_userID)
        j = 0
        while j < 7:
            polluters[index].append(curr_tweets_weekday.count(j))
            j += 1
        k = 0
        while k < 7:
            polluters[index].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
            k += 1
        urls_polluters += urls_tweets
        polluters[index].append(len(urls_tweets) / len(curr_tweets))
        polluters[index].append(len(set(urls_tweets)) / len(curr_tweets))
        polluters[index].append(len(at_tweets) / len(curr_tweets))
        polluters[index].append(len(set(at_tweets)) / len(curr_tweets))
        polluters[index].append(len(hashtags_tweets) / len(curr_tweets))
        polluters[index].append(len(set(hashtags_tweets)) / len(curr_tweets))
        urls_tweets = []
        at_tweets = []
        hashtags_tweets = []
        curr_tweetIDs = []
        curr_tweets = []
        curr_tweets_created_at = []
        curr_tweets_weekday = []
        if curr_line == '':
            break
    urls_tweets += re.findall('http[\S]+', tokens[2])
    at_tweets += re.findall('@[\S]+', tokens[2])
    hashtags_tweets += re.findall('#[\S]+', tokens[2])
    curr_tweets.append(tokens[2])
    curr_tweetIDs.append(int(tokens[1]))
    curr_tweets_weekday.append(post_date.weekday())
    curr_line = reader_5.readline()
reader_5.close()
print("data read from social_honeypot_icwsm_2011\content_polluters_tweets.txt")

#read the text file of tweets posted by each human user
reader_6 = open("social_honeypot_icwsm_2011\legitimate_users_tweets.txt", encoding = 'utf-8', mode = 'r')

urls_humans = []            #collection of urls from humans
#convert the contend read into usable dataset
prev_userID = 0
curr_userID = 0
curr_tweetIDs = []          #collection of ids of tweets posted by current user
curr_tweets = []            #collection of tweets posted by current user
curr_tweets_created_at = [] #create dates of tweets
curr_tweets_weekday = []    #weekdays of create dates of tweets
urls_tweets = []             #collection of urls from tweets of current user
at_tweets = []               #collection of @ from tweets of current user
hashtags_tweets = []         #collection of hashtags from tweets of current user
legitimate_user_ids = [s[0] for s in legitimate_users]
curr_line = reader_6.readline()
while 1:
    prev_userID = curr_userID
    if curr_line != '':
        tokens = [r for r in re.split("[\t\n]", curr_line) if r != ""]
        #the date the tweet was posted
        date_info = re.split("[-\s]", tokens[3])
        post_date = dt.date(int(date_info[0]), int(date_info[1]), int(date_info[2]))

        curr_userID = int(tokens[0])
    #new user found / end of the file
    if prev_userID != 0 and curr_userID != prev_userID or curr_line == '':
        index = legitimate_user_ids.index(prev_userID)
        j = 0
        while j < 7:
            legitimate_users[index].append(curr_tweets_weekday.count(j))
            j += 1
        k = 0
        while k < 7:
            legitimate_users[index].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
            k += 1
        urls_humans += urls_tweets
        legitimate_users[index].append(len(urls_tweets) / len(curr_tweets))
        legitimate_users[index].append(len(set(urls_tweets)) / len(curr_tweets))
        legitimate_users[index].append(len(at_tweets) / len(curr_tweets))
        legitimate_users[index].append(len(set(at_tweets)) / len(curr_tweets))
        legitimate_users[index].append(len(hashtags_tweets) / len(curr_tweets))
        legitimate_users[index].append(len(set(hashtags_tweets)) / len(curr_tweets))
        urls_tweets = []
        at_tweets = []
        hashtags_tweets = []
        curr_tweetIDs = []
        curr_tweets = []
        curr_tweets_created_at = []
        curr_tweets_weekday = []
        if curr_line == '':
            break
    urls_tweets += re.findall('http[\S]+', tokens[2])
    at_tweets += re.findall('@[\S]+', tokens[2])
    hashtags_tweets += re.findall('#[\S]+', tokens[2])
    curr_tweets.append(tokens[2])
    curr_tweetIDs.append(int(tokens[1]))
    curr_tweets_weekday.append(post_date.weekday())
    curr_line = reader_6.readline()
reader_6.close()
print("data read from social_honeypot_icwsm_2011\legitimate_users_tweets.txt")


#Found 44 ambiguous users who are in both polluters and legitimate users, delete those users
#The user ids are found in ascending order
i = 0
j = 0
count = 0
while i < len(polluters) and j < len(legitimate_users):
    if polluters[i][0] == legitimate_users[j][0]:
        polluters.pop(i)
        legitimate_users.pop(j)
        count += 1
    elif polluters[i][0] < legitimate_users[j][0]:
        i += 1
    else:
        j += 1
print("%d mislabeled users deleted!" % count)


#add 0's for missing values (some users have no tweets recorded)
i = 0
while i < len(polluters):
    if len(polluters[i]) < 23:
        polluters[i] += [0] * 15
    i += 1
j = 0
while j < len(legitimate_users):
    if len(legitimate_users[j]) < 23:
        legitimate_users[j] += [0] * 15
    j += 1
print("added 0's for missing values")


#write collected & processed data into files
fwriter_1 = open('polluters.txt', 'w')
for record in polluters:
    s = '\t'.join([str(r) for r in record]) + '\n'
    fwriter_1.write(s)
fwriter_1.close()
print("data written to polluters.txt")

fwriter_2 = open('legitimate_users.txt', 'w')
for record in legitimate_users:
    s = '\t'.join([str(r) for r in record]) + '\n'
    fwriter_2.write(s)
fwriter_2.close()
print("data written to legitimate_users.txt")

fwriter_3 = open('urls_polluters.txt', encoding = 'utf-8', mode = 'w')
for url in urls_polluters:
    s = url + '\n'
    fwriter_3.write(s)
fwriter_3.close()
print("data written to urls_polluters.txt")

fwriter_4 = open('urls_legitimate_users.txt', encoding = 'utf-8', mode = 'w')
for url in urls_humans:
    s = url + '\n'
    fwriter_4.write(s)
fwriter_3.close()
print("data written to urls_legitimate_users.txt")

#some index numbers of specific information for future use
userID = 0
numberOfFollowings = 1
numberOfFollowers = 2
numberOfTweets = 3
lengthOfScreenName = 4
lengthOfDescriptionInUserProfile = 5
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
