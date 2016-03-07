import re
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import numpy as np
from Lib import statistics as stat
import warnings
import csv
import datetime as dt
import sys

#warnings.filterwarnings('always')

#read the text file of polluters information
reader_1 = open("social_honeypot_icwsm_2011\content_polluters.txt", 'r')

#convert the content read into usable dataset
polluters = []
for line in reader_1:
    tokens = re.split("[\t\n]", line)
    #parse strings to numbers, take out the dates
    parsed_tokens = [int(tokens[0]), tokens[1], tokens[2]] + [int(r) for r in (tokens[3:8])]
    polluters.append(parsed_tokens)
reader_1.close()

#read the file of legitimate users information, and convert it into usable dataset
reader_2 = open("social_honeypot_icwsm_2011\legitimate_users.txt", 'r')
legitimate_users = []
for line in reader_2:
    tokens = re.split("[\t\n]", line)
    parsed_tokens = [int(tokens[0]), tokens[1], tokens[2]] + [int(r) for r in (tokens[3:8])]
    legitimate_users.append(parsed_tokens)
reader_2.close()

#read text file containing series of number of followings for each bot user
reader_3 = open("social_honeypot_icwsm_2011\content_polluters_followings.txt", 'r')

#convert the contend read into usable dataset
i = 0
for line in reader_3:
    tokens = [int(r) for r in re.split("[\D]", line) if r != ""][1:]
    #calculate standard deviation of the differences of the data
    sd = stat.pstdev(tokens)
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
    polluters[i].append(ac)
    i += 1
reader_3.close()

#read text file containing series of number of followings for each human user
reader_4 = open("social_honeypot_icwsm_2011\legitimate_users_followings.txt", 'r')

#convert the contend read into usable dataset
i = 0
for line in reader_4:
    tokens = [int(r) for r in re.split("[\D]", line) if r != ""][1:]
    #calculate standard deviation of the differences of the data
    sd = stat.pstdev(tokens)
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
    legitimate_users[i].append(ac)
    i += 1
reader_4.close()

#read the text file of tweets posted by each bot user
reader_5 = open("social_honeypot_icwsm_2011\content_polluters_tweets.txt", encoding = 'utf-8', mode = 'r')

#convert the contend read into usable dataset
prev_userID = 0
curr_userID = 0
curr_tweetIDs = []          #collection of ids of tweets posted by current user
curr_tweets = []            #collection of tweets posted by current user
curr_tweets_created_at = [] #create dates of tweets
curr_tweets_weekday = []    #weekdays of create dates of tweets
polluter_ids = [s[0] for s in polluters]
for line in reader_5:
    tokens = [r for r in re.split("[\t\n]", line) if r != ""]
    #the date the tweet was posted
    date_info = re.split("[-\s]", tokens[3])
    post_date = dt.date(int(date_info[0]), int(date_info[1]), int(date_info[2]))

    prev_userID = curr_userID
    curr_userID = int(tokens[0])
    #new user found
    if curr_userID != prev_userID and curr_tweetIDs != []:
        j = 0
        while j < 7:
            polluters[polluter_ids.index(prev_userID)].append(curr_tweets_weekday.count(j))
            j += 1
        k = 0
        while k < 7:
            polluters[polluter_ids.index(prev_userID)].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
            k += 1
        curr_tweetIDs = []
        curr_tweets = []
        curr_tweets_created_at = []
        curr_tweets_weekday = []
    
    curr_tweetIDs.append(int(tokens[1]))
    curr_tweets_weekday.append(post_date.weekday())
    
j = 0
while j < 7:
    polluters[polluter_ids.index(curr_userID)].append(curr_tweets_weekday.count(j))
    j += 1
k = 0
while k < 7:
    polluters[polluter_ids.index(curr_userID)].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
    k += 1


#read the text file of tweets posted by each bot user
reader_6 = open("social_honeypot_icwsm_2011\legitimate_users_tweets.txt", encoding = 'utf-8', mode = 'r')

#convert the contend read into usable dataset
prev_userID = 0
curr_userID = 0
curr_tweetIDs = []          #collection of ids of tweets posted by current user
curr_tweets = []            #collection of tweets posted by current user
curr_tweets_created_at = [] #create dates of tweets
curr_tweets_weekday = []    #weekdays of create dates of tweets
legitimate_user_ids = [s[0] for s in legitimate_users]
for line in reader_6:
    tokens = [r for r in re.split("[\t\n]", line) if r != ""]
    #the date the tweet was posted
    date_info = re.split("[-\s]", tokens[3])
    post_date = dt.date(int(date_info[0]), int(date_info[1]), int(date_info[2]))

    prev_userID = curr_userID
    curr_userID = int(tokens[0])
    #new user found
    if curr_userID != prev_userID and curr_tweetIDs != []:
        j = 0
        while j < 7:
            legitimate_users[legitimate_user_ids.index(prev_userID)].append(curr_tweets_weekday.count(j))
            j += 1
        k = 0
        while k < 7:
            legitimate_users[legitimate_user_ids.index(prev_userID)].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
            k += 1
        curr_tweetIDs = []
        curr_tweets = []
        curr_tweets_created_at = []
        curr_tweets_weekday = []
    
    curr_tweetIDs.append(int(tokens[1]))
    curr_tweets_weekday.append(post_date.weekday())
    
j = 0
while j < 7:
    legitimate_users[legitimate_user_ids.index(curr_userID)].append(curr_tweets_weekday.count(j))
    j += 1
k = 0
while k < 7:
    legitimate_users[legitimate_user_ids.index(curr_userID)].append(curr_tweets_weekday.count(k) / len(curr_tweets_weekday))
    k += 1

i = 0
for u in polluters:
    if len(u) != 24:
        if len(u) != 10:
            print("error at: ", i)
        else:
            u += [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    i += 1

i = 0
for u in legitimate_users:
    if len(u) != 24:
        if len(u) != 10:
            print("error at: ", i)
        else:
            u += [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    i += 1


#some index numbers of specific information for future use
userID = 0
createdAt = 1
collectedAt = 2
numberOfFollowings = 3
numberOfFollowers = 4
numberOfTweets = 5
lengthOfScreenName = 6
lengthOfDescriptionInUserProfile = 7
standard_deviation_diff = 8
lag1_autocorrelation = 9
number_tweets_Monday = 10
#omited index numbers for number of tweets posted each day Tuesday - Saturday
number_tweets_Sunday = 16
ratio_tweets_Monday = 17
#omited index numbers for ratio of tweets posted each day Tuesday - Saturday
ratio_tweets_Sunday = 23




#combine 2 datasets for classification, dataset_Y is the target list
#the data will be shuffled before classify
dataset_X = polluters + legitimate_users
dataset_Y = [0] * len(polluters) + [1] * len(legitimate_users)

#need KFold iterator since method "cross_val_score" doesn't provide shuffling function
kFold = KFold(n = len(dataset_X), n_folds = 10, shuffle = True)

#build the classifier and do classification based on: all numeric features & each numeric feature
lr = LogisticRegression()

#number of iteration
n = 6

dataset_X_all_numeric_features = [r[3:24] for r in dataset_X]
i = 0
score = 0
while i < n:
    score += np.mean(cross_validation.cross_val_score(lr, dataset_X_all_numeric_features, dataset_Y, cv = kFold))
    i += 1
print("All numeric features: ", score / n)

dataset_X_numberOfFollowings = [[x[3]] for x in dataset_X]
i = 0
scores_numberOfFollowings = 0
while i < n:
    scores_numberOfFollowings += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfFollowings, dataset_Y, cv = kFold))
    i += 1
print("Number of followings: ", scores_numberOfFollowings / n)

dataset_X_numberOfFollowers = [[x[4]] for x in dataset_X]
i = 0
scores_numberOfFollowers = 0
while i < n:
    scores_numberOfFollowers += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfFollowers, dataset_Y, cv = kFold))
    i += 1
print("Number of followers: ", scores_numberOfFollowers / n)

dataset_X_numberOfTweets = [[x[5]] for x in dataset_X]
i = 0
scores_numberOfTweets = 0
while i < n:
    scores_numberOfTweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfTweets, dataset_Y, cv = kFold))
    i += 1
print("Number of tweets: ", scores_numberOfTweets / n)

dataset_X_lengthOfScreenName = [[x[6]] for x in dataset_X]
i = 0
scores_lengthOfScreenName = 0
while i < n:
    scores_lengthOfScreenName += np.mean(cross_validation.cross_val_score(lr, dataset_X_lengthOfScreenName, dataset_Y, cv = kFold))
    i += 1
print("Length of screen name: ", scores_lengthOfScreenName / n)

dataset_X_lengthOfDescriptionInUserProfile = [[x[7]] for x in dataset_X]
i = 0
scores_lengthOfDescriptionInUserProfile = 0
while i < n:
    scores_lengthOfDescriptionInUserProfile += np.mean(cross_validation.cross_val_score(lr, dataset_X_lengthOfDescriptionInUserProfile, dataset_Y, cv = kFold))
    i += 1
print("Length of description in user profile: ", scores_lengthOfDescriptionInUserProfile / n)

dataset_X_standard_deviation_diff = [[r[8]] for r in dataset_X]
i = 0
scores_standard_deviation_diff = 0
while i < n:
    scores_standard_deviation_diff += np.mean(cross_validation.cross_val_score(lr, dataset_X_standard_deviation_diff, dataset_Y, cv = kFold))
    i += 1
print("standard deviation diff: ", scores_standard_deviation_diff / n)

dataset_X_lag1_autocorrelation = [[r[9]] for r in dataset_X]
i = 0
scores_lag1_autocorrelation = 0
while i < n:
    scores_lag1_autocorrelation += np.mean(cross_validation.cross_val_score(lr, dataset_X_lag1_autocorrelation, dataset_Y, cv = kFold))
    i += 1
print("lag1 autocorrelation: ", scores_lag1_autocorrelation / n)

dataset_X_number_tweets_weekdays = [r[10:17] for r in dataset_X]
i = 0
scores_number_tweets_weekdays = 0
while i < n:
    scores_number_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_number_tweets_weekdays, dataset_Y, cv = kFold))
    i += 1
print("Number of tweets each weekday: ", scores_number_tweets_weekdays / n)

dataset_X_ratio_tweets_weekdays = [r[17:] for r in dataset_X]
i = 0
scores_ratio_tweets_weekdays = 0
while i < n:
    scores_ratio_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_ratio_tweets_weekdays, dataset_Y, cv = kFold))
    i += 1
print("Ratio of tweets each weekday: ", scores_ratio_tweets_weekdays / n)

print("\n****************************************************\n")

dataset_X_no_numberOfFollowings = [x[4:] for x in dataset_X]
i = 0
scores_no_numberOfFollowings = 0
while i < n:
    scores_no_numberOfFollowings += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfFollowings, dataset_Y, cv = kFold))
    i += 1
print("No number of followings: ", scores_no_numberOfFollowings / n)

dataset_X_no_numberOfFollowers = [[x[3]] + x[5:] for x in dataset_X]
i = 0
scores_no_numberOfFollowers = 0
while i < n:
    scores_no_numberOfFollowers += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfFollowers, dataset_Y, cv = kFold))
    i += 1
print("No number of followers: ", scores_no_numberOfFollowers / n)

dataset_X_no_numberOfTweets = [x[3:5] + x[6:] for x in dataset_X]
i = 0
scores_no_numberOfTweets = 0
while i < n:
    scores_no_numberOfTweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfTweets, dataset_Y, cv = kFold))
    i += 1
print("No number of tweets: ", scores_no_numberOfTweets / n)

dataset_X_no_lengthOfScreenName = [x[3:6] + x[7:] for x in dataset_X]
i = 0
scores_no_lengthOfScreenName = 0
while i < n:
    scores_no_lengthOfScreenName += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lengthOfScreenName, dataset_Y, cv = kFold))
    i += 1
print("No length of screen name: ", scores_no_lengthOfScreenName / n)

dataset_X_no_lengthOfDescriptionInUserProfile = [x[3:7] + x[8:] for x in dataset_X]
i = 0
scores_no_lengthOfDescriptionInUserProfile = 0
while i < n:
    scores_no_lengthOfDescriptionInUserProfile += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lengthOfDescriptionInUserProfile, dataset_Y, cv = kFold))
    i += 1
print("No length of description in user profile: ", scores_no_lengthOfDescriptionInUserProfile / n)

dataset_X_no_standard_deviation_diff = [r[3:8] + r[9:] for r in dataset_X]
i = 0
scores_no_standard_deviation_diff = 0
while i < n:
    scores_no_standard_deviation_diff += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_standard_deviation_diff, dataset_Y, cv = kFold))
    i += 1
print("no standard deviation diff: ", scores_no_standard_deviation_diff / n)

dataset_X_no_lag1_autocorrelation = [r[3:9] + r[10:] for r in dataset_X]
i = 0
scores_no_lag1_autocorrelation = 0
while i < n:
    scores_no_lag1_autocorrelation += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lag1_autocorrelation, dataset_Y, cv = kFold))
    i += 1
print("no lag1 autocorrelation: ", scores_no_lag1_autocorrelation / n)

dataset_X_no_number_tweets_weekdays = [r[3:10] + r[17:] for r in dataset_X]
i = 0
scores_no_number_tweets_weekdays = 0
while i < n:
    scores_no_number_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_number_tweets_weekdays, dataset_Y, cv = kFold))
    i += 1
print("no number of tweets each weekday: ", scores_no_number_tweets_weekdays / n)

dataset_X_no_ratio_tweets_weekdays = [r[3:17] for r in dataset_X]
i = 0
scores_no_ratio_tweets_weekdays = 0
while i < n:
    scores_no_ratio_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_ratio_tweets_weekdays, dataset_Y, cv = kFold))
    i += 1
print("no ratio of tweets each weekday: ", scores_no_ratio_tweets_weekdays / n)

print("\n****************************************************\n")
print("Coefficients: \n")

lr.fit(dataset_X_all_numeric_features, dataset_Y)
coef = lr.coef_[0]
print("Number of followings: ", coef[0])
print("Number of followers: ", coef[1])
print("Number of tweets: ", coef[2])
print("Length of screen name: ", coef[3])
print("Length of description in user profile: ", coef[4])
print("Standard deviation differences: ", coef[5])
print("Lag one autocorrelation: ", coef[6])
print("Number of tweets each weekday (avg): ", np.mean(coef[7:14]))
print("Ratio of tweets each weekday (avg): ", np.mean(coef[14:]))

