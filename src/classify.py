import re
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import os
from pymongo import MongoClient

#This file reads data from polluters.txt and legitimate_users.txt (old data) or from Mongodb (recent data) and classify the records

"""
#Read data from polluters.txt and legitimate_users.txt
polluters = []
reader_1 = open("old_data" + os.path.sep + "bots.txt", 'r')
for line in reader_1:
    tokens = [float(r) for r in re.split("[\t\n]", line) if r != ""]
    polluters.append(tokens)
reader_1.close()
print("data read from " + "old_data" + os.path.sep + "bots.txt")

legitimate_users = []
reader_2 = open("old_data" + os.path.sep + "humans.txt", 'r')
for line in reader_2:
    tokens = [float(r) for r in re.split("[\t\n]", line) if r != ""]
    legitimate_users.append(tokens)
reader_2.close()
print("data read from " + "old_data" + os.path.sep + "humans.txt")

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
***
geo_enabled
***
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
***
avg_number_of_rts
ratio_tweets_ending_with_punctuation_link_hashtag
num_sources
coordinated
ratio rt?
ratio reply?
tweet syntax
***
"""

#Read data from Mongodb
mClient = MongoClient()
def get_data(ids, collection):
    users = []
    for id in ids:
        user = list(collection.find({'id': id}))[0]
        basic_data = [user['id'], len(user['name']), len(user['screen_name']), user['friends_count'], user['followers_count'], user['friends_count']/user['followers_count'] if user['followers_count'] != 0 else 0, user['statuses_count'], len(user['description']), \
                1 if user['goe_enabled'] == True else 0]
        timeline_data = []
        posts = [r for r in user['timeline'] if r['is_rt'] == False and r['is_reply'] == False]
        urls, mentions, hashtags = [], [], []
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]))
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]) / (len(posts) if len(posts) != 0 else 1))
        for post in posts:
            urls += post['urls']
            mentions += post['mentions']
            hashtags += post['hashtags']
        timeline_data += [len(urls)/(len(posts) if len(posts) != 0 else 1), len(set(urls))/(len(posts) if len(posts) != 0 else 1), len(mentions)/(len(posts) if len(posts) != 0 else 1), len(set(mentions))/(len(posts) if len(posts) != 0 else 1), \
                          len(hashtags)/(len(posts) if len(posts) != 0 else 1), len(set(hashtags))/(len(posts) if len(posts) != 0 else 1)]
        avg_rts = sum([r['rt_count'] for r in user['timeline']]) / len(user['timeline'])
        end_with_phu = len([r for r in user['timeline'] if r['is_rt'] == False and r['is_reply'] == False and r['end_with_phu'] == True]) / len(user['timeline'])
        num_sources = len(set([r['source'] for r in user['timeline']]))
        ratio_coordinate = len([r for r in posts if r['coordinated'] == True]) / (len(posts) if len(posts) != 0 else 1)
        ratio_rt = len([r for r in user['timeline'] if r['is_rt'] == True]) / len(user['timeline'])
        ratio_reply = len([r for r in user['timeline'] if r['is_reply'] == True]) / len(user['timeline'])
        timeline_data += [avg_rts, end_with_phu, num_sources, ratio_coordinate, ratio_rt, ratio_reply]
        users.append(basic_data + timeline_data)
    return users
bots = mClient['new_data']['bots']
humans = mClient['new_data']['humans']
u_bots_id = [r['id'] for r in bots.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}}, {'id': 1})]
u_bots = get_data(u_bots_id, bots)
u_humans_id = [r['id'] for r in humans.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}}, {'id': 1})]
u_humans = get_data(u_humans_id, humans)

mClient.close()

dataset_X = [r[1:] for r in u_bots + u_humans]
dataset_Y = [0] * len(u_bots) + [1] * len(u_humans)

"""
#Combine 2 datasets for classification, dataset_Y is the target list
#the data will be shuffled before classify
dataset_X = [r[1:] for r in polluters + legitimate_users]
dataset_Y = [0] * len(polluters) + [1] * len(legitimate_users)


#Separate dataset_X into bins by # followings and # followers and try classification
#Drop some outliers based on the # followers vs. #followings graph
data_x = []
data_y = []
outliers_x = []
outliers_y = []
i = 0
while i < len(dataset_X):
    if dataset_X[i][0] < 500:
        if dataset_X[i][1] > 1500:
            data_x.append(dataset_X[i])# + [1, 0, 0])
            data_y.append(dataset_Y[i])
        else:
            data_x.append(dataset_X[i])# + [0, 1, 0])
            data_y.append(dataset_Y[i])
    elif dataset_X[i][1] < dataset_X[i][0]*0.9+10000:
        #The #followers/#followings ratio is below the generalized line
        if dataset_X[i][0] < 2005:
            data_x.append(dataset_X[i])# + [0, 1, 0])
            data_y.append(dataset_Y[i])
        else:
            data_x.append(dataset_X[i])# + [0, 0, 1])
            data_y.append(dataset_Y[i])
    else:
        outliers_x.append(dataset_X[i])
        outliers_y.append(dataset_Y[i])
    i += 1
dataset_X = data_x
dataset_Y = data_y
print("%d outliers are dropped" % len(outliers_x))
"""

#Scale each feature of the data with its maximum value
#This is found to give a better result than no scale or scale with both maximum and standard deviation
max_values = np.matrix(dataset_X).mean(0).tolist()[0][:-1] + [1]
dataset_X = [[x/y for x, y in list(zip(z, max_values))] for z in dataset_X]
print("data scaled")
print("%d instances, where %g are bots" % (len(dataset_X), dataset_Y.count(0)/len(dataset_Y)))


#Need KFold iterator since method "cross_val_score" doesn't provide shuffling function
kFold = KFold(n = len(dataset_X), n_folds = 10, shuffle = True)

#Build the classifier and do classification based on: all numeric features & each numeric feature
lr = LogisticRegression()
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50)


#Do classification with Random Forest classifier

#First use cross_val_score to get a total accuracy
score = np.mean(cross_validation.cross_val_score(rf, dataset_X, dataset_Y, cv = kFold))

#Then do cross validation manually to get the f1 score and confusion matrix
#Also count misclassified bots and humans respectively, and calculate the accuracies respectively
e_h, e_b = 0, 0
predictions, y_tests = [], []

for train_index, test_index in kFold:
    dataset_X, dataset_Y = np.array(dataset_X), np.array(dataset_Y)
    X_train, X_test = list(dataset_X[train_index]), list(dataset_X[test_index])
    y_train, y_test = list(dataset_Y[train_index]), list(dataset_Y[test_index])
    rf.fit(X_train, y_train)
    prediction = list(rf.predict(X_test))
    predictions += prediction
    y_tests += y_test
    i = 0
    while i < len(y_test):
        if prediction[i] != y_test[i]:
            if y_test[i] == 0:
                e_b += 1
            else:
                e_h += 1
        i += 1
f1_bots = f1_score(y_tests, predictions, pos_label = 0)
f1_humans = f1_score(y_tests, predictions, pos_label = 1)
conf_matrix = np.matrix(list(confusion_matrix(y_tests, predictions)))

print("\naccuracy:\ntotal:\t%g%%" % round(score*100, 2))
print("humans:\t%g%%" % round((1 - e_h/y_tests.count(1))*100, 2))
print("bots:\t%g%%" % round((1 - e_b/y_tests.count(0))*100, 2))
print("\nf1 scores:\nhumans:\t%g%%\nbots:\t%g%%" % (round(f1_humans*100, 2), round(f1_bots*100, 2)))
print("\nconfusion matrix:\n", conf_matrix, "\n")

#FYI, the labels for confusion matrix are:
#         classified as
#         bots     humans
# bots
# humans

