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
"""

#Read data from Mongodb
mClient = MongoClient()
db = mClient['new_data']
weekdays = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sta': 5, 'Sun': 6}
bots_basic = db['a_bots'].find({'protected': False}, {'id': 1, 'screen_name': 1, 'followers_count': 1, 'friends_count': 1, 'description': 1, 'statuses_count': 1})
bots_basic = [r for r in bots_basic]
bots_timeline = db['a_bots_timeline'].find({})
bots_timeline = [r for r in bots_timeline]
bots_timeline_f = []
for b in bots_timeline:
    this_bot = {'id': b['id'], 'num_weekday': [0, 0, 0, 0, 0, 0, 0], 'ratio_weekday': [0, 0, 0, 0, 0, 0, 0], 'ratio_urls': [0, 0], 'ratio_at': [0, 0], 'ratio_hashtags': [0, 0]}
    urls, mentions, hashtags = [], [], []
    for tweet in b['timeline']:
        day = weekdays[tweet['created_at'][0:3]]
        this_bot['num_weekday'][day] += 1
        mentions.append([r['id_str'] for r in tweet['user_mentions']])
        urls.append(tweet['urls'])
        hashtags.append([r['text'] for r in tweet['hashtags']])
    if b['timeline'] != []:
        this_bot['ratio_weekday'] = list(np.array(this_bot['num_weekday']) / np.sum(this_bot['num_weekday']))
        this_bot['ratio_urls'] = [len(urls)/len(b['timeline']), len(set(urls))//len(b['timeline'])]
        this_bot['ratio_at'] = [len(mentions)/len(b['timeline']), len(set(mentions))//len(b['timeline'])]
        this_bot['ratio_hashtags'] = [len(hashtags)/len(b['timeline']), len(set(hashtags))//len(b['timeline'])]
    bots_timeline_f.append(this_bot)
        
    
bots = [[r['id'], r['friends_count'], r['followers_count'], r['statuses_count'], len(r['screen_name']), len(r['description'])] for r in bots_basic]
humans_basic = db['a_humans'].find({'protected': False}, {'id': 1, 'screen_name': 1, 'followers_count': 1, 'friends_count': 1, 'description': 1, 'statuses_count': 1})
humans_basic = [r for r in humans_basic]
humans = [[r['id'], r['friends_count'], r['followers_count'], r['statuses_count'], len(r['screen_name']), len(r['description'])] for r in humans_basic]

dataset_X = [r[1:] for r in bots + humans]
dataset_Y = [0] * len(bots) + [1] * len(humans)

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

