import re
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold

#read the text file of polluters information
reader_1 = open("social_honeypot_icwsm_2011\content_polluters.txt", 'r')

#convert the content read into usable dataset
polluters = []
for line in reader_1:
    tokens = re.split("[\t\n]", line)
    #parse strings to numbers, take out the dates
    parsed_tokens = [int(r) for r in ([tokens[0]] + tokens[3:8])]
    polluters.append(parsed_tokens)

#read the file of legitimate users information, and convert it into usable dataset
reader_2 = open("social_honeypot_icwsm_2011\legitimate_users.txt", 'r')
legitimate_users = []
for line in reader_2:
    tokens = re.split("[\t\n]", line)
    parsed_tokens = [int(r) for r in ([tokens[0]] + tokens[3:8])]
    legitimate_users.append(parsed_tokens)

#some index numbers of specific information for future use
userID = 0
createdAt = 1
collectedAt = 2
numberOfFollowings = 3
numberOfFollowers = 4
numberOfTweets = 5
lengthOfScreenName = 6
lengthOfDescriptionInUserProfile = 7

#combine 2 datasets for classification, dataset_Y is the target list
#the data will be shuffled before classify
dataset_X = polluters + legitimate_users
dataset_Y = [0] * len(polluters) + [1] * len(legitimate_users)

"""
#How to implement cross validation by LogisticRegressionCV?

kFold = KFold(n = len(dataset_X), n_folds = 10, shuffle = True)
lr = LogisticRegressionCV(cv = kFold)
scores_1 = lr.score(dataset_X, dataset_Y)
print(scores_1)
"""

#need KFold iterator since method "cross_val_score" doesn't provide shuffling function
kFold = KFold(n = len(dataset_X), n_folds = 10, shuffle = True)

#build the classifier and do classification
lr = LogisticRegression()
scores_2 = cross_validation.cross_val_score(lr, dataset_X, dataset_Y, cv = kFold)
print(scores_2)


