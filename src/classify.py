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


#This file reads data from polluters.txt and legitimate_users.txt and classify the records


#Read data from polluters.txt and legitimate_users.txt
polluters = []
reader_1 = open("polluters.txt", 'r')
for line in reader_1:
    tokens = [float(r) for r in re.split("[\t\n]", line)[:23]]
    polluters.append(tokens)
reader_1.close()
print("data read from polluters.txt")
    
legitimate_users = []
reader_2 = open("legitimate_users.txt", 'r')
for line in reader_2:
    tokens = [float(r) for r in re.split("[\t\n]", line)[:23]]
    legitimate_users.append(tokens)
reader_2.close()
print("data read from legitimate_users.txt")

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


#Combine 2 datasets for classification, dataset_Y is the target list
#the data will be shuffled before classify
dataset_X = polluters + legitimate_users
dataset_Y = [0] * len(polluters) + [1] * len(legitimate_users)

data_1 = [r[1:] for r in dataset_X]


"""
#Plot distribution graphs for each feature
labels = ["Number of followings", "Number of follwers", "Number of tweets", "Length of screen name", "Length of self description", "Standard deviation difference", "Lag1 autocorrelation", "Ratio of urls over tweets"]
data = [dataset_X_numberOfFollowings, dataset_X_numberOfFollowers, dataset_X_numberOfTweets, dataset_X_lengthOfScreenName, dataset_X_lengthOfDescriptionInUserProfile, dataset_X_standard_deviation_diff, dataset_X_lag1_autocorrelation, dataset_X_ratio_urls_tweets]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

i = 0
for d in data:
    plt.plot(d, dataset_Y, 'bo')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel(labels[i])
    plt.show()
    i += 1


for d in list(range(7)):
    dataset_X_number_tweets_d = [r[d] for r in dataset_X_number_tweets_weekdays]
    plt.plot(dataset_X_number_tweets_d, dataset_Y, 'bo')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("Number of tweets posted on " + weekdays[d] + "s")
    plt.show()
    dataset_X_ratio_tweets_d = [r[d] for r in dataset_X_ratio_tweets_weekdays]
    plt.plot(dataset_X_number_tweets_d, dataset_Y, 'bo')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("Ratio of tweets posted on " + weekdays[d] + "s")
    plt.show()


#Plot CDF functions for each feature
#This method doesn't seem to work correctly, another method is used below
#num_bins = 20
#counts_1, bin_edges_1 = np.histogram(dataset_X_numberOfFollowings[:len(polluters)], bins = num_bins, normed = True)
#cdf_1 = np.cumsum(counts_1)
#counts_2, bin_edges_2 = np.histogram(dataset_X_numberOfFollowings[len(polluters):], bins = num_bins, normed = True)
#cdf_2 = np.cumsum(counts_2)
#plt.plot(bin_edges_1[1:] + bin_edges_2[1:], cdf_1 + cdf_2)
#plt.show()


#Second alternative method to plot CDF functions
i = 0
for d in data:
    sorted_data_1 = list(np.sort(d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_data_1))/float(len(sorted_data_1)))
    sorted_data_2 = list(np.sort(d[len(polluters):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_data_1, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - " + labels[i])
    plt.show()
    i += 1

for d in list(range(7)):
    dataset_X_number_tweets_d = [r[d] for r in dataset_X_number_tweets_weekdays]
    sorted_data_1 = list(np.sort(dataset_X_number_tweets_d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_data_1))/float(len(sorted_data_1)))
    sorted_data_2 = list(np.sort(dataset_X_number_tweets_d[len(polluters):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_data_1, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - Number of tweets posted on " + weekdays[d] + "s")
    plt.show()
    
    dataset_X_ratio_tweets_d = [r[d] for r in dataset_X_ratio_tweets_weekdays]
    sorted_data_1 = list(np.sort(dataset_X_ratio_tweets_d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_data_1))/float(len(sorted_data_1)))
    sorted_data_2 = list(np.sort(dataset_X_ratio_tweets_d[len(polluters):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_data_1, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - Ratio of tweets posted on " + weekdays[d] + "s")
    plt.show()

#Plot graph of #followers / #followings to see the data distribution patterns
plt.plot([x[1] for x in dataset_X][len(polluters):], [x[2] for x in dataset_X][len(polluters):], "go")
plt.plot([x[1] for x in dataset_X][:len(polluters)], [x[2] for x in dataset_X][:len(polluters)], "bo")
plt.axis([-10, 5000, -10, 40000])
plt.show()    


#Try Support Vector Classification with different kernels
#This only works when kernel='rbf', can't figure out why
X_train, X_test, y_train, y_test = train_test_split(data_1, dataset_Y, test_size = 0.8, random_state = 0)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
i = 0
e = 0
while i < len(prediction):
    if prediction[i] != y_test[i]:
        e += 1
    i += 1
print("# errors (total: %d): " % len(X_test))
print("rbf: ", e)
"""

#Need KFold iterator since method "cross_val_score" doesn't provide shuffling function
kFold = KFold(n = len(data_1), n_folds = 10, shuffle = True)

#Build the classifier and do classification based on: all numeric features & each numeric feature
lr = LogisticRegression()
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50)
"""
#Randomly separate data into training (80%) and test (20%) sets
#Call lr.fit() and lr.predict() and compare the prediction versus real classes
#write instances that are classified wrongly into error.txt
fwriter = open("error.txt", 'w')
X_train, X_test, y_train, y_test = train_test_split(data_1, dataset_Y, test_size = 0.2, random_state = 0)
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
i = 0
e = 0
while i < len(X_test):
    if prediction[i] != y_test[i]:
        e += 1
        s = "\t".join([str(r) for r in [round(a, 1) for a in X_test[i]]])
        fwriter.write(s + "\n\n")
    i += 1
fwriter.close()
print("errors written to error.txt")
print("# errors: ", e)
print("# total instances: ", len(X_test))
"""


#Separate data_1 into bins by #followings and try classification
#Drop some outliers based on the #followers / #followings distribution graph
i = 0
while i < len(data_1):
    data_1[i] = data_1[i][:4] + data_1[i][5:]
    if data_1[i][1] == 0:
        data_1[i].append(0)
    else:
        data_1[i].append(data_1[i][0]/data_1[i][1])
    i += 1

data_x = []
data_y = []
outliers_x = []
outliers_y = []
i = 0
while i < len(data_1):
    if data_1[i][0] < 500:
        if data_1[i][1] > 1500:
            data_x.append(data_1[i] + [1, 0, 0])
            data_y.append(dataset_Y[i])
        else:
            data_x.append(data_1[i] + [0, 1, 0])
            data_y.append(dataset_Y[i])
    elif data_1[i][1] < data_1[i][0]*0.9+10000:
        #The #followers/#followings ratio is below the generalized line
        if data_1[i][0] < 2005:
            data_x.append(data_1[i] + [0, 1, 0])
            data_y.append(dataset_Y[i])
        else:
            data_x.append(data_1[i] + [0, 0, 1])
            data_y.append(dataset_Y[i])
    else:
        outliers_x.append(data_1[i])
        outliers_y.append(dataset_Y[i])
    i += 1

print("%d outliers are dropped" % len(outliers_x))


#scale the data
max_values = np.matrix(data_x).mean(0).tolist()[0][:-1] + [1]
data_x = [[x/y for x, y in list(zip(z, max_values))] for z in data_x]
print("data scaled")
print("%d instances, where %g are bots" % (len(data_x), data_y.count(0)/len(data_y)))

kFold_1 = KFold(n = len(data_x), n_folds = 10, shuffle = True)
e_h = 0
e_b = 0
predictions = []
y_tests = []
conf_matrix = np.matrix([[0, 0], [0, 0]])
score = np.mean(cross_validation.cross_val_score(rf, data_x, data_y, cv = kFold_1))
for train_index, test_index in kFold_1:
    data_x, data_y = np.array(data_x), np.array(data_y)
    X_train, X_test = list(data_x[train_index]), list(data_x[test_index])
    y_train, y_test = list(data_y[train_index]), list(data_y[test_index])
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


"""
#number of iterations (6)
n = list(range(6))

print("\n****************************************************\n")
print("Classification accuracy (10 fold cross validation iterated %d times):" % len(n))

scores_all_numeric_features = 0
for i in n:
    scores_all_numeric_features += np.mean(cross_validation.cross_val_score(lr, data_1, dataset_Y, cv = kFold))
print("All numeric features: ", scores_all_numeric_features / len(n))

scores_numberOfFollowings = 0
for i in n:
    scores_numberOfFollowings += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfFollowings, dataset_Y, cv = kFold))
print("Number of followings: ", scores_numberOfFollowings / len(n))

scores_numberOfFollowers = 0
for i in n:
    scores_numberOfFollowers += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfFollowers, dataset_Y, cv = kFold))
print("Number of followers: ", scores_numberOfFollowers / len(n))

scores_numberOfTweets = 0
for i in n:
    scores_numberOfTweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_numberOfTweets, dataset_Y, cv = kFold))
print("Number of tweets: ", scores_numberOfTweets / len(n))

scores_lengthOfScreenName = 0
for i in n:
    scores_lengthOfScreenName += np.mean(cross_validation.cross_val_score(lr, dataset_X_lengthOfScreenName, dataset_Y, cv = kFold))
print("Length of screen name: ", scores_lengthOfScreenName / len(n))

scores_lengthOfDescriptionInUserProfile = 0
for i in n:
    scores_lengthOfDescriptionInUserProfile += np.mean(cross_validation.cross_val_score(lr, dataset_X_lengthOfDescriptionInUserProfile, dataset_Y, cv = kFold))
print("Length of description in user profile: ", scores_lengthOfDescriptionInUserProfile / len(n))

scores_standard_deviation_diff = 0
for i in n:
    scores_standard_deviation_diff += np.mean(cross_validation.cross_val_score(lr, dataset_X_standard_deviation_diff, dataset_Y, cv = kFold))
print("standard deviation diff: ", scores_standard_deviation_diff / len(n))

scores_lag1_autocorrelation = 0
for i in n:
    scores_lag1_autocorrelation += np.mean(cross_validation.cross_val_score(lr, dataset_X_lag1_autocorrelation, dataset_Y, cv = kFold))
print("lag1 autocorrelation: ", scores_lag1_autocorrelation / len(n))

scores_number_tweets_weekdays = 0
for i in n:
    scores_number_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_number_tweets_weekdays, dataset_Y, cv = kFold))
print("Number of tweets each weekday: ", scores_number_tweets_weekdays / len(n))

scores_ratio_tweets_weekdays = 0
for i in n:
    scores_ratio_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_ratio_tweets_weekdays, dataset_Y, cv = kFold))
print("Ratio of tweets each weekday: ", scores_ratio_tweets_weekdays / len(n))

scores_ratio_urls_tweets = 0
for i in n:
    scores_ratio_urls_tweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_ratio_urls_tweets, dataset_Y, cv = kFold))
print("Ratio of urls over tweets: ", scores_ratio_urls_tweets / len(n))

print("\n****************************************************\n")

dataset_X_no_numberOfFollowings = [x[2:] for x in dataset_X]
scores_no_numberOfFollowings = 0
for i in n:
    scores_no_numberOfFollowings += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfFollowings, dataset_Y, cv = kFold))
print("No number of followings: ", scores_no_numberOfFollowings / len(n))

dataset_X_no_numberOfFollowers = [[x[1]] + x[3:] for x in dataset_X]
scores_no_numberOfFollowers = 0
for i in n:
    scores_no_numberOfFollowers += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfFollowers, dataset_Y, cv = kFold))
print("No number of followers: ", scores_no_numberOfFollowers / len(n))

dataset_X_no_numberOfTweets = [x[1:3] + x[4:] for x in dataset_X]
scores_no_numberOfTweets = 0
for i in n:
    scores_no_numberOfTweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_numberOfTweets, dataset_Y, cv = kFold))
print("No number of tweets: ", scores_no_numberOfTweets / len(n))

dataset_X_no_lengthOfScreenName = [x[1:4] + x[5:] for x in dataset_X]
scores_no_lengthOfScreenName = 0
for i in n:
    scores_no_lengthOfScreenName += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lengthOfScreenName, dataset_Y, cv = kFold))
print("No length of screen name: ", scores_no_lengthOfScreenName / len(n))

dataset_X_no_lengthOfDescriptionInUserProfile = [x[1:5] + x[6:] for x in dataset_X]
scores_no_lengthOfDescriptionInUserProfile = 0
for i in n:
    scores_no_lengthOfDescriptionInUserProfile += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lengthOfDescriptionInUserProfile, dataset_Y, cv = kFold))
print("No length of description in user profile: ", scores_no_lengthOfDescriptionInUserProfile / len(n))

dataset_X_no_standard_deviation_diff = [r[1:6] + r[7:] for r in dataset_X]
scores_no_standard_deviation_diff = 0
for i in n:
    scores_no_standard_deviation_diff += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_standard_deviation_diff, dataset_Y, cv = kFold))
print("no standard deviation diff: ", scores_no_standard_deviation_diff / len(n))

dataset_X_no_lag1_autocorrelation = [r[1:7] + r[8:] for r in dataset_X]
scores_no_lag1_autocorrelation = 0
for i in n:
    scores_no_lag1_autocorrelation += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_lag1_autocorrelation, dataset_Y, cv = kFold))
print("no lag1 autocorrelation: ", scores_no_lag1_autocorrelation / len(n))

dataset_X_no_number_tweets_weekdays = [r[1:9] + r[15:] for r in dataset_X]
scores_no_number_tweets_weekdays = 0
for i in n:
    scores_no_number_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_number_tweets_weekdays, dataset_Y, cv = kFold))
print("no number of tweets each weekday: ", scores_no_number_tweets_weekdays / len(n))

dataset_X_no_ratio_tweets_weekdays = [r[1:15] + [r[22]] for r in dataset_X]
scores_no_ratio_tweets_weekdays = 0
for i in n:
    scores_no_ratio_tweets_weekdays += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_ratio_tweets_weekdays, dataset_Y, cv = kFold))
print("no ratio of tweets each weekday: ", scores_no_ratio_tweets_weekdays / len(n))

dataset_X_no_ratio_urls_tweets = [r[1:22] for r in dataset_X]
scores_no_ratio_urls_tweets = 0
for i in n:
    scores_no_ratio_urls_tweets += np.mean(cross_validation.cross_val_score(lr, dataset_X_no_ratio_urls_tweets, dataset_Y, cv = kFold))
print("no ratio of urls over tweets: ", scores_no_ratio_urls_tweets / len(n))

print("\n****************************************************\n")
print("Coefficients: \n")

lr.fit(data_1, dataset_Y)
coef = lr.coef_[0]
print("Number of followings: ", coef[0])
print("Number of followers: ", coef[1])
print("Number of tweets: ", coef[2])
print("Length of screen name: ", coef[3])
print("Length of description in user profile: ", coef[4])
print("Standard deviation differences: ", coef[5])
print("Lag one autocorrelation: ", coef[6])
print("Number of tweets each weekday (avg): ", np.mean(coef[7:14]))
print("Ratio of tweets each weekday (avg): ", np.mean(coef[14:21]))
print("Ratio of urls over tweets: ", coef[21])
"""
