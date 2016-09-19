from pymongo import MongoClient
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import re
import os

#This file reads data from MongoDB and do analysis on the data

mClient = MongoClient()
db = mClient['new_data']
bots = [[r['friends_count'], r['followers_count']] for r in db['bots'].find({'active': True}, {'followers_count': 1, 'friends_count': 1})]
humans = [[r['friends_count'], r['followers_count']] for r in db['humans'].find({'active': True}, {'followers_count': 1, 'friends_count': 1})]
dataset_X = bots + humans
dataset_Y = [0] * len(bots) + [1] * len(humans)

#Plot graph of # followers vs. #friends, take either [bots, humans] or unlabeled_users as input (new users are not classified)
def plot_num_followers_vs_friends(labeled_users=None, unlabeled_users=None):
    if labeled_users != None:
        bots = [[r['friends_count'], r['followers_count']] for r in 
                  labeled_users[0].find({'friends_count': {'$exists': 1}, 
                                         'followers_count': {'$exists': 1}}, 
                                        {'friends_count': 1, 'followers_count': 1})]
        
        humans = [[r['friends_count'], r['followers_count']] for r in 
                    labeled_users[1].find({'friends_count': {'$exists': 1}, 
                                           'followers_count': {'$exists': 1}}, 
                                          {'friends_count': 1, 'followers_count': 1})]
        
        bots_patch = mpatches.Patch(color='blue', label='Bots')
        humans_patch = mpatches.Patch(color='green', label='Humans')
        
        plt.figure(figsize=(12, 10))
        plt.legend(handles=[bots_patch, humans_patch])
        plt.xlabel("Number of followings")
        plt.ylabel("Number of followers")
        plt.title("# followers vs. # followings graph for labeled users (Bots overlapping)")
        plt.plot([r[0] for r in bots], [r[1] for r in bots], 'go')
        plt.plot([r[0] for r in humans], [r[1] for r in humans], 'bo')
        plt.show()
        
        plt.figure(figsize=(12, 10))
        plt.legend(handles=[bots_patch, humans_patch])
        plt.xlabel("Number of followings")
        plt.ylabel("Number of followers")
        plt.title("# followers vs. # followings graph for labeled users (humans overlapping)")
        plt.plot([r[0] for r in humans], [r[1] for r in humans], 'bo')
        plt.plot([r[0] for r in bots], [r[1] for r in bots], 'go')
        plt.show()
        
    if unlabeled_users != None:
        users = [[r['friends_count'], r['followers_count']] for r in 
                   unlabeled_users.find({'friends_count': {'$exists': 1},
                                         'followers_count': {'$exists': 1}},
                                        {'friends_count': 1, 'followers_count': 1})]
        
        plt.figure(figsize=(12, 10))
        plt.xlabel("Number of followings")
        plt.ylabel("Number of followers")
        plt.title("# followers vs. # followings graph for unlabeled users")
        plt.plot([r[0] for r in users], [r[1] for r in users], 'bo')
        plt.show()

'''
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
i = 0
for d in data:
    sorted_dataset_X = list(np.sort(d[:len(bots)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(d[len(bots):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_dataset_X, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - " + labels[i])
    plt.show()
    i += 1

for d in list(range(7)):
    dataset_X_number_tweets_d = [r[d] for r in dataset_X_number_tweets_weekdays]
    sorted_dataset_X = list(np.sort(dataset_X_number_tweets_d[:len(bots)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(dataset_X_number_tweets_d[len(bots):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_dataset_X, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - Number of tweets posted on " + weekdays[d] + "s")
    plt.show()
    
    dataset_X_ratio_tweets_d = [r[d] for r in dataset_X_ratio_tweets_weekdays]
    sorted_dataset_X = list(np.sort(dataset_X_ratio_tweets_d[:len(bots)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(dataset_X_ratio_tweets_d[len(bots):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_dataset_X, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - Ratio of tweets posted on " + weekdays[d] + "s")
    plt.show()  


#Try Support Vector Classification with different kernels
#This only works when kernel='rbf', can't figure out why
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size = 0.8, random_state = 0)
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
'''

def main():
    mClient = MongoClient()
    old_data = [mClient['old_data']['bots'], mClient['old_data']['humans']]
    new_data = [mClient['new_data']['bots'], mClient['new_data']['humans']]
    new_users = mClient['new_data']['new_users']
    
    plot_num_followers_vs_friends(labeled_users=old_data)
    plot_num_followers_vs_friends(labeled_users=new_data)
    plot_num_followers_vs_friends(unlabeled_users=new_users)
    
    mClient.close()
    
    
if __name__ == '__main__':
    main()