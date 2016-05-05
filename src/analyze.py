import re
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#This file reads data from polluters.txt and legitimate_users.txt and do analysis on the data

#Read data from polluters.txt and legitimate_users.txt
polluters = []
reader_1 = open("bots.txt", 'r')
for line in reader_1:
    tokens = [float(r) for r in re.split("[\t\n]", line) if r != ""]
    polluters.append(tokens)
reader_1.close()
print("data read from polluters.txt")
    
legitimate_users = []
reader_2 = open("humans.txt", 'r')
for line in reader_2:
    tokens = [float(r) for r in re.split("[\t\n]", line) if r != ""]
    legitimate_users.append(tokens)
reader_2.close()
print("data read from legitimate_users.txt")


#Combine 2 datasets for classification, dataset_Y is the target list
dataset_X = [r[1:] for r in polluters + legitimate_users]
dataset_Y = [0] * len(polluters) + [1] * len(legitimate_users)


#Plot graph of #followers / #followings to see the data distribution patterns
humans_patch = mpatches.Patch(color='green', label='Humans')
bots_patch = mpatches.Patch(color='blue', label='Bots')
plt.figure(figsize=(12, 10))
plt.plot([x[0] for x in dataset_X][len(polluters):], [x[1] for x in dataset_X][len(polluters):], "go")
plt.plot([x[0] for x in dataset_X][:len(polluters)], [x[1] for x in dataset_X][:len(polluters)], "bo")
plt.legend(handles=[bots_patch, humans_patch])
plt.axis([-10, 5000, -10, 40000])
plt.ylabel("Number of followers")
plt.xlabel("Number of followings")
plt.title("# followers vs. # followings graph (Bots overlapping)")
plt.show()

plt.figure(figsize=(12, 10))
plt.plot([x[0] for x in dataset_X][:len(polluters)], [x[1] for x in dataset_X][:len(polluters)], "bo")
plt.plot([x[0] for x in dataset_X][len(polluters):], [x[1] for x in dataset_X][len(polluters):], "go")
plt.legend(handles=[bots_patch, humans_patch])
plt.axis([-10, 5000, -10, 40000])
plt.ylabel("Number of followers")
plt.xlabel("Number of followings")
plt.title("# followers vs. # followings graph (Humans overlapping)")
plt.show()



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
i = 0
for d in data:
    sorted_dataset_X = list(np.sort(d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(d[len(polluters):]))
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
    sorted_dataset_X = list(np.sort(dataset_X_number_tweets_d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(dataset_X_number_tweets_d[len(polluters):]))
    yvals_2 = list(np.arange(len(sorted_data_2))/float(len(sorted_data_2)))
    plt.plot(sorted_dataset_X, yvals_1, 'b')
    plt.plot(sorted_data_2, yvals_2, 'g')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin-xmax*0.05, xmax+xmax*0.05, -0.05, 1.05])
    plt.xlabel("CDF - Number of tweets posted on " + weekdays[d] + "s")
    plt.show()
    
    dataset_X_ratio_tweets_d = [r[d] for r in dataset_X_ratio_tweets_weekdays]
    sorted_dataset_X = list(np.sort(dataset_X_ratio_tweets_d[:len(polluters)]))
    yvals_1 = list(np.arange(len(sorted_dataset_X))/float(len(sorted_dataset_X)))
    sorted_data_2 = list(np.sort(dataset_X_ratio_tweets_d[len(polluters):]))
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
"""