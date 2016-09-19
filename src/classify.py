from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.cross_validation import KFold
from pymongo import MongoClient
import numpy as np

#This file reads data from polluters.txt and legitimate_users.txt (old data) or from Mongodb (recent data) and classify the records

num_decimals = 4
#Read data from Mongodb, this function only works for new_data
def get_data_new(ids, collection, old_data=False):
    users = []
    for id in ids:
        user = list(collection.find({'id': id}))[0]
        #basic_data: 9 features in total including tweets_sim and excluding id
        basic_data = [user['id'], 
                      len(user['name']), 
                      len(user['screen_name']), 
                      user['friends_count'], 
                      user['followers_count'], 
                      user['friends_count']/user['followers_count'] if user['followers_count'] != 0 else 0, 
                      user['statuses_count'], 
                      len(user['description']), 
                      int(user['goe_enabled'])]
                      #+ user['tweets_sim']
        
        #timeline_data: 14+6+6 = 26 features
        timeline_data = []
        posts = [r for r in user['timeline'] if r['is_rt'] == False and r['is_reply'] == False]
        if len(posts) == 0:
            continue
        
        urls, mentions, hashtags = [], [], []
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]))
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]) / (len(posts) if len(posts) != 0 else 1))
        
        for post in posts:
            urls += post['urls']
            mentions += post['mentions']
            hashtags += post['hashtags']
        timeline_data += [len(urls)/len(posts), 
                          len(set(urls))/len(posts), 
                          len(mentions)/len(posts), 
                          len(set(mentions))/len(posts), 
                          len(hashtags)/len(posts), 
                          len(set(hashtags))/len(posts)]
                          
        avg_rts = sum([r['rt_count'] for r in user['timeline']]) / len(user['timeline'])
        end_with_phu = len([r for r in user['timeline'] if r['is_rt'] == False and r['is_reply'] == False and r['end_with_phu'] == True]) / len(user['timeline'])
        num_sources = len(set([r['source'] for r in user['timeline']]))
        ratio_coordinate = len([r for r in posts if r['coordinated'] == True]) / (len(posts) if len(posts) != 0 else 1)
        ratio_rt = len([r for r in user['timeline'] if r['is_rt'] == True]) / len(user['timeline'])
        ratio_reply = len([r for r in user['timeline'] if r['is_reply'] == True]) / len(user['timeline'])
        timeline_data += [avg_rts, 
                          end_with_phu, 
                          num_sources, 
                          ratio_coordinate, 
                          ratio_rt, 
                          ratio_reply]
        
        #each user sample contains 9+26=35 features
        users.append(basic_data + timeline_data)
    return users


#Scale each feature of the data with its mean or maximum value
def scale(dataset, with_mean=False, with_max=False):
    if with_mean == True:
        max_values = np.matrix(dataset).mean(0).tolist()[0]
    if with_max == True:
        max_values = np.matrix(dataset).max(0).tolist()[0]
    dataset = [[x/y for x, y in list(zip(z, max_values))] for z in dataset]


#Do cross validation manually to get the f1 score and confusion matrix
#Also count misclassified bots and humans respectively, and calculate the accuracies respectively
def cross_val(data_x, data_y, classifier, kFold):
    e_h, e_b = 0, 0
    predictions, y_tests = [], []

    for train_index, test_index in kFold:
        data_x_, data_y_ = np.array(data_x), np.array(data_y)
        X_train, X_test = list(data_x_[train_index]), list(data_x_[test_index])
        y_train, y_test = list(data_y_[train_index]), list(data_y_[test_index])
        classifier.fit(X_train, y_train)
        prediction = list(classifier.predict(X_test))
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
    
    total_acc = 1-(e_h+e_b)/len(y_tests)
    bot_acc = 1 - e_b/y_tests.count(0)
    human_acc = 1 - e_h/y_tests.count(1)
    f1_bots = f1_score(y_tests, predictions, pos_label = 0)
    f1_humans = f1_score(y_tests, predictions, pos_label = 1)
    conf_matrix = np.matrix(list(confusion_matrix(y_tests, predictions)))
    
    return [total_acc, bot_acc, human_acc, f1_bots, f1_humans, conf_matrix]

#FYI, the labels for confusion matrix are:
#         classified as
#         bots     humans
# bots
# humans


def main():
    mClient = MongoClient()
    
    bots = mClient['new_data']['bots']
    bots_id = [r['id'] for r in bots.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}, 'lang': 'en'}, {'id': 1})]
    bots_data = get_data_new(bots_id, bots)
    humans = mClient['new_data']['humans']
    humans_id = [r['id'] for r in humans.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}, 'lang': 'en'}, {'id': 1})]
    humans_data = get_data_new(humans_id, humans)
    print("Data read for bots and humans from new_data")
    
    dataset_X = [r[1:] for r in bots_data + humans_data]
    dataset_Y = [0] * len(bots_data) + [1] * len(humans_data)
    
    scale(dataset_X, with_max=True)
    print("Data scaled")
    print("\n%d instances, where %g are bots\n" % (len(dataset_X), dataset_Y.count(0)/len(dataset_Y)))
    
    kFold = KFold(n = len(dataset_X), n_folds = 10, shuffle = True)
    rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50)
    
    result_cv = cross_val(dataset_X, dataset_Y, rf, kFold)
    print("Total accuracy: %g%% \nBots: %g%% \nHumans: %g%%\n" % tuple([round(r, num_decimals)*100 for r in result_cv[:3]]))
    print("f1 score: \nBots: %g%% \nHumans: %g%%\n" % tuple([round(r, num_decimals)*100 for r in result_cv[3:5]]))
    print(result_cv[5])
    
    
    new = mClient['new_data']['new_users']
    new_id = [r['id'] for r in new.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}, 'lang': 'en'}, {'id': 1})]
    new_data = get_data_new(new_id, new)
    print("\n\nData read for new users from new_users")
    
    dataset_new = [r[1:] for r in new_data]
    
    scale(dataset_new, with_mean=True)
    print("Data scaled")
    
    rf.fit(dataset_X, dataset_Y)
    prediction = list(rf.predict(dataset_new))
    print("\n%d users in total\n%d are classified as bots \n%d are classified as humans\n" % (len(prediction), prediction.count(0), prediction.count(1)))


if __name__ == '__main__':
    main()