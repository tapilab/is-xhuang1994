from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_curve, accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.cross_validation import KFold
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt

#This file reads data from polluters.txt and legitimate_users.txt (old data) or from Mongodb (recent data) and classify the records

num_decimals = 4
#Read data from Mongodb, this function only works for new_data
def get_data_new(ids, collection):
    users = []
    for id in ids:
        user = list(collection.find({'id': id}))[0]
        #basic_data: 9 features in total including tweets_sim and excluding id
        basic_data = [user['id'], 
                      len(user['name']), 
                      len(user['screen_name']), 
                      user['friends_count'], 
                      user['followers_count'], 
                      user['friends_count']/user['followers_count'] if user['followers_count'] != 0 else 1e9, 
                      
                      user['statuses_count']/user['followers_count'] if user['followers_count'] != 0 else 1e9,
                      user['statuses_count']/user['friends_count'] if user['friends_count'] != 0 else 1e9,
                      
                      user['statuses_count'], 
                      len(user['description']), 
                      int(user['goe_enabled']), 
                      np.std(user['tweets_sim'])] \
                      + user['tweets_sim']
        
        #timeline_data: 14+6+6 = 26 features
        timeline_data = []
        posts = [r for r in user['timeline'] if r['is_rt'] == False and r['is_reply'] == False]
        if len(posts) == 0:
            continue
        
        urls, mentions, hashtags = [], [], []
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]))
        for d in list(range(7)):
            timeline_data.append(len([r for r in posts if r['weekday'] == d]) / (len(posts) if len(posts) != 0 else 1e-15))
        
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


def plot_curve(a, b, name, w=0.5):
    
    plt.plot(a, b, 'b')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    if name == 'ROC':
        plt.title('Receiver Operating Characteristic\ncost of bots = %g' % w)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    elif name == 'PR':
        plt.title('Precision-Recall\ncost of bots = %g' % w)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
    plt.show()




#Do cross validation manually to get the f1 score and confusion matrix
#Also count misclassified bots and humans respectively, and calculate the accuracies respectively
def cross_val(data_x, data_y, classifier, kFold, b_cost=1, h_cost=1, w=0.5):
    e_h, e_b = 0, 0
    y_tests, pred_probas = [], []
    
    for train_index, test_index in kFold:
        data_x_, data_y_ = np.array(data_x), np.array(data_y)
        X_train, X_test = list(data_x_[train_index]), list(data_x_[test_index])
        y_train, y_test = list(data_y_[train_index]), list(data_y_[test_index])
        classifier.fit(X_train, y_train)
        pred_proba = [r[0] for r in classifier.predict_proba(X_test)]
        y_tests += y_test
        pred_probas += pred_proba
    
    predictions = [0 if p*b_cost > (1-p)*h_cost else 1 for p in pred_probas]
    roc_auc = roc_auc_score(y_tests, pred_probas)
    total_acc = accuracy_score(y_tests, predictions)
    precision, recall, thresholds = precision_recall_curve(y_tests, pred_probas, pos_label=0)
    fpr, tpr, thresholds = roc_curve(y_tests, pred_probas, pos_label=0)
    precision_bots = precision_score(y_tests, predictions, pos_label = 0)
    precision_humans = precision_score(y_tests, predictions, pos_label = 1)
    recall_bots = recall_score(y_tests, predictions, pos_label = 0)
    recall_humans = recall_score(y_tests, predictions, pos_label = 1)
    f1_bots = f1_score(y_tests, predictions, pos_label = 0)
    f1_humans = f1_score(y_tests, predictions, pos_label = 1)
    conf_matrix = np.matrix(list(confusion_matrix(y_tests, predictions)))
    
    #plot_curve(fpr, tpr, 'ROC', w)
    plot_curve(recall, precision, 'PR', w)
    
    return [total_acc, precision_bots, precision_humans, recall_bots, recall_humans, f1_bots, f1_humans, roc_auc, conf_matrix]

#FYI, the labels for confusion matrix are:
#         classified as
#         bots     humans
# bots
# humans


def main():
    mClient = MongoClient()
    '''
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
    
    print(rf.feature_importances_)
    '''
    new = mClient['new_data']['new_users']
    new_bots = [r['id'] for r in new.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}, 'label': 0}, {'id': 1})]
    new_bots_data = get_data_new(new_bots, new)
    new_humans = [r['id'] for r in new.find({'timeline': {'$exists': 1, '$not': {'$size': 0}}, 'label': 1}, {'id': 1})]
    new_humans_data = get_data_new(new_humans, new)
    print("\n\nData read for bots and humans from new_users")
    mClient.close()
    
    dataset_X = [r[1:] for r in new_bots_data + new_humans_data]
    dataset_Y = [0] * len(new_bots_data) + [1] * len(new_humans_data)
    
    print("\n%d instances, where %g are bots\n" % (len(dataset_X), dataset_Y.count(0)/len(dataset_Y)))
    
    kFold = KFold(n = len(dataset_X), n_folds = 4, shuffle = True, random_state=0)
    rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50, class_weight = {0: 5})
    
    result_cv = cross_val(dataset_X, dataset_Y, rf, kFold)
    print("Total accuracy: %0.4f \n" % result_cv[0])
    print("Precision: \nBots: %0.4f \nHumans: %0.4f\n" % (result_cv[1], result_cv[2]))
    print("Recall: \nBots: %0.4f \nHumans: %0.4f\n" % (result_cv[3], result_cv[4]))
    print("f1 score: \nBots: %g \nHumans: %g\n" % tuple([round(r, num_decimals) for r in result_cv[5:7]]))
    print("ROC_AUC score: \n", result_cv[7], '\n')
    print(result_cv[8], '\n')
    print(rf.feature_importances_)
    
    
    for i in [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]:
        print("\nWith cost of bots = %g:\n" % i)
        rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50, class_weight = {0: i, 1: 1-i}, random_state = 0)
        result_cv = cross_val(dataset_X, dataset_Y, rf, kFold, w=i)
        #rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50)
        #result_cv = cross_val(dataset_X, dataset_Y, rf, kFold, i, 1-i, i)
        print("Precision: %0.4f" % result_cv[1])
        print("Recall: %0.4f" % result_cv[3])
    
    
if __name__ == '__main__':
    main()