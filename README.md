
## Problem

The task is to develop a classification model to Distinguish bots from humans on Twitter based on available data. Bots, are users on social media (e.g., Twitter) that are manipulated by computer programs. Bots can post, retweet, and even reply automatically. They are usually created for some purpose like hyping, advertising/promoting, and news/rumors spreading.

## Related work

On this topic several papers are found that uses the same method: find and collect numeric features of each user, and feed the data to a classification algorithm. Highest classfication accuracy result have been reported is around 99%.  

Related papers:  
Who is Tweeting on Twitter: Human, Bot, or Cyborg?  
Detecting and Analyzing Automated Activity on Twitter  
Detecting Spammers on Twitter  
Uncovering Social Spammers: Social Honeypots + Machine Learning  
A Long-Term Study of Content Polluters on Twitter  

## Data source

I used the social honeypot dataset[1] in the beginning. The classification model was developed based on papers I read and tested with data from honeypot dataset. My initial model uses Random Forest Classifier with default parameters from sk-learn. The cross validation result is:
![Image](../master/src/graphs/Result.png?raw=true)  
Figure 1: Cross validation results with old data of old users

As is suggested by Dr. Culotta, the performance is a little "too good" and it is mentioned in the paper that there are some threshold used in collecting users so it could make the result not generalizable. To reveal the problem I tried plotting graphs with the user data. Here's some funny graphs I've got:
![Image](../master/src/graphs/old_data.png?raw=true)
Figure 2: Funny graphs based on old data of old users

As you can see there's a clear boundry for the ratio of # followers on # followings (about 0.9), which doesn't make any sense unless it is a threshold used in collecting users and their data. So the data we've got in social honeypot dataset is highly biased.

To prove this I collected the most recent data of same users and tested my model with it. The cross validation result and the graph both become a little bit "normal" this time:
![Image](../master/src/graphs/Result_new.png?raw=true)  
Figure 3: Cross validation results with new data of old users
![Image](../master/src/graphs/new_data.png?raw=true)
Figure 4: Graphs based on new_data of old users.

The performace is a little bit worse which is expected. And in the graph some of the users appear below the boundry, but it still looks biased as most of the users are still above the boundry.

After I realize the problem I "randomly" collected 1100 users on Twitter through Streaming API – the first 1100 who posted anything after I started collecting process were collected. It's not completely random but we might still draw some generalizable conclusions from it. I then spent a few weeks to manually label 523 of the users, in which 54 are bots and 469 are humans. I also plotted the same graph as above and now it looks completely unbiased:
![Image](../master/src/graphs/new_users.png?raw=true)
Figure 5: Graph based on new_users

## Features

For each user, 45 features were extracted from available data, the 22 most important features are listed below (ordered by importance)

 Number of replies over number of posts
 3-gram Jaccard similarity between each two tweets
 Number of retweets over number of posts
 Average number of time each tweet is retweeted
 Number of tweets posted on Sundays
 Number of tweets posted on Saturdays
 1-gram Jaccard similarity between each two tweets
 Number of statuses over number of followers
 Number of friends over number of followers
 Number of unique mentions over number of posts
 Number of hashtags over number of posts
 Number of friends
 Number of statuses over number of friends
 Number of statuses
 2-gram Jaccard similarity between each two tweets
 Number of tweets posted on Fridays
 Length of user description
 Number of unique urls over number of tweets
 Number of mentions over number tweets
 Number of tweets posted on Saturdays
 Length of username
 Ratio of unique hashtags over tweets

## Evaluation
### Performance Measuring

I used precision and recall of bots to measure the performance because the classes are imbalanced. I found that I can’t guarantee both precision and recall at same time. There’s always a trade-off between them. But by defining a false negative cost for bots, I can guarantee either precision or recall with the following decision formulas: (p is the probability of being bot)
p * cost > 1 – p => Bot
p * cost < 1 – p => Human
In this case, by varying the cost of bots, we can have a high precision or a high recall. If someone wants to use my model to find and analyze some bots, he would like to have a high precision, so the users returned are very likely to be bots. If someone wants to use my model to find as many bots as possible from a set of users, he would like to have a high recall. The precision would not be very high and he still needs to identify each user, but the workload would be reduced a lot.
Since the precision and recall changes with bots-cost, I have to plot graph of precision and recall to show the model performance.

### Classifiers

Result by Logistic Regression with penalty = L2, C = 1, and bots_cost = 1:

Precision: 0.7540 
Recall: 0.5080 
AUC of ROC: 0.7454

Result by Random Forest Classifier with default parameters and bots_cost = 1:

Precision: 0.7587 
Recall: 0.5482 
AUC of ROC: 0.7654

It turns out that Random Forest Classifier has better performance than Logistic Regression.
The change of precision and recall with max_depth is shown in Figure 6. The best value for max_depth, according to the figure, is 7.
![Image](../master/src/graphs/max_depth.png?raw=true)
Figure 6: Precision, recall, and AUC of ROC vs. max_depth

The change of precision and recall with min_samples_leaf is shown in Figure 7. The best value for min_samples_leaf, according to the figure, is 6.
![Image](../master/src/graphs/min_samples_leaf.png?raw=true)
Figure 7: Precision, recall, and AUC of ROC vs. min_samples_leaf

So the classification model I selected is Random Forest Classifier with max_depth = 7 and min_samples_leaf = 6.

## Performance

I used 4-fold cross-validation with different bots_cost to show the performance of my model. The result of my selected classification model is shown in Figure 8. It turns out that precision or recall can be very close to or even equal to 1, by varying the bots_cost. As bots_cost increases, precision of bots decreases and recall of bots increases, dramatically. I think it proves that my classification model is effective.
![Image](../master/src/graphs/final_result.png?raw=true)
Figure 8: Precision, recall, and AUC of ROC vs. bots_cost using selected classification model.

## Something Funny

We know that bots and humans are distinguished by their behaviors (e.g., autonomous behavior indicates bots). But to build a classification model, there may not always be a good feature to represent a behavior. If a user is a phony fan, it tends to have many more followers than friends. This behavior can be represented as number of followers over number of friends. But if a user keeps sharing articles from same website and also post some words that are duplicated (exactly same words can be found from elsewhere), it might be considered as a bot. But such behavior cannot be easily represented by a feature. That sometimes makes it hard to distinguish bots from humans.

Here's something interesting: some users may have very normal stats and normal actions like human users, but I labeled them as bots for some other reason, which can’t be well represented by any feature. For example, some users have used language processing technique:

![Image](../master/src/graphs/funny1.png?raw=true)
![Image](../master/src/graphs/funny2.png?raw=true)
![Image](../master/src/graphs/funny3.png?raw=true)
Figures 9-10: Examples of users using language processing technique

In the first two figures the user is clearly a sharer bot that uses language processing, by replacing words with other alternatives. In the second figure the user add pound signs before some words to make them hashtags, which don't make any sense unless it's autonomous behavior.

## Conclusion

From the result I can conclude that I have a right direction about the problem, though my result is not very good. Some bots are very hard to be distinguished from humans by my model, and some users I still have trouble labeling them. Yet my model does work for some special cases. It can achieve either high precision or high recall. So I think I’m in the right direction.

There are still a lot of things can be done to improve the performance, such as digging more good features and maybe do some analysis on the relation graph of the users. Also I can expand my data – I only have 523 instances for training because I can’t find a good dataset for this topic.


## References

[1] K. Lee, B. Eoff, and J. Caverlee. Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter. In Proceeding of the 5th International AAAI Conference on Weblogs and Social Media (ICWSM), Barcelona, July 2011. (Bibtex)
