
## Problem

Develop a method to distinguish between humans and bots based on the data available.

## Research questions

1. What kind of data should I use
2. What algorithm should I use for classification
3. How do I evaluate the classification model

## Related work

On this topic several papers are found that uses the same method: find and collect numeric features of each user, and feed the data to a classification algorithm. Highest classfication accuracy result have been reported is around 99%.  

Related papers:  
Who is Tweeting on Twitter: Human, Bot, or Cyborg?  
Detecting and Analyzing Automated Activity on Twitter  
Detecting Spammers on Twitter  
Uncovering Social Spammers: Social Honeypots + Machine Learning  
A Long-Term Study of Content Polluters on Twitter  

## Data

The data I'm using are extracted from honeypot dataset.

For each user, I have these data fed to classifier:
[# followings, # followers, # tweets, length of description in user profile, standard deviation of # follwings, standard deviation of differences of # follwings (the change rate of # followings), lag1 autocorrelation of # follwings (the change rate of # followings), # tweets posted on each weekday, ratio of tweets posted on each weekday, ratio of urls in tweets, ratio of unique urls in tweets, ratio of @'s in tweets, ratio of unique @'s in tweets, ratio of hashtags in tweets, ratio of unique hashtags in tweets]

Besides, length of screen name was also used as a feature in the beginning, but it was found to have a negative effect on the accuracy results and was removed. 

In addition, I plotted a graph of # followers vs. # followings and found an interesting distribution patter of the users. Based on the graph, I divided the users into 3 bins, with a few marked as outliers (See Figure 1-4). And for each user, there are 3 more features each is a 0 or 1 that represents which bin it belongs to.

![Image](../master/src/graphs/graph-bots-overlapping.png?raw=true)  
Figure 1: Bots overlapping (# followers vs. # followings graph. Blue points are bots and green points are humans)  
![Image](../master/src/graphs/graph-humans-overlapping.png?raw=true)  
Figure 2: Humans overlapping  
![Image](../master/src/graphs/graph-zoomed-in.png?raw=true)  
Figure 3: Figure 2 Zoomed in  
![Image](../master/src/graphs/graph-clustered.png?raw=true)  
Figure 4: Clustering users  

## Methods

I collect all the data mentioned above for every user, scale each feature by its maximum value, and feed the data to a classifier (random forest is used now), and do a 10-fold cross validation to see the results (total accuracy, respective accuracy for humans and bots, f1 socres for humans and bots, and confusion matrix)

## Results

Here is a screenshot of accuracy results:  
![Image](../master/src/graphs/Result.png?raw=true)  
Figure 5: Results for classification  

## Conclusions / Future Work

The results show a high classification accuracy -- 94% in total -- while the highest accuracy reported in the papers is over 99%. There could be some criterions in collecting the bot and human users that are not mentioned in the papers. I have several plans for Summer vacation and Fall semester:

1. Collect domain names of urls collected from recent tweets of each user I have. Find a way to rank the domains and adding this as an additional feature to my current model and test for accuracy. This can be achieved in doing work 2.
2. Recollect all the features needed for each user. Put data into my current model and test for accuracy (also need to replot the # followings vs. # followers graph and see the users distribution patterns)
3. Find more users (say, roughly 10000) in addition to the current dataset and try best to balance bots and humas. In order to balance, I need to do some analysis on the followers&followings of the users I have in the dataset (e.g., a user following a lot of bots is likely to be a bot). The lists of followers and followings of each user will be collected in doing work 2. Only active (and influential, maybe) users should be collected since inactive users could be very hard to identify.
4. Manually identify some of the new users (roughly 1000 during Summer vacation).
4. Collect the number of followings (and followers, perhaps) of each user every 3/4 days for calculating the change rate of # followings. This is a very strong feature for classification.
5. See if I can improve the classifier, e.g., change parameters, and apply standard boosting and bagging mentioned in the papers.
6. With Pearson's value, I might be able to see which two features have some relationship.
7. Sending any request to Twitter API with a deactivated account or a suspended account both return a 404 error. To distinguish these two kinds of accounts, go to https://twitter.com/intent/user?user_id=[userid] with the userid. If the account has been deactivated by the user, it will say "Sorry, that page doesn’t exist!". If the account has been suspended, it will say "Account suspended". Since there are only user ids but screen names in the honeypot dataset, that's the only way I've found to identify if an account has been suspended. So I might use some techniques to get this information automatically (e.g., web crawling?).
