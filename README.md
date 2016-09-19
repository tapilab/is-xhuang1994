
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

[9/19 Update]

The datasets I'm using include:
old_data: The data extracted from honeypot dataset[1]
new_data: The most recent data I recollected for each user in honeypot dataset[1] through Rest API
new_users: The data for 1100 new users I found through Streaming API

For each user, I have these data fed to classifier:
[# followings, # followers, # tweets, length of user name, length of user screen name, length of description in user profile, if the user is geo-enabled, standard deviation of # follwings, standard deviation of differences of # follwings (the change rate of # followings), lag1 autocorrelation of # follwings (the change rate of # followings), # tweets posted on each weekday, ratio of tweets posted on each weekday, ratio of urls in tweets, ratio of unique urls in tweets, ratio of mentions in tweets, ratio of unique mentions in tweets, ratio of hashtags in tweets, ratio of unique hashtags in tweets, if the tweets have gps location, if the tweets content ends with punctuations/hashtags/urls, Jaccard Similarity of tweets content]

In addition, I plotted graphs of # followers vs. # followings of users within each of the datasets. 

![Image](../master/src/graphs/old_data.png?raw=true)
Figure 1: Graph based on old_data. 
![Image](../master/src/graphs/new_data.png?raw=true)
Figure 2: Graph based on new_data.
![Image](../master/src/graphs/new_users.png?raw=true)
Figure 3: Graph based on new_users

As we can see, the graph based on old_data read from honeypot dataset appears to be highly formed and converged. The second figure based on new_data is less, but still, formed and converged. And the third figure based on new_users which are randomly fetched from Twitter looks more normal and rational.
This might indicate that the authors of honeypot dataset have used some threshold (at least on # followers and # friends) in collecting and labeling users. And the result of classification accuracy can not be applied to other users ion Twitter. Therefore I randomly fetched more than 1000 new users through Streaming API which will be manually identified to test generality of my classification model.

## Methods

I collect all the data mentioned above for every user, scale each feature by its maximum value, and feed the data to a classifier (random forest is used now), and do a 10-fold cross validation to see the results (total accuracy, respective accuracy for humans and bots, f1 socres for humans and bots, and confusion matrix)

## Results

Here is a screenshot of accuracy results:  
![Image](../master/src/graphs/Result.png?raw=true)  
Figure 4: Accuracy results for classification  

[9/19 Update]

Classification accuracy with new_data is shown below:
![Image](../master/src/graphs/Result_new.png?raw=true)  
Figure 5: Accuracy results for classification with new_data

The new result is noticeably lower than old result, which reflects the insufficient of generality of the classification model. This model still needs improvement.

## Conclusions / Future Work

The results show a high classification accuracy -- 94% in total -- while the highest accuracy reported in the papers is over 99%. There could be some criterions in collecting the bot and human users that are not mentioned in the papers. I have several plans for Summer vacation and Fall semester:

1. [Postponed] Collect domain names of urls collected from recent tweets of each user I have. Find a way to rank the domains and adding this as an additional feature to my current model and test for accuracy. This can be achieved in doing work 2. <br />
[9/19 Update] <br />
It turns out that the 'expanded_url' field of tweet objects returned by Rest API doesn't always give the 'expanded' url. Actually most of the urls are still shortened. And I couldn't find a good way to rank websites. So this work is postponed. <br /> <br />
2. [Complete] Recollect all the features needed for each user. Put data into my current model and test for accuracy (also need to replot the # followings vs. # followers graph and see the users distribution patterns) <br /> <br />
3. [Complete] Find more users (say, roughly 10000) in addition to the current dataset and try best to balance bots and humas. In order to balance, I need to do some analysis on the followers&followings of the users I have in the dataset (e.g., a user following a lot of bots is likely to be a bot). The lists of followers and followings of each user will be collected in doing work 2. Only active (and influential, maybe) users should be collected since inactive users could be very hard to identify. <br />
[9/19 Update] <br />
It turns out that getting list of friends & followers for each user is very time consuming, since the rate limit is 15 requests per 15-minute window. So I turned to get more users randomly, assuming that there are quite a number of bots on Twitter and I won't miss all of them. <br /> <br />
4. [In-Progress] Manually identify some of the new users. <br /> <br />
5. [Postponed] Collect the number of followings (and followers, perhaps) of each user every hour for calculating the change rate of # followings. This is a very strong feature for classification. <br />
[9/19 Update] <br />
This is doable under the request rate limit. But it will take roughly a week to run and can not be paused once started. So I'll probably do it if I have an idled computer. <br /> <br />
6. [In-Progress] See if I can improve the classifier, e.g., change parameters, and apply standard boosting and bagging mentioned in the papers. <br /> <br />
7. [Postponed] With Pearson's value, I might be able to see which two features have some relationship. <br />
[9/19 Update] <br />
I'm not sure if this is meaningful. I'll do feature selection work instead. <br /> <br />
8. [Postponed] Sending any request to Twitter API with a deactivated account or a suspended account both return a 404 error. To distinguish these two kinds of accounts, go to https://twitter.com/intent/user?user_id=[userid] with the userid. If the account has been deactivated by the user, it will say "Sorry, that page doesn’t exist!". If the account has been suspended, it will say "Account suspended". Since there are only user ids but screen names in the honeypot dataset, that's the only way I've found to identify if an account has been suspended. So I might use some techniques to get this information automatically (e.g., web crawling?). <br />
[9/19 Update] <br />
I did found a number of deactivated users in honeypot dataset. But I don't know if it's meaningful to analyze these users. I'll probably do it as a practice with web crawling when I have time.

## References

[1] K. Lee, B. Eoff, and J. Caverlee. Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter. In Proceeding of the 5th International AAAI Conference on Weblogs and Social Media (ICWSM), Barcelona, July 2011. (Bibtex)
