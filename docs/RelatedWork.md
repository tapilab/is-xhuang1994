1. Who is Tweeting on Twitter: Human, Bot, or Cyborg?

Description:
The study uses feature-based detection method and multiple classification algorithms to recognize spammers on Twitter.

Data and Method:
The data they collected from each user include:
 - cumulative distribution function (CDF) of twitter count, 
 - numbers of follwers and friends
 - CDF of ratio of followers over friends
 - number of tweets per hour/week/quarter/year
 - tweeting device makeup
 - External URL ratio in tweets
They applied multiple classification algorithms on the data collected, including entropy-based algorithm and machine-learning algorithm. Their classification accuracies are 94.9%, 93.7%, and 82.8% for human, bot, and cyborg, repectively.


2. Detecting and Analyzing Automated Activity on Twitter

Description:
The study uses feature-based detection method and mathematically analyzes the collected feature to distinguish between human and bots.

Data and Mathod:
They collected update times of every tweet of users and used Pearson's X^2 test to assess whether a set of update times is consistent with the uniform second-of-the-minute and minute-ofthe-hour distributions expected from human users. They captured a number of accounts  that exhibit anomalous timing behavior.


3. Detecting Spammers on Twitter

Description:
The study uses feature-based detection method and analyzes the features using classification algorithms and Pearson's X^2 value.

Data and Mathod:
The data they collected from each user include:
 - fraction of tweets with URLs
 - fraction of tweets with spam word
 - hashtags per tweet (average)
 - number of followers per number of followees
 - account age
 - number of tweets received
The used a Support Vector Machine (SVM) classifier on the collected features. Their accuracy of recognizing spammers is 70.1% while the accuracy of recognizing non-spammers is 96.4%. They also used the Pearson's X^2 value to find out which features should be used. Besides, a tring on detecting spammers by detecting spams has got a similar accuracy.


4. Uncovering Social Spammers: Social Honeypots + Machine Learning

Description:
The study uses feature-based detection method. It compared the importance of different features and compared the accuracy of different classifier in this case.

Data and Mathod:
The data they collected from each user include:
 - number of friends
 - About Me (AM) content in tweets
 - AM length
 - user age
 - martial status
 - sex
 - following-followers ratio
 - number of @ per tweet
 - number of URLs per tweet
 - account age
 - number of unique @ per tweet
 - number of unique URLs per tweet
 - tweets similarity
 - number of tweets
They tried many classifiers to distinguish between humans and bots using the features collected. The Decorate classifier gives the best result: 88.98% of accuracy.
