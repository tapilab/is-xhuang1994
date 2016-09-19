## Source code

read_honeypot.py <br />
This file downloads and reads data from honeypot dataset, and store data into MongoDB -> old_data

new_data.py <br />
This file uses REST API to get most recent data of features of users in old_data, and store new data into MongoDB -> new_data

new_users.py <br />
This file finds new users through Streaming API, gets recent data for users, and store new users into MongoDB -> new_users

analyze.py <br />
This file plots graphs with different features to see distribution patterns of features. It also does feature selection work which is to be complete.

classify.py <br />
This file reads data from MongoDB, test classification accuracy with new_data, and also predict new_users with the classification model.

Replicate.ipynb <br />
This notebook replicates what were done before May 2016. It's outdated but generally explains what I did last semester.
