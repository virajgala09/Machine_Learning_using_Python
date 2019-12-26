Text Classification
===================

Machine learning algorithm for classifying input text as Decision, Action or Risk statement. 

Dataset
-------

The Dataset contains 50 statements with labels Decision and Action each and 100 statements of label Risk. The dataset is then split into 140 statements for training and 60 statements for testing.

Encoding
--------

Using LabelEncoder 3 labels are encoded between 0 to 2. Used is Tfidf_vectorizer to convert text into vectorized matrix. 

Model
-----

Models used for classification are SVM (Support Vector Machines), SGD (Stochastic Gradient Descent) and MultinomialNB (Naive Bayes)  and evaluated on the basis of confusion matrix and classification report.

Conclusion
----------

From the comparisons between Support Vector Machines, Stochastic Gradient Descent and MultinomialNB naive_bayes, it was observed that though Naïve Bayes is a great machine learning model for text data, SGD and LSVM have better accuracy for our dataset. It can be concluded that the Stochastic Gradient Descent is the best fit for our dataset as it has better F1 score and lower false predictions compared to Support Vector Machines.


