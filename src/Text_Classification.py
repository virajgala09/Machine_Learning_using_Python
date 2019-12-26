#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import model_selection, svm, naive_bayes
from sklearn.preprocessing import LabelEncoder

#Loading data from local storage
df = pd.read_csv(('data/text.csv'), index_col=False).sample(frac=1)
df.head(5)

#Splitting data into train and test data
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['sentence'],df['label'],test_size=0.3)

#Encoding the labels Decision, Action and Risk as 0, 1 and 2 numeric values respectively. 
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
print(Train_Y[0:5])

#The Tfidf_vectorizer converts the dataset of sentences into vectorized inputs.
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['sentence'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print(Tfidf_vect.vocabulary_)


# In[2]:


#model_name is the classification model that is selected
def model(model_name,Train_X_Tfidf,Test_X_Tfidf,Train_Y):
    
    model_name.fit(Train_X_Tfidf,Train_Y)
    predictions= model_name.predict(Test_X_Tfidf)
    return predictions;


# In[3]:


def evaluation(predictions,Test_Y):
    
    #To plot the confusion matrix
    evaluation_matrix = confusion_matrix(Test_Y, predictions)
    
    df_plot = pd.DataFrame(evaluation_matrix, index=['Action','Decision','Risk'], columns=['Action','Decision','Risk'])
    sns.heatmap(df_plot, annot=True)
    
    # Using accuracy_score function to get the accuracy
    print("Accuracy Score -> ",accuracy_score(predictions, Test_Y)*100)
    
    #Classification report for F1_score and precision
    print(classification_report(predictions, Test_Y))
    
    return;


# In[4]:


SVM_Classifier = svm.SVC(C=1.0, kernel='linear') #C is Regularization parameter that is multipled by sqaured L2 penalty and the result added to the cost function. Used default value here.

#first parameter of the model function (model_name) is the classifier to be selected
model_predictions = model(SVM_Classifier,Train_X_Tfidf,Test_X_Tfidf,Train_Y)
evaluation(model_predictions,Test_Y)


# In[5]:


SGD_Classifier = SGDClassifier()
model_predictions = model(SGD_Classifier,Train_X_Tfidf,Test_X_Tfidf,Train_Y)
evaluation(model_predictions,Test_Y)


# In[6]:


NB_Gaussian_Classifier = naive_bayes.MultinomialNB()
model_predictions = model(NB_Gaussian_Classifier,Train_X_Tfidf,Test_X_Tfidf,Train_Y)
evaluation(model_predictions,Test_Y)


# From the comparisons between Support Vector Machines, Stochastic Gradient Descent and MultinomialNB naive_bayes, it was observed that though Naïve Bayes is a great machine learning model for text data, SGD and LSVM have better accuracy for our dataset. It can be concluded that the Stochastic Gradient Descent is the best fit for our dataset as it has better F1 score and lower false predictions compared to Support Vector Machines.
