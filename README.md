# Practical-Application-17-1

Practical Application III: Comparing Classifiers
Overview: In this practical application, the goal is to compare the performance of the 4 different types of classifiers:  K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. To compare the performance we used a dataset related to marketing bank products over the telephone.


The dataset comes from the UCI Machine Learning repository and is the data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

#Problem 1: Understanding the Data
The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. 

#Problem 2: Read in the Data
The data is in a csv file but was delimited by semicolns instead of commas. I read the data into a DataFrame named df, but rename it once I have dome some inital investigation namely .head(), and .info() and a starting correlation of the numeric features. 

#Problem 3: Understanding the Features
Input variables:
-bank client data: age , job, marital, education, default,housing,loan
-related with the last contact of the current campaign:contact. month .day_of_week. duration
-other attributes:campaign,pdays, previous,poutcome
-social and economic context attributes emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

Output variable (desired target):
21 The output data is stored in column 'y' - has the client subscribed a term deposit? (binary: 'yes','no')

#Problem 4: Understanding the Task
The objective for this project is two fold:
 #1 Correctly predict the person to call whowill say "yes" to a marketing campaign
 #2 Compare 4 different model types 

#Problem 5: Engineering Features
The data is imbalanced making it easy to predict 'no' and be correct. no=26781, yes= 3886
The recall and precision scores will be the most telling measure.
  precision = From all of the positive class we predicted as 'yes', how many are actually 'yes'?
  recall = From all of the positive class, how many we predicted correctly.

the main data cleaning was: 
remove 'unknown' from data
convert object data to binary
Change object 'education' data to ordinal 
Change nominal/catagorical data to numeric e.g. 'job','marital', 'month', 'day_of_week'

I need to pair down any unneded or unhelpful features.  
Run RFE to answer: What are the 10 best Logistic Regression feature?
                    What are the 10 best DecisionTree Classifier features
RFE was not much help since they both used different features
Logistic Regression: = ['previous', 'emp.var.rate', 'job_retired', 'job_student', 'month_apr', 'month_mar', 'month_may', 'month_nov', 'month_sep', 'day_of_week_mon']

Decision Tree = ['age', 'education', 'housing', 'duration', 'campaign', 'previous', 'emp.var.rate', 'cons.conf.idx', 'marital_married', 'day_of_week_thu']

Let's look at the df_bank correlations now that all of the data has been transfomed where needed

relevant_features
duration          0.394254
pdays             0.325933
previous          0.228206
emp.var.rate      0.305049
cons.price.idx    0.128968
euribor3m         0.315522
nr.employed       0.363606
y                 1.000000
job_retired       0.102017
month_mar         0.146073
month_may         0.113197
month_oct         0.140435
month_sep         0.123232

Check for correlated columns that can be removed.
Drop 'poutcome' - too many unknowns.
Drop 'dpays' - too many were 999 -39673
Drop contact_telephone   corr= 0.144249, contact_cellular corr=  0.144249
Drop cons.price.idx  higly corr. with emp.var.rate
Drop euribor3m higly corr. with emp.var.rate
Drop nr.employed higly corr. with emp.var.rate

#Problem 6: Train/Test Split
Create train and test data sets
X_train, X_test, y_train, y_test = train_test_split(df_bank_X, df_bank_y, test_size=0.5, random_state=42 )
Smaller test size might make for more accurate results but will take longer to train.
X_train

#Problem 7: A Baseline Model
Before we build our first model, we want to establish a baseline. What is the baseline performance that our classifier should aim to beat?

Get data for baseline
The number of actual false=13398, the number of true=1935

Theoretical values if our model always chooses 0
tn=13398
tp=0
fn=1935
fp=13398

Baseline our model should do better than just guessing the largest class 100% of the time. 
'Baseline Accuracy =', 1- true/false)
'Baseline Precision =', tp / (tp + fp))
'Baseline Recall = ', tp / (tp + fn))

Create a table of scores for the base model types
	Model	              Train Time	Training Accuracy	Test Accuracy	Test Recall	Test Precision
0	svc	                3.381329	0.880454	        0.879875	    0.169657	  0.598553
0	Decision Tree	      0.016613	0.878497	        0.876484	    0.169657	  0.525933
0	Logistic Regression	0.039870	0.890498	        0.890179	    0.345464	  0.623497
0	KNN	                0.004016	0.929759	        0.873353	    0.313685	  0.503704


#Problem 11: Improving the Model
Now that we have some basic models on the board, we want to try to improve these. 
Use Gridsearch CV to try and improve the models

Below is a list of the final results
	Model	              Train Time	Test Accuracy	Test Recall	Test Precision
0	KNN	                421.103098	0.874071	    0.158380	  0.516722
0	Logistic Regression	121.839924	0.890244	    0.345976	  0.623845
0	SVC	                1193.528379	0.872766	    0.000000	  0.000000
0	Decision Tree	        1.033461	0.899504	    0.536648	  0.621734

#Conclusion
The DecisionTreeClassifier offered the best model determining who and when to call to get a 'yes' for this data set.
It was the most accurate, ran the fastest and had the best performing precision and recall scores. It correctly classified 542 of the 1935 positive outcomes and only misclassified 481 of the negative outcomes.
The decision tree is probably the easist to emplement as well. 

Although it didn't fit the parameters of this exercise, the duration of the call seemed to have a large effect if getting to 'yes'. Perhaps there is more to the 'script' used than the demographics or other features.
