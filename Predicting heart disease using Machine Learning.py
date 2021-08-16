###########################################################################################################################
#               Hassan Shahzad
#               18i-0441
#               CS-D
#               FAST-NUCES ISB
#               chhxnshah@gmail.com
#               "Predicting heart disease using Machine Learning"
#               The following project was one of the guided projects offered by courera. 

###########################################################################################################################


################################################# Predicting Heart Disease from Clinical and Laboratorial Data ################################################################

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
%matplotlib inline

# Loading the dataset
data = pd.read_csv('heart.csv')
data.head()

# Shape
data.shape

# Variable Types
data.dtypes


######################################################################## EDA and Pre-Processing #################################################################################

############################
## Outcome Variable Count ##
############################

# Creates a graph of patients having heart disease and those without it
sns.catplot(x = 'target', kind = 'count', palette= 'ch:.25', data = data)

# Categorical Predictive Variables

# sex
sns.catplot(x = 'sex', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# cp:
sns.catplot(x = 'cp', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# fbs:
sns.catplot(x = 'fbs', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# restecg:
sns.catplot(x = 'restecg', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# exang:
sns.catplot(x = 'exang', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# slope:
sns.catplot(x = 'slope', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# ca:
sns.catplot(x = 'ca', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')

# thal:
sns.catplot(x = 'thal', kind = 'count', hue = 'target', data= data, palette = 'ch:.25')


#########################################
## Distributional Predictive Variables ##
#########################################


data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].describe()

sns.displot(x='age', multiple = 'stack', hue = 'target', data= data, palette = 'ch:.25')
sns.displot(x='trestbps', multiple = 'stack', hue = 'target', data= data, palette = 'ch:.25')
sns.displot(x='chol', multiple = 'stack', hue = 'target', data= data, palette = 'ch:.25')
sns.displot(x='thalach', multiple = 'stack', hue = 'target', data= data, palette = 'ch:.25')
sns.displot(x='oldpeak', multiple = 'stack', hue = 'target', data= data, palette = 'ch:.25')


##################################
## Splitting and Pre-Processing ##
##################################

# Defining x_train, x_test, y-train and y_test
x = data.drop('target', axis =1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

# Scaling the data
sc = StandardScaler().fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

########################
## Training The Model ##
########################

# Parameters for grid search
knn = KNeighborsClassifier()
parameters = {'n_neighbors': [3,5,7,9,11], 'weights': ['uniform', 'distance']}

# Fiting training data and grid searching
grid = GridSearchCV(knn, parameters, cv = 4, scoring = 'accuracy')
grid.fit(x_train,y_train)

# Displaying best parameters
print(grid.best_params_)

# Picking the best model
model = grid.best_estimator_

##########################
## Evaluating the Model ##
##########################

# Model score on test data
model.score (x_test,y_test)

# Confusion (Prediction) Matrix
predictions = model.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)
cm = pd.DataFrame(cm)
sns.heatmap(cm, annot=True)
plt.show()

# Calculating sensitivity, specificity, PPV and NPV
TP = 28
FP = 2
TN = 27
FN = 4
sensitivity = TP / (TP + FN) *100
specificity = TN / (TN + FP) * 100
ppv = TP / (TP + FP) * 100
npv = TN / (TN + FN) * 100

# Printing sensitivity, specificity, PPV and NPV
print('Sensitivity:', sensitivity,'% ','Specificity:', specificity,'% ','positive predictive value:',ppv,'% ','negative predictive value:',npv,'%' )

# AUC Score
probs = model.predict_proba(x_test)[:, 1]
auc = metrics.roc_auc_score(y_test, probs)
print(auc)

# ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test,probs)
plt.figure()
plt.grid()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.show()


######################################################################################################################
################################################### THE END ##########################################################
######################################################################################################################