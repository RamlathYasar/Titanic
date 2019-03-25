#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT DATASET
trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
X_train = trainset.iloc[: ,[2,4]].values
y_train =trainset.iloc[: , 1].values
X_test = testset.iloc[: , [1,3]].values

# HANDLING WITH CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder
labelencoder_Xtr = LabelEncoder()
X_train[:,1] = labelencoder_Xtr.fit_transform(X_train[:,1])
labelencoder_Xtest = LabelEncoder()
X_test[:,1] = labelencoder_Xtest.fit_transform(X_test[:,1])


#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#PREDICTING THE TEST SET RESULT
y_pred1 = classifier.predict(X_train) 

#MAKING THE CONFUSION MATRIX ( TO COMPUTE THE NO.OF CORRECT VALUES)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train,y_pred1)

#visualization of classifiers
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X_1 = np.arange(start = X_set[:,0].min() -1,stop = X_set[:,0].min() +1, step = 0.01)
X_2 = np.arange(start = X_set[:,1].min() -1,stop = X_set[:,1].min() +1, step = 0.01)
X1,X2 = np.meshgrid(X_1,X_2)
X1_ravel = X1.ravel()
X2_ravel = X2.ravel()
X1X2_array = np.array([X1_ravel, X2_ravel])
X1X2_array_t = X1X2_array.T
X1X2_pred = classifier.predict(X1X2_array_t)
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape,alpha=0.75,cmap = ListedColormap(('red', 'green')))

#plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X1.min(),X1.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0.0],X_set[y_set == j, 1.0],c = ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()
