from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Get the train and test variables for cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#scale data to 0 and 1
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training

#Get data points for matplotlib plot on changing C and gamma
trainScoreC = []
testScoreC = []
for num in [0.1,1,10,100]:
    svm = SVC(kernel='rbf',C=num,gamma=1).fit(X_train_scaled, y_train)
    trainScoreC.append(svm.score(X_train_scaled, y_train))
    testScoreC.append(svm.score(X_test_scaled, y_test))
trainScoreG = []
testScoreG = []
for num in [0.1,1,10,100]:
    svm = SVC(kernel='rbf',C=1,gamma=num).fit(X_train_scaled, y_train)
    trainScoreG.append(svm.score(X_train_scaled, y_train))
    testScoreG.append(svm.score(X_test_scaled, y_test))

#Make the subplots
fig, ax = plt.subplots(2, 1, figsize=(5.75,6.5), squeeze=False)
mainList = [[trainScoreC, testScoreC], [trainScoreG, testScoreC]]
#Use zip to plot everything
for i, theList, theString in zip([0,1], mainList, ['C', 'gamma']):

    #Plot the train and test onto the graphs and add word details
    ax[i][0].plot([0.1, 1, 10,100], theList[0], label="train")
    ax[i][0].plot([0.1,1,10,100], theList[1], label="test")
    ax[i][0].legend(loc="upper left")
    ax[i][0].set_xlabel(theString)
    ax[i][0].set_ylabel('Accuracy')
    #Set the x scale to log 
    ax[i][0].set_xscale('log')

plt.show()
    


