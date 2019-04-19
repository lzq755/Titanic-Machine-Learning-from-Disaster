import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

train = pd.read_csv('~/Documents/Kaggle/Titanic/Data/train.csv')
test = pd.read_csv('~/Documents/Kaggle/Titanic/Data/test.csv')

train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train['Age'] = train['Age'].interpolate()
train['Embarked'] = train['Embarked'].interpolate(method='pad')
test['Age'] = test['Age'].interpolate()
test['Fare'] = test['Fare'].interpolate()

le = preprocessing.LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex']).astype('float64')
train['Embarked'] = le.fit_transform(train['Embarked']).astype('float64')
test['Sex'] = le.fit_transform(test['Sex']).astype('float64')
test['Embarked'] = le.fit_transform(test['Embarked']).astype('float64')

X = train.drop(['PassengerId', 'Survived'], axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.LinearSVC()
clf.fit(X_train, y_train)

test_data = test.drop(['PassengerId'], axis=1)
prediction = clf.predict(test_data)

submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = prediction

submission.to_csv('submission.csv', index=False)