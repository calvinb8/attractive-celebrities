import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

# importing the data
FILENAME = 'list_attr_celeba.csv'
df = pd.read_csv(FILENAME)

# splitting into X and y
X = df
X = X.drop('image_id', axis=1)
X = X.drop('Attractive', axis=1)
y = df['Attractive']

# splitting into training and testing
TEST_SIZE = 0.2
RANDOM_STATE = 123
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= TEST_SIZE, random_state= RANDOM_STATE)

# feature selection
# using select k best
NUM_FEATURES = 30
features = SelectKBest(k=NUM_FEATURES).fit(X_train, y_train)
X_train = features.transform(X_train)
X_test = features.transform(X_test)

# logistic regression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
print('Logistic Regression Score:')
print(model.score(X_test, y_test))
print()

# support vector machine
model = svm.LinearSVC()
model.fit(X_train, y_train)
print('SVM Score:')
print(model.score(X_test, y_test))
print()

# decision tree
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
print('DT Score:')
print(model.score(X_test, y_test))
print()

# stacking the models
estimators = [
    ('lr', LogisticRegression(random_state=0)),
    ('svm', svm.LinearSVC()),
    ('dt', DecisionTreeClassifier(random_state=0))
]
model = StackingClassifier(estimators=estimators)
model.fit(X_train, y_train)
print('Stacked Score:')
print(model.score(X_test, y_test))