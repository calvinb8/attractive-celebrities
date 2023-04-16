import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate 

def main():
    
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
    model = LogisticRegression(max_iter=500, random_state=0)
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
    
    #parameter tunning for decision tree
    
    scores={}
    
    for i in range(10, 100,10):
        for j in range(10, 100, 10):
            modelT=DecisionTreeClassifier(max_depth=i, min_samples_leaf=j)
            score=cross_validate(modelT, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = np.mean(score['test_score'])
            name="max depth: {}, min leaf sample: {}".format(i, j)
            scores[name]=mean_score
        
    print("best parameters for decision tree: ")
    print(max(scores, key=scores.get))
    
    #max depth: 20, min leaf sample: 80
    
    bestDepth=20
    bestLeafSample=80
    
    # decision tree
    model = DecisionTreeClassifier(max_depth=bestDepth, min_samples_leaf=bestLeafSample, random_state=0)
    model.fit(X_train, y_train)
    print('DT Score:')
    print(model.score(X_test, y_test))
    print()
    
    # stacking the models
    estimators = [
        ('lr', LogisticRegression(max_iter=500, random_state=0)),
        ('svm', svm.LinearSVC()),
        ('dt', DecisionTreeClassifier(max_depth=bestDepth, min_samples_leaf=bestLeafSample,random_state=0))
    ]
    model = StackingClassifier(estimators=estimators)
    model.fit(X_train, y_train)
    print('Stacked Score:')
    print(model.score(X_test, y_test))
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    