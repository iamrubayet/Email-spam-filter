import pandas as pd
from sklean.feature_extraction.text import countVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from skelarn import svm
from sklearn.model_selection import GridsearchCV

##step1: load the data

dataframe = pd.read_csv("spam.csv")
print(dataframe.head())

##step2: splitting the dataset into test and training
## 80% will be training and 20% will be test

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

##step3: Extract Features

cv = CountVectorizer()
features = cv.fit_transform(x_train)

##step4: build a model

tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],
                    'C':[1,10,100,1000]}

model = GridSearchCV(svm.SVC(),tuned_parameters)

model.fit(features,y_train)

print(model.best_params_)

##step5: Test Accuracy

features_test = cv.transform(x_test)

print("Accuracy of the model is:",model.score(features_test,y_test))









