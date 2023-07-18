import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from statistics import mean 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor


#load the data set and some pre proccesing on the data
dataset=pd.read_csv("aqardata_2.csv");

dataset["purpose"].fillna("غير محدد", inplace = True)
dataset["streetwidth"].fillna(dataset["streetwidth"].median(), inplace = True)
dataset = dataset.dropna()
datast = dataset.drop_duplicates()

#drop outliers
dataset.drop([37,1636,2013,2419,1246,2899,999,1193,2021,145,1380,1483,1209,2089,2295,2303,2655,2107,274,541,1199,1267,1554,1658,1721,2015,2268,2457,2328,2932,1582,2159,2469,1960,2551,2707,2856,2024,1902,2030,314,653,2211,1777,334],inplace= True)


#Scale the data using normlization
datasetReduced_N = dataset[dataset.columns.difference(['mainlocation','sublocation','neighborhood','frontage','purpose','Pricepm'])]
scaler = MinMaxScaler().fit(datasetReduced_N)
datasetReduced_N_Scaled = scaler.transform(datasetReduced_N)
dataset[datasetReduced_N.columns]= datasetReduced_N_Scaled


#One-hot encode catigorical data


dataset = pd.get_dummies(dataset, columns = ['mainlocation', 'sublocation','neighborhood','frontage','purpose'])
X = dataset.iloc[:, dataset.columns != 'Pricepm'].values
y = dataset.iloc[:, 2].values

#Create the 5 folds and traing the data in each iteration then take the avrage of the evalutions

# i have used randomsearchCV to hyperparameter tuning but i have commented it.


kfold = KFold(shuffle=True,n_splits=5,random_state=93)
scores1 = list()
scores2 = list()
scores3 = list()
for train_ix, test_ix in kfold.split(X):

    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
   
    # grid = RandomizedSearchCV(
    # estimator=GradientBoostingRegressor(),
    # param_distributions=  {
    # "n_estimators":[5,50,250,500],
    # "max_depth":[1,3,5,7,9],
    # "learning_rate":[0.01,0.1,1,10,100]}    
    # ,scoring='neg_mean_absolute_error',n_jobs=-1)
    
    # print("aaa")
    # grid.fit(train_X,train_y)
    # #print the best parameters from all possible combinations
    # print("best parameters are: ", grid.best_params_)
    # print("aaa")

    regressor = GradientBoostingRegressor(n_estimators=250,max_depth=9,learning_rate=0.1)
    regressor.fit(train_X, train_y)
    y_pred = regressor.predict(test_X)
    mse = mean_squared_error(test_y,y_pred)**0.5
    mae = mean_absolute_error(test_y,y_pred)
    r2 = r2_score(test_y,y_pred)
    scores1.append(mse)
    scores2.append(r2)
    scores3.append(mae)
   
# summarize model performance
print('Mean RMSE :',mean(scores1));
print('Mean MAE :',mean(scores3));
print('Mean R2 :',mean(scores2));


