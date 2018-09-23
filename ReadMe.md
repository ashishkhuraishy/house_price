

```python
# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dtr




# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'C:/Users/Ashish Khuraishy/Desktop/New folder/Housing Data/train.csv'
home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
data = home_data.drop(['SalePrice'], axis = 1 )
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = data.select_dtypes(exclude = ['object'])


```

    c:\users\ashish khuraishy\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    


```python
#home_data.describe()
```


```python
#X.head()
```

# Comparing
## 1. Decision Tree regressor
## 2. random Forest Regressor


```python
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.7)

def score_data(train_X, train_y, val_X, val_y):
    model1 = dtr()
    model2 = rfr(random_state = 1)
    model1.fit(train_X, train_y)
    model2.fit(train_X, train_y)
    pred1 = model1.predict(val_X)
    pred2 = model2.predict(val_X)
    score1 = model1.score(val_X, val_y)
    score2 = model2.score(val_X, val_y)
    MAE1 = mae(pred1, val_y)
    MAE2 = mae(pred2, val_y)
    return "Accuracy Score DTR : {}% \tMean Absolute Error DTR : {}$ \nAccuracy Score RFR : {}% \tMean Absolute Error RFR : {}$".format(round(score1*100, 2), round(MAE1, 2), round(score2*100, 2), round(MAE2, 2))


```

    c:\users\ashish khuraishy\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\model_selection\_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    

# Mean Absolute Error And AccuracY Score:

## 1.Dropping Down Columns


```python
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

red_train_x = train_X.drop(cols_with_missing, axis = 1)
red_val_x = val_X.drop(cols_with_missing, axis = 1)
print(score_data(red_train_x, train_y, red_val_x, val_y))
```

    Accuracy Score DTR : 73.11% 	Mean Absolute Error DTR : 27584.21$ 
    Accuracy Score RFR : 88.23% 	Mean Absolute Error RFR : 18135.35$
    

## 2.Model Score From Imputation


```python
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = val_X.copy()

cols_with_missing = (col for col in train_X.columns if train_X[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
#print(imputed_X_train_plus, "\n", imputed_X_test_plus)
```


```python
from sklearn.preprocessing import Imputer
impt = Imputer()
impt_train_x = impt.fit_transform(train_X)
impt_val_x = impt.fit_transform(val_X)
print(score_data(impt_train_x, train_y, impt_val_x, val_y))
```

    Accuracy Score DTR : 76.4% 	Mean Absolute Error DTR : 27602.28$ 
    Accuracy Score RFR : 86.35% 	Mean Absolute Error RFR : 18920.76$
    

# Final code
_____________________
### From our comparing we reached a conclusion that RFR has greater Accuracy and lesser Mean Absolute Error than DTR.
### So we will be using RFR for our Final_Model


```python
iowa_test_path = 'C:/Users/Ashish Khuraishy/Desktop/New folder/Housing Data/test.csv'
test_iowa = pd.read_csv(iowa_test_path)
test_data = test_iowa.select_dtypes(exclude = ['object'])
impt_X = impt.fit_transform(X)
impt_test_data = impt.fit_transform(test_data)
#test_data.head()
#print(impt_test_data)
```


```python
final_model = rfr()
final_model.fit(impt_X, y)
pred = final_model.predict(impt_test_data)
#print(pred)
```

# Output


```python
output = pd.DataFrame({'Id' : test_data.Id, 'SalePrice' : pred})
output.to_csv('submission2.csv', index = False)
```
