import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

dataset = pd.read_csv("HR_comma_sep.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,9].values
y.ravel(-1)


le_x1 = LabelEncoder()
x[:,7] = le_x1.fit_transform(x[:,7])
le_x2 = LabelEncoder()
x[:,8] = le_x1.fit_transform(x[:,8])
ohe = OneHotEncoder(categorical_features = [7,8])
x = ohe.fit_transform(x).toarray()




from sklearn.cross_validation import train_test_split
y = pd.factorize(dataset['left'].values)[0].reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)



from sklearn.metrics import r2_score
r2_score(y_test , y_pred)