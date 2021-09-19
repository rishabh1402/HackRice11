# Importing the libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle

#Loading Dataset
store = pd.read_csv('sales.csv')

#Encoding The IteM Type Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_type = le.fit_transform(store['Item Type'])
store['encoded_type'] = encoded_type


#Assigning values to x & y
x = store.drop(columns=['Region','Country','Item Type','Sales Channel','Order Priority','Order Date','Order ID','Ship Date','Unit Price','Unit Cost','Total Revenue','Total Cost','Total Profit'])
y = store['Total Profit'].values

#Splitting the data set into training and testing sets (80:20)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Training the model with training sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(x_train,y_train)

#Predicting output by using Trained Model with testing sets
pred= regressor.predict(x_test)
 
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,103]]))