#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# In[6]:


df.head(11)


# In[4]:


df.shape


# In[5]:


df['seller_type'].unique()


# In[6]:


df['transmission'].unique()


# In[7]:


df['owner'].unique()


# In[8]:


df['name'].unique().shape


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


final_dataset=df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type','transmission', 'owner']]


# In[13]:


final_dataset.head()


# In[14]:


final_dataset


# In[15]:


final_dataset.head()


# In[16]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[17]:


final_dataset=final_dataset.rename(columns={'owner_Fourth & Above Owner':'owner_forth','owner_Third Owner':'owner_third','owner_Second Owner':'owner_second','owner_Test Drive Car':'owner_first'})


# In[18]:


final_dataset=final_dataset.drop(['seller_type_Trustmark Dealer'],axis=1)


# In[19]:


final_dataset.columns


# In[20]:


final_dataset.corr()


# In[21]:


final_dataset.head()


# In[22]:


import seaborn as sns


# In[23]:


sns.pairplot(final_dataset)


# In[24]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(final_dataset.corr(),annot=True)


# In[25]:


X=final_dataset[['year', 'km_driven', 'fuel_Diesel', 'fuel_Electric',
       'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
        'transmission_Manual', 'owner_forth',
       'owner_second', 'owner_first', 'owner_third']]
y=final_dataset['selling_price']


# In[26]:


X.head()


# In[27]:


y.head()


# In[28]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[29]:


model.feature_importances_


# In[30]:


f_impo=pd.Series(model.feature_importances_,index=X.columns)
f_impo.nlargest(5).plot(kind='barh')


# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_split=train_test_split(X,y,test_size=0.2)


# In[32]:


X_train


# In[33]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()


# In[34]:


#### let we check for hyperparameters
import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[35]:


from sklearn.model_selection import RandomizedSearchCV


# In[36]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[37]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[38]:


rf = RandomForestRegressor()


# In[39]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[40]:


rf_random.fit(X_train,y_train)


# In[41]:




rf_random.best_params_


# In[42]:


predictions=rf_random.predict(X_test)


# In[43]:


predictions


# In[44]:


sns.distplot(y_split-predictions)


# In[45]:


sns.scatterplot(y_split,predictions)


# In[46]:


import pickle


# In[47]:


file=open('random_forest.pkl','wb')
pickle.dump(rf_random,file)


# In[7]:


from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('random_forest.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    fuel_Diesel=0
    fuel_Electric=0
    fuel_LPG=0
    owner_second = 0
    owner_third = 0
    owner_forth = 0

    if request.method == 'POST':
        year = int(request.form['year'])
        km_driven=int(request.form['km_driven'])
        km_driven2=np.log(km_driven)
        owner_first = request.form['owner_first']
        if(owner_first=='first'):
            owner_first=1
            owner_second=0
            owner_third=0
            owner_forth=0
        elif(owner_first == 'second'):
            owner_first = 0
            owner_second = 1
            owner_third = 0
            owner_forth = 0
        elif(owner_first == 'third'):
            owner_first = 0
            owner_second = 0
            owner_third = 1
            owner_forth = 0
        else:
            owner_first = 0
            owner_second = 0
            owner_third = 0
            owner_forth = 1
        fuel_Petrol = request.form['fuel_Petrol']
        if (fuel_Petrol == 'Petrol'):
            fuel_Petrol = 1
            fuel_Diesel = 0
            fuel_Electric = 0
            fuel_LPG = 0
        elif(fuel_Petrol=='Diesel'):
            fuel_Petrol = 0
            fuel_Diesel = 1
            fuel_Electric = 0
            fuel_LPG = 0
        elif (fuel_Petrol=='Electric'):
            fuel_Petrol = 0
            fuel_Diesel = 0
            fuel_Electric = 1
            fuel_LPG = 0
        else:
            fuel_Petrol = 0
            fuel_Diesel = 0
            fuel_Electric = 0
            fuel_LPG = 1
        seller_type_individual=request.form['seller_type_individual']
        if(seller_type_individual=='individual'):
            seller_type_individual=1
        else:
            seller_type_individual=0
        transmission_Mannual=request.form['transmission_Mannual']
        if(transmission_Mannual=='Mannual'):
            transmission_Mannual=1
        else:
            transmission_Mannual=0
        prediction=model.predict([[km_driven2,owner_first,owner_second,owner_third,owner_forth,year,fuel_Petrol,fuel_Diesel,fuel_Electric,fuel_LPG,seller_type_individual,transmission_Mannual]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


# In[ ]:




