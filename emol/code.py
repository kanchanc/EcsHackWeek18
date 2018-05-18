import os

from sklearn import linear_model
import pandas as pd

data_path=os.path.join(os.path.dirname(__file__),'data')

def potential(Type,X):
    data=pd.read_csv(os.path.join(data_path,'data.csv'))
    
    df1=data.iloc[0:10]
    df2=data.iloc[10:]
    
    if Type=='A':
        df=df1
    else:
        df=df2
            
    x = df.values[:,1:10]
    y = df['E1'].values
    
   
    NoF=df['NoF'].values
    F=df['F'].values
    CF3=df['CF3'].values
    
    model = linear_model.LinearRegression(normalize=True)  
    model.fit(x, y)
    
    y_fit = model.predict(X)
    
    return y_fit
