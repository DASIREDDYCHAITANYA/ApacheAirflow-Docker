from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


with DAG(dag_id='machine_learning_model',start_date=datetime(2023,1,1),schedule_interval='@weekly',catchup=False,) as dag:
  @task

  def preprocess():
    df=yf.download('TSLA')
    df=pd.DataFrame(df)
    df = df.reset_index()
    df=df.drop('Date', axis=1)
    return df
  @task
  def train(df):
    x=df.drop('Adj Close',axis=1)
    y=df['Adj Close']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)
    print('rmse',rmse)
    print('r2',r2)
  df=preprocess()
  train(df)