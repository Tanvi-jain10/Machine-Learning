import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

file_path="/Users/anishjain/Downloads/AirPassengers.csv"
df=pd.read_csv(file_path)  # read the data set
print(df.head())  # starting 5 rows 
print(df.tail())  # ending 5 rows
print(df.size)    # size
print(df.shape)   # no of rows and columns

print(df.describe())  #  describe the columns 
print(df.columns)
df1 = pd.DataFrame()
df1['ds'] = pd.to_datetime(df['Month'])
df1['y'] = df['#Passengers']
df1.head()
m=Prophet()
m.fit(df1)
future = m.make_future_dataframe(periods=12 * 5,
                                 freq='M')

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower',
          'yhat_upper', 'trend',
          'trend_lower', 'trend_upper']].head()
forecast[['ds', 'yhat', 'yhat_lower',
          'yhat_upper', 'trend',
          'trend_lower', 'trend_upper']].tail()
fig1 = m.plot(forecast, include_legend=True)
fig2 = m.plot_components(forecast)
import numpy as np
actual = df1['y']  # Replace with actual values
predicted = forecast['yhat'][:len(actual)]  # Predicted values
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
print(f"MAPE: {mape:.2f}%")


