import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
file_path="/Users/anishjain/Downloads/AirPassengers.csv"
df=pd.read_csv(file_path)  # read the data set
print(df.head())  # starting 5 rows 
print(df.tail())  # ending 5 rows
print(df.size)    # size
print(df.shape)   # no of rows and columns

print(df.describe())  #  describe the columns 
print(df.columns)


print(df.isnull().sum()) # checking a null values 

#  to convert the month cloumn in datetime format 
from datetime import datetime 
df['Month']=pd.to_datetime(df['Month'],infer_datetime_format=True)
df1=df.set_index(['Month'])
print(df1.head())

#ploting a graph
# ploting a graph of date and number of passengers
plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.plot(df1)
plt.show()

# Identifying trends in noisy time series data.
# Detecting seasonality by comparing the rolling mean with the actual data.
# Monitoring changes in volatility over time.
# Detecting periods of high or low stability.


rolmean=df1.rolling(window=12).mean()
rolstd=df1.rolling(window=12).std()
print(rolmean,rolstd)

# ploting rolling 

orig=plt.plot(df1,color='blue',label='Original')
mean=plt.plot(rolmean,color='red',label="Rolling Mean")
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Rolling Standard Deviation")
plt.show()







# estimating trend 
# Many time series models require the mean and variance to be constant over time (stationarity). Log transformation is a step toward achieving this.

df_logscale=np.log(df1)
plt.plot(df_logscale)
plt.show()


# moving average : in time series modeling, moving averages are used to remove trends and make the data stationary.
movingAverage = df_logscale.rolling(window=12).mean()
movingSTD = df_logscale.rolling(window=12).std()
plt.plot(df_logscale)
plt.plot(movingAverage,color='red')
plt.show()

datasetlogsczleMinusMovingaverage=df_logscale-movingAverage
print(datasetlogsczleMinusMovingaverage.head(12))

# removing nan values
datasetlogsczleMinusMovingaverage.dropna(inplace=True)
print(datasetlogsczleMinusMovingaverage.head(12))


# determing rolling statistic
# to check time is stationay or not 
from statsmodels.tsa.stattools import adfuller 
def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    orig = plt.plot(timeseries,color='blue',label='Original')
    mean = plt.plot(movingAverage,color='red',label='Rolling mean')
    std  = plt.plot(movingSTD,color='black',label='Rolling STD')
    plt.legend(loc='best')
    plt.title("Rolling Mean 7 Standard Deviation")
    plt.show()

    # Dickey fuller test
    

    dftest=adfuller(timeseries['#Passengers'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags Used','Number of observations Used'])
    for key ,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key]=value
    print(dfoutput)

test_stationarity(datasetlogsczleMinusMovingaverage)


# null hypothesis 
# p value is less than 0.5  and Test Statistic  and critical value is approx same 



# we need to see the trend that is present inside the time series 

exponentialDecayweightAverage = df_logscale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(df_logscale)
plt.plot(exponentialDecayweightAverage,color='red')
plt.show()
# the trend is going upward 

dataserlogscaleminusMovingExpoentialdecayaverage= df_logscale-exponentialDecayweightAverage
test_stationarity(dataserlogscaleminusMovingExpoentialdecayaverage)

# use function shift 
datasetlogdiffshifting=df_logscale-df_logscale.shift()
plt.plot(datasetlogdiffshifting)
plt.show()

datasetlogdiffshifting.dropna(inplace=True)
test_stationarity(datasetlogdiffshifting)



# 
from statsmodels.tsa.seasonal import seasonal_decompose
deccomposition = seasonal_decompose(df_logscale)

trend =deccomposition.trend
seasonal = deccomposition.seasonal
residual= deccomposition.resid  # it is irregular in nature

plt.subplot(411)
plt.plot(df_logscale,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#decomposedlogdata= residual
#decomposedlogdata.dropna(inplace=True)
#test_stationarity(decomposedlogdata)

# ACF AND PACF plots  
# we have to find the value of p and q 


from statsmodels.tsa.stattools import acf, pacf
lag_acf=acf(datasetlogdiffshifting,nlags=20)
lag_pacf=pacf(datasetlogdiffshifting,nlags=20,method='ols')


# plot ACF 
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle="--",color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogdiffshifting)),linestyle="--",color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogdiffshifting)),linestyle="--",color='gray')
plt.title('Autocorrelation function')
plt.show()

# plot Pacf
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle="--",color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogdiffshifting)),linestyle="--",color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogdiffshifting)),linestyle="--",color='gray')
plt.title(' Partial Autocorrelation function')
plt.tight_layout()
plt.show()


# ARIMA Model 
# AR 

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df_logscale, order = (2,1,2))
results_AR= model.fit()
fitted_values_AR=results_AR.fittedvalues
aligned_actual_AR = datasetlogdiffshifting["#Passengers"].reindex(fitted_values_AR.index).dropna()

# Calculate RSS for AR model
rss_AR = np.sum((aligned_actual_AR - fitted_values_AR)**2)
plt.plot(datasetlogdiffshifting,color="blue")
plt.plot(fitted_values_AR,color='red')
plt.title(f"AR Model RSS: {rss_AR:.4f}")
print("Plotting AR model")

plt.show()

# MA Model
model = ARIMA(df_logscale,order = (0,1,2))
results_MA= model.fit()
fitted_values_MA = results_MA.fittedvalues
aligned_actual_MA = datasetlogdiffshifting["#Passengers"].reindex(fitted_values_MA.index).dropna()

# Calculate RSS for MA model
rss_MA = np.sum((aligned_actual_AR - fitted_values_AR)**2)
plt.plot(datasetlogdiffshifting)
plt.plot(fitted_values_MA ,color='red')
plt.title(f"MA Model RSS: {rss_MA:.4f}")
print("Plotting MA model")

plt.show()




# ARIMA Model
model = ARIMA(df_logscale,order = (2,1,2))
results_ARIMA= model.fit()
fitted_values_ARIMA = results_ARIMA.fittedvalues
aligned_actual_ARIMA = datasetlogdiffshifting["#Passengers"].reindex(fitted_values_ARIMA.index).dropna()

# Calculate RSS for MA model
rss_MA = np.sum((aligned_actual_MA - fitted_values_MA)**2)

plt.plot(datasetlogdiffshifting)
plt.plot(fitted_values_ARIMA,color='red')
plt.title(f"ARIMA Model RSS: {rss_MA:.4f}")

plt.show()


 


predictions_ARIMA_diff= pd.Series(fitted_values_ARIMA,copy=True)
print(predictions_ARIMA_diff.head())

# Cumulative sum
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log=pd.Series(df_logscale['#Passengers'].iloc[0],index=df_logscale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print(predictions_ARIMA_log.head())

print(df_logscale)


#print(results_ARIMA.get_forecast(1,264))
x=results_ARIMA.forecast(steps=120) # print the predictaed values of next ten years 
print(x)

#print(df1.head())  # Check the first few rows of the original data
#print(predictions_ARIMA_log.head())


#predictions_ARIMA = predictions_ARIMA.reindex(df1.index) 
# If log values are negative or too extreme, clip them to avoid issues
predictions_ARIMA_log = np.clip(predictions_ARIMA_log, -700, 700)

# Now apply np.exp() to recover the original data

predictions_ARIMA = np.exp(predictions_ARIMA_log)

# Ensure indices are in datetime format
df1.index = pd.to_datetime(df1.index)
predictions_ARIMA.index = pd.to_datetime(predictions_ARIMA.index)

# Align the data and predictions by index
predictions_ARIMA = predictions_ARIMA.reindex(df1.index, method='ffill')


# Reindex predictions to align with original data
#predictions_ARIMA = predictions_ARIMA.reindex(df1.index)

# Drop NaN values if any
#predictions_ARIMA = predictions_ARIMA.dropna()




# Check for Inf or NaN values in both df1 and predictions_ARIMA
print(np.any(np.isinf(df1)))
print(np.any(np.isinf(predictions_ARIMA_log)))

print(np.any(np.isnan(df1)))
print(np.any(np.isnan(predictions_ARIMA)))
print(np.any(np.isinf(predictions_ARIMA)))


predictions_ARIMA.replace([np.inf, -np.inf], np.nan, inplace=True)
print(np.any(np.isinf(predictions_ARIMA)))


# ploting the original data 
plt.plot(df1.values,color="blue")
plt.title("Original Data")
plt.show()


# ploting the predications 
plt.plot(predictions_ARIMA.values,color="red")
plt.show()




# ploting the graph 
plt.plot(df1.values, color="blue", label="Original Data")
plt.plot(predictions_ARIMA.values, color="red", label="Predicted Data")
plt.legend()
plt.title('Actual vs Predicted Data')
plt.show()


x = results_ARIMA.forecast(steps=120)  # x is a tuple, x[0] contains the forecasted log values
forecasted_log_values = x  # Extract the predicted log values
#forecasted_log_values = x[0].iloc[:]  # This ensures position-based access
forecasted_original_values = np.exp(forecasted_log_values)

forecasted_original_values_int = np.round(forecasted_original_values).astype(int)
print("Predicted values (as integers):", forecasted_original_values_int)
