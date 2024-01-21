import os
os.system('cls')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import itertools
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def test_stationarity(dframe,var):
    dframe['rollMean']=dframe[var].rolling(window=12).mean()
    dframe['rollSTD']=dframe[var].rolling(window=12).std()
    adfTest = adfuller(dframe[var],autolag='AIC')
    stat_result = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    print(stat_result)
    for key,values in adfTest[4].items():
        print('criticality',key,':',values)
    plt.plot(dframe['Date'],dframe[var],label='data')
    plt.plot(dframe['Date'],dframe['rollMean'],label='rolling mean')
    plt.plot(dframe['Date'],dframe['rollSTD'],label='rolling standard deviation')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

cargo_throughput_data = pd.read_excel(io = "cargo data (2018-2023).xlsx",engine = 'openpyxl', sheet_name="Sheet1")
cargo_throughput_data['Date'] = pd.to_datetime(cargo_throughput_data['Date'])
print('Dataset of Cargo Throughput at Selected Ports - Peningsular Malaysia:')
print(cargo_throughput_data)

#Filter Import(Port Dickson)
portd_import = cargo_throughput_data.loc[:,cargo_throughput_data.columns.isin(['Date','Import(Port Dickson)'])].reset_index(drop=True)
portd_import.head()
print("Import value in Port Dickson")
print(portd_import)

#plot original time series data
plt.figure(figsize=(10,6))
plt.plot(portd_import['Date'], portd_import['Import(Port Dickson)'])
plt.title('Cargo export value in Port Dickson')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.show()

#set traning and testing set
portd_import_train = portd_import[:round(len(portd_import)*70/100)]
portd_import_test = portd_import[round(len(portd_import)*70/100):]

#testing stationarity
test_stationarity(portd_import,'Import(Port Dickson)')

#transformation
portd_import_df = portd_import[['Date','Import(Port Dickson)']]
portd_import_df['shift']=portd_import_df['Import(Port Dickson)'].shift()
portd_import_df['shiftDiff']=portd_import_df['Import(Port Dickson)']-portd_import_df['shift']
print(portd_import_df)

#re-test stationarity
test_stationarity(portd_import_df.dropna(),'shiftDiff')

#autocorrelation factor
fig,ax= plt.subplots(2, figsize=(10,6))
ax[0] = plot_acf(portd_import_df['shiftDiff'].dropna(), ax=ax[0], lags = 15)
ax[1] = plot_pacf(portd_import_df['shiftDiff'].dropna(), ax=ax[1], lags = 15)
plt.show()

#decomposition
portd_import['Date'] = pd.to_datetime(portd_import['Date'])
portd_import_df = portd_import.set_index('Date')
decomp = sm.tsa.seasonal_decompose(portd_import_df['Import(Port Dickson)'],model='additive')
decomp.plot()
plt.show()

#find pdq and AIC values
p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
metric_aic_dict = dict()

for pm in pdq:
    for pm_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(portd_import_train['Import(Port Dickson)'],
                                              order=pm,
                                              seasonal_order=pm_seasonal,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            result = model.fit()
            print('ARIMA{}x{}12-AIC:{}'.format(pm,pm_seasonal,result.aic))
            metric_aic_dict.update({(pm,pm_seasonal):result.aic})
        except:
            continue
#sort the AIC value in ascending order
print({k:v for k, v in sorted(metric_aic_dict.items(),key=lambda x:x[1])})

#fit the model using lowest AIC
model_portd_import = sm.tsa.statespace.SARIMAX(portd_import_train['Import(Port Dickson)'],order=(0,0,0),seasonal_order=(0,1,1,12),
                                  enforce_stationarity=False,enforce_invertibility=False)
result = model_portd_import.fit()
print(result.summary().tables[1])

#check the residuals
check_residuals = result.resid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Time series plot of residuals
axes[0, 0].plot(check_residuals)
axes[0, 0].set_title('Residuals Time Series Plot')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals')

# Histogram of residuals
axes[0, 1].hist(check_residuals, bins=10)
axes[0, 1].set_title('Histogram of Residuals')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')

# Q-Q plot to check normality
sm.qqplot(check_residuals, line='s', ax=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot of Residuals')

# ACF plot of residuals
sm.graphics.tsa.plot_acf(check_residuals, lags=15, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# PACF plot of residuals
sm.graphics.tsa.plot_pacf(check_residuals, lags=15, ax=axes[1, 1])
axes[1, 1].set_title('PACF of Residuals')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Partial Autocorrelation')

# Remove empty subplot
fig.delaxes(axes[1, 2])

# Adjust layout
plt.tight_layout()
plt.show()

#build forecast model
portd_import_test['Date'] = pd.to_datetime(portd_import_test['Date'])
forecast_model = result.get_forecast(steps=len(portd_import_test))
predict_mean = forecast_model.predicted_mean
plt.figure(figsize=(10,6))
plt.plot(portd_import['Date'],portd_import['Import(Port Dickson)'],label='Training Set')
plt.plot(portd_import_test['Date'],portd_import_test['Import(Port Dickson)'],label='Testing Set')
plt.plot(portd_import_test['Date'], predict_mean,label='Forecast')
plt.title('Forecast Model of Port Dickson Cargo Export Value')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.legend()
plt.show()

#check accuracy
actual_data = np.array(portd_import_test['Import(Port Dickson)'])
predict_data = np.array(predict_mean)
squared_error = (predict_data - actual_data)**2
mse = squared_error.mean()
rmse = np.sqrt(mse)
print('Root Mean Squared Error: ', rmse)

#check accuracy(MAPE)
mape = 100*(mean_absolute_percentage_error(actual_data,predict_data))
print('Mean Absolute Percentage Error: ',mape)

#Predict future value
future_pred= result.get_forecast(steps=15)
future_df=pd.DataFrame(future_pred.predicted_mean)
future_df.columns = ['Prediction']
future_df['Date']=pd.date_range(start='2023-10-31',end='2024-12-31',freq='M')
future_df = future_df[['Date','Prediction']]
future_df.set_index('Date')
print(future_df)

#Plot time series
plt.figure(figsize=(10,6))
plt.plot(portd_import['Date'],portd_import['Import(Port Dickson)'],label='Original Data')
plt.plot(future_df['Date'],future_df['Prediction'],label='Forecast')
plt.title('Future Cargo Export Value Prediction')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.legend()
plt.show()