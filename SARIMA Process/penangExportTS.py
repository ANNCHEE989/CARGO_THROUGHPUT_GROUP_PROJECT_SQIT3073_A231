import os
os.system('cls')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import itertools
import warnings
from sklearn.metrics import mean_absolute_percentage_error
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

#Filter Export(Penang)
penang_export = cargo_throughput_data.loc[:,cargo_throughput_data.columns.isin(['Date','Export(Penang)'])].reset_index(drop=True)
penang_export.head()
print("Export value in Penang")
print(penang_export)

#plot original time series data
plt.figure(figsize=(10,6))
plt.plot(penang_export['Date'], penang_export['Export(Penang)'])
plt.title('Cargo export value in Penang')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.show()

#set traning and testing set
penang_export_train = penang_export[:round(len(penang_export)*70/100)]
penang_export_test = penang_export[round(len(penang_export)*70/100):]

#testing stationarity
test_stationarity(penang_export,'Export(Penang)')

#transformation
penang_export_df = penang_export[['Date','Export(Penang)']]
penang_export_df['shift']=penang_export_df['Export(Penang)'].shift()
penang_export_df['shiftDiff']=penang_export_df['Export(Penang)']-penang_export_df['shift']
print(penang_export_df)

#re-test stationarity
test_stationarity(penang_export_df.dropna(),'shiftDiff')

#autocorrelation factor
fig,ax= plt.subplots(2, figsize=(10,6))
ax[0] = plot_acf(penang_export_df['shiftDiff'].dropna(), ax=ax[0], lags = 15)
ax[1] = plot_pacf(penang_export_df['shiftDiff'].dropna(), ax=ax[1], lags = 15)
plt.show()

#decomposition
penang_export_df['Date'] = pd.to_datetime(penang_export_df['Date'])
penang_export_df = penang_export_df.set_index('Date')
decomp = sm.tsa.seasonal_decompose(penang_export_df['Export(Penang)'],model='additive')
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
            model = sm.tsa.statespace.SARIMAX(penang_export_train['Export(Penang)'],
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
model_penang_export = sm.tsa.statespace.SARIMAX(penang_export_train['Export(Penang)'],order=(0,1,1),seasonal_order=(0,1,1,12),
                                  enforce_stationarity=False,enforce_invertibility=False)
result = model_penang_export.fit()


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
penang_export_test['Date'] = pd.to_datetime(penang_export_test['Date'])
forecast_model = result.get_forecast(steps=len(penang_export_test))
predict_mean = forecast_model.predicted_mean
plt.figure(figsize=(10,6))
plt.plot(penang_export['Date'],penang_export['Export(Penang)'],label='Training Set')
plt.plot(penang_export_test['Date'],penang_export_test['Export(Penang)'],label='Testing Set')
plt.plot(penang_export_test['Date'], predict_mean,label='Forecast')
plt.title('Forecast Model of Penang Cargo Export Value')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.legend()
plt.show()

#check accuracy
actual_data = np.array(penang_export_test['Export(Penang)'])
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
plt.plot(penang_export['Date'],penang_export['Export(Penang)'],label='Original Data')
plt.plot(future_df['Date'],future_df['Prediction'],label='Forecast')
plt.title('Future Cargo Export Value Prediction')
plt.xlabel('Date')
plt.ylabel('\'000 Tan Metrik(Freightweight)')
plt.legend()
plt.show()