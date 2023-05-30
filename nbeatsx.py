import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
import neuralforecast.tsdataset
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = NBEATSx(h=12, input_size=24,
                #loss=MQLoss(level=[80, 90]),
                loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                scaler_type='robust',
                stat_exog_list=['airline1'],
                futr_exog_list=['trend'],
                max_steps=200,
                val_check_steps=10,
                early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='M'
)
print(Y_train_df)
# pd.set_option('display.max_rows', None)
# print(Y_train_df)
# print(AirPassengersStatic)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
Y_hat_df = nf.predict(futr_df=Y_test_df)
# Plot quantile predictions
Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=['unique_id','ds'])

plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline2'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NBEATSx'], c='purple', label='mean')
plt.plot(plot_df['ds'], plot_df['NBEATSx-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df['NBEATSx-lo-90'][-12:].values, 
                 y2=plot_df['NBEATSx-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')

print(plot_df)

mape = 0
y1=plot_df['NBEATSx'][-12:].values
y2=plot_df['y'][-12:].values

for i in range(12):
    mape += abs(y1[i] - y2[i]) / y2[i]

print(mape / 12)

plt.legend()
plt.grid()
plt.plot()
plt.show()