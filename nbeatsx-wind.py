import pandas as pd
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast

def load_data(filename=r'./data.xlsx'):
    df = pd.read_excel(filename, sheet_name='Sheet2')
    df = df.drop_duplicates(subset='time', keep='first')
    df.index = [i for i in range(len(df.index))]
    print(df)
    date_range = pd.date_range(start='2020-01-01 00:00:00', end='2021-05-31 23:45:00', freq='15min')
    df = df.reindex(date_range, method='ffill')
    df['trend1'] = [i % 96 for i in range(len(date_range))]
    df['trend2'] = [i % (96 * 366) // 96 for i in range(len(date_range))]
    df.rename(columns={'time':'ds'})
    return df

if __name__ == '__main__':
    df = load_data()
    print(df)
    y_train = df[:-12]
    y_test = df[-12:]

    model = NBEATSx(h=12, input_size=84,
                #loss=MQLoss(level=[80, 90]),
                loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                scaler_type='robust',
                stat_exog_list=[],
                futr_exog_list=['trend1', 'trend2'],
                max_steps=200,
                val_check_steps=10,
                early_stop_patience_steps=2)
    
    nf = NeuralForecast(
        models=[model],
        freq='M'
    )

    nf.fit(df=y_train, val_size=12)
    y_hat = nf.predict(futr_df=y_test)
    # Plot quantile predictions
    print(y_hat)

    plot_df = pd.concat([y_test, y_hat], axis=1)
    plot_df = pd.concat([y_train, plot_df])

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
