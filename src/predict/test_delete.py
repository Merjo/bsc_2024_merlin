import matplotlib.pyplot as plt
import numpy as np
from darts.utils import timeseries_generation as tg

np.random.seed(42)

LENGTH = 3 * 365  # 3 years of daily data

# Melting: a sine with yearly periodicity and additive white noise
melting = (0.9 * tg.sine_timeseries(length=LENGTH, value_frequency=(1 / 365),
                                    freq='D',
                                    column_name='melting')
           + 0.1 * tg.gaussian_timeseries(length=LENGTH, freq='D'))

# Rainfalls: a sine with bi-weekly periodicity and additive white noise
rainfalls = (0.5 * tg.sine_timeseries(length=LENGTH,
                                      value_frequency=(1 / 14),
                                      freq='D',
                                      column_name='rainfall')
             + 0.5 * tg.gaussian_timeseries(length=LENGTH, freq='D'))

# We shift the melting by 5 days
melting_contribution = 0.5 * melting.shift(5)

# We compute similar contribution from the rainfalls
all_contributions = [melting_contribution] + [0.1 * rainfalls.shift(lag) for lag in range(5)]

# We compute the final flow as the sum of everything; trimming series so they
# all have the same start time
flow = sum([series[melting_contribution.start_time():][:melting.end_time()]
            for series in all_contributions]).with_columns_renamed('melting', 'flow')

# add some white noise
flow += 0.1 * tg.gaussian_timeseries(length=len(flow))

plt.figure(figsize=(12, 5))
melting.plot()
rainfalls.plot()
flow.plot(lw=4)

from darts.metrics import rmse

# We first set aside the first 80% as training series:
flow_train, _ = flow.split_before(0.8)


def eval_model(model, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=flow,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8,
                                          retrain=False,
                                          verbose=True,
                                          forecast_horizon=10)

    flow[-len(backtest) - 100:].plot()
    backtest.plot(label='backtest (n=10)')
    print('Backtest RMSE = {}'.format(rmse(flow, backtest)))


from darts.models import RNNModel

rnn_rain = RNNModel(input_chunk_length=30,
                    training_length=40,
                    n_rnn_layers=2)

rnn_rain.fit(flow_train,
             future_covariates=rainfalls,
             epochs=10,
             verbose=True)

eval_model(rnn_rain,
           future_covariates=rainfalls)

prediction = rnn_rain.predict(n=3,
                              series=flow_train,
                              future_covariates=rainfalls)
flow_vals = flow_train.values()
values = prediction.values()

plt.plot(np.arange(874), np.concatenate([flow_vals[:, 0], values[:, 0]], axis=0))
plt.show()

a = 0
