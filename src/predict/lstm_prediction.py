import statistics

import numpy as np
import pandas as pd
from src.predict.prediction_tools import split_future_stocks_to_base_year_categories, \
    copy_stocks_across_scenarios
from src.read_data.load_data import load_region_names_list
from src.tools.config import cfg
from darts.metrics import mape
from darts.models import RNNModel
from darts import TimeSeries
from matplotlib import pyplot as plt
import os
import datetime

do_normalize_stocks = True
n_epochs = 700
do_show_plot = False
do_create_new_model = False
input_chunk_length = 95  # 109-14 time steps for future
output_chunk_length = 14  # or 92 ?
n_rnn_layers = 8

cfg.n_epochs = n_epochs
cfg.n_rnn_layers = n_rnn_layers


def predict_lstm(stocks, gdp, pop, include_scenarios=cfg.include_gdp_and_pop_scenarios_in_prediction):
    past_stocks_by_category = stocks.copy()
    stocks = np.sum(stocks, axis=2)

    if not include_scenarios:
        future_stocks = _lstm_stock_curve(stocks, gdp[:, :, 1], pop[:, :, 1], do_create_new_model=do_create_new_model)
    else:
        future_stocks = np.array(
            [_lstm_stock_curve(stocks, gdp[:, :, i], pop[:, :, i], do_create_new_model=do_create_new_model) for
             i, scenario in enumerate(cfg.scenarios)])
        future_stocks = np.moveaxis(future_stocks, 0, -1)

    future_stocks = split_future_stocks_to_base_year_categories(past_stocks_by_category, future_stocks,
                                                                is_future_stocks_with_scenarios=include_scenarios)
    if include_scenarios:
        past_stocks_by_category = copy_stocks_across_scenarios(past_stocks_by_category)

    stocks = np.concatenate([past_stocks_by_category, future_stocks], axis=0)

    return stocks


def _lstm_stock_curve(stocks, gdp, pop, do_create_new_model):
    if not do_create_new_model:
        model, model_mape = _load_model()
        if model is None:
            do_create_new_model = True

    if do_normalize_stocks:
        # stocks are normalized to be within the range 0 and 1, with 'room' to rise up to 40 tonnes / capita
        min_stocks = np.min(stocks, axis=0)
        max_stocks = np.ones_like(min_stocks) * 40
        stocks = (stocks - min_stocks) / (max_stocks - min_stocks)

        min_gdp = np.min(gdp, axis=0)
        max_gdp = np.max(gdp, axis=0)
        gdp = (gdp - min_gdp) / (max_gdp - min_gdp)

    stock_times = pd.date_range("1900-01-01", periods=109, freq="Y")
    gdp_times = pd.date_range("1900-01-01", periods=201, freq="Y")

    regions = load_region_names_list()

    ts_stocks = TimeSeries.from_times_and_values(stock_times, stocks)
    ts_stocks_train, _ = ts_stocks.split_before(0.8)
    ts_stocks_list = [TimeSeries.from_times_and_values(stock_times, stocks[:, r]) for r, region in enumerate(regions)]

    ts_gdp_list = [TimeSeries.from_times_and_values(gdp_times, gdp[:, r]) for r, region in enumerate(regions)]

    ts_covariates = ts_gdp_list
    if do_create_new_model:
        ts_stocks_list_without_ref = ts_stocks_list[:9] + ts_stocks_list[9 + 1:]
        ts_covariates_without_ref = ts_covariates[:9] + ts_covariates[9 + 1:]
        model, model_mape = _create_new_model(
            ts_stocks_list_without_ref if cfg.model_type == 'inflow' else ts_stocks_list,
            ts_covariates_without_ref if cfg.model_type == 'inflow' else ts_covariates)

    prediction = model.predict(n=92,
                               series=ts_stocks_list,
                               future_covariates=ts_gdp_list)
    values = np.array([prediction[r].values() for r, region in enumerate(regions)]).transpose()
    orig_times = np.concatenate([stocks, values[0]], axis=0)
    if do_normalize_stocks:
        orig_stocks = orig_times * (max_stocks - min_stocks) + min_stocks
    else:
        orig_stocks = orig_times

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    f_name = f'{model_mape:.2f}%_{timestamp}_{cfg.steel_data_source}_{cfg.region_data_source}_' \
             f'{cfg.model_type}_{cfg.n_epochs}_{cfg.n_rnn_layers}_{cfg.hidden_dim}_{input_chunk_length}_{output_chunk_length}'
    base_path = os.path.join(cfg.data_path, 'models', 'lstm_models')
    pic_path = os.path.join(base_path, f'{f_name}.png')
    model_path = os.path.join(base_path, f'{f_name}.pt')
    if do_create_new_model or do_show_plot:
        plt.plot(np.arange(1900, 2101), orig_stocks)
        plt.legend(regions)
        plt.xlabel('Time (y)')
        plt.ylabel('Steel (t)')
        plt.title('Steel stocks per capita with normal and smoothed (--) predictions \n'
                  f'{cfg.region_data_source}_{cfg.steel_data_source}_{cfg.model_type}_{cfg.n_epochs}_{cfg.n_rnn_layers}_{cfg.hidden_dim}')
    if do_create_new_model:
        model.save(model_path)
        plt.savefig(pic_path, dpi=300)
    if do_show_plot:
        plt.show()
    plt.clf()  # clear figure if doing several tests
    return orig_stocks[109:]


def _load_model():
    base_path = os.path.join(cfg.data_path, 'models', 'lstm_models')
    files = os.listdir(base_path)
    models = [file.split('_') for file in files if file.endswith('.pt')]
    models = [model for model in models if
              model[3] == cfg.steel_data_source
              and model[4] == cfg.region_data_source and model[5] == cfg.model_type and float(model[0][:-1]) < 10]
    # model needs to match desired steel and region data source and model type and have an accuracy of at least 10 %

    if len(models) == 0:
        return None, None
    models = ['_'.join(model) for model in models]
    final = min(models)
    final_path = os.path.join(base_path, final)
    print(final_path)
    model_mape = float(final.split('%')[0])
    model = RNNModel.load(final_path)
    return model, model_mape


def _create_new_model(ts_stocks_list, ts_covariates):
    print(
        f'Creating new model with {cfg.n_rnn_layers} rnn layers, {cfg.hidden_dim} hidden dim size and {cfg.n_epochs} epochs.')
    model = RNNModel(model='LSTM',
                     input_chunk_length=input_chunk_length,
                     output_chunk_length=output_chunk_length,
                     n_rnn_layers=cfg.n_rnn_layers)
    model.fit(ts_stocks_list,
              future_covariates=ts_covariates,
              epochs=cfg.n_epochs,
              verbose=True)
    model_mape = eval_model(model, ts_stocks_list,
                            future_covariates=ts_covariates)

    return model, model_mape


def eval_model(model, stocks, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=stocks,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=95,  # 2008-(2022-2009)=1995))
                                          retrain=False,
                                          verbose=True)

    # stocks[-len(backtest) - 100:].plot()
    # backtest.plot(label='backtest (n=10)')
    mape_list = mape(stocks, backtest)
    average_mape = statistics.fmean(mape_list)
    print('Backtest MAPE = {}'.format(average_mape))
    return average_mape


if __name__ == '__main__':
    normal = True
    from src.predict.calc_steel_stocks import test

    if normal:
        test(strategy='LSTM', do_visualize=True)
    else:
        region_tests = ['REMIND']  # Options: ['Pauliuk', 'REMIND']
        data_tests = ['IEDatabase']  # Options: ['Mueller', 'IEDatabase', 'ScrapAge']
        model_types = ['stock', 'inflow', 'change']
        n_epochen = [1000]
        n_rnn_layers = [3, 6, 9, 12]
        hidden_dims = [5, 15, 25, 35]
        for region in region_tests:
            cfg.region_data_source = region
            for data_set in data_tests:
                cfg.steel_data_source = data_set
                for model_type in model_types:
                    cfg.model_type = model_type
                    for n_epoch in n_epochen:
                        cfg.n_epochs = n_epoch
                        for n_rnn in n_rnn_layers:
                            cfg.n_rnn_layers = n_rnn
                            for hidden_dim in hidden_dims:
                                cfg.hidden_dim = hidden_dim
                                print(f'\n\nTest {region} {data_set} {model_type} {n_epoch} {n_rnn} {hidden_dim}\n\n')
                                test(strategy='LSTM', do_visualize=False)
