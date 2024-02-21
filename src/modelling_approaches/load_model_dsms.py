import os
import pickle
import numpy as np
from src.tools.config import cfg
from src.predict.calc_steel_stocks import get_np_steel_stocks_with_prediction
from src.odym_extension.MultiDim_DynamicStockModel import MultiDim_DynamicStockModel
from src.read_data.load_data import load_lifetimes
from src.base_model.model_tools import calc_change_timeline


def load_model_dsms(country_specific, do_past_not_future, model_type=cfg.model_type, do_econ_model=cfg.do_model_economy,
                    recalculate=cfg.recalculate_data, forming_fabrication=None, indirect_trade=None):
    file_name = _get_dsms_file_name(country_specific, do_past_not_future, model_type, do_econ_model)
    file_path = os.path.join(cfg.data_path, 'models', file_name)
    if os.path.exists(file_path) and not recalculate:
        dsms = pickle.load(open(file_path, "rb"))
        return dsms
    else:
        dsms = _get_dsms(country_specific, do_past_not_future, model_type, do_econ_model, forming_fabrication,
                         indirect_trade)
        pickle.dump(dsms, open(file_path, "wb"))
        return dsms


def get_dsm_data(dsms):
    dsms = np.array(dsms)
    inflows = np.array([dsms[idx].i for idx in np.ndindex(dsms.shape)])
    inflows = inflows.reshape(dsms.shape + (len(inflows[0]),))
    inflows = np.moveaxis(inflows, -1, 0)
    stocks = np.array([dsms[idx].s for idx in np.ndindex(dsms.shape)])
    stocks = stocks.reshape(dsms.shape + (len(stocks[0]),))
    stocks = np.moveaxis(stocks, -1, 0)
    outflows = np.array([dsms[idx].o for idx in np.ndindex(dsms.shape)])
    outflows = outflows.reshape(dsms.shape + (len(outflows[0]),))
    outflows = np.moveaxis(outflows, -1, 0)

    return inflows, stocks, outflows


def get_dsm_lifetimes(dsms):
    dsms = np.array(dsms)
    lifetime_means = np.array([dsms[idx].lt['Mean'] for idx in np.ndindex(dsms.shape)])
    lifetime_means = lifetime_means.reshape(dsms.shape + (len(lifetime_means[0]),))
    lifetime_means = np.moveaxis(lifetime_means, -1, 0)
    lifetime_sds = np.array([dsms[idx].lt['StdDev'] for idx in np.ndindex(dsms.shape)])
    lifetime_sds = lifetime_sds.reshape(dsms.shape + (len(lifetime_sds[0]),))
    lifetime_sds = np.moveaxis(lifetime_sds, -1, 0)

    return lifetime_means, lifetime_sds


def _get_dsms(country_specific, do_past_not_future, model_type, do_econ_model, forming_fabrication=None,
              indirect_trade=None):
    if do_past_not_future:
        if model_type == 'stock':
            from src.modelling_approaches.model_2_stock_driven import get_stock_driven_past_dsms
            dsms = get_stock_driven_past_dsms(country_specific)
            return dsms
        elif model_type == 'inflow':
            from src.modelling_approaches.model_1_inflow_driven import get_inflow_driven_past_dsms
            dsms = get_inflow_driven_past_dsms(country_specific, forming_fabrication, indirect_trade)
            return dsms
        elif model_type == 'change':
            from src.modelling_approaches.model_3_change_driven import get_change_driven_past_dsms
            dsms = get_change_driven_past_dsms(country_specific, forming_fabrication, indirect_trade)
            return dsms
        else:
            raise RuntimeError()
    else:  # do future
        dsms = _calc_future_dsms(country_specific, model_type, forming_fabrication, indirect_trade, do_econ_model)
        return dsms
    raise RuntimeError()


def _calc_future_dsms(country_specific, model_type, forming, indirect_trade, do_econ_model):
    past_dsms = load_model_dsms(country_specific, do_past_not_future=True, model_type=model_type,
                                forming_fabrication=forming, indirect_trade=indirect_trade, do_econ_model=do_econ_model)
    past_inflows, past_stocks, past_outflows = get_dsm_data(past_dsms)
    past_lifetime_means, past_lifetime_sds = get_dsm_lifetimes(past_dsms)
    print(f'Model Type: {model_type}')
    stocks = get_np_steel_stocks_with_prediction(country_specific=False,
                                                 get_per_capita=False,
                                                 stocks=past_stocks)

    past_lifetime_means = np.maximum(0.001, past_lifetime_means)
    past_lifetime_sds = np.maximum(0.0003, past_lifetime_sds)

    lifetime_means = np.ones(stocks.shape[:3])  # ignore scenario axis
    lifetime_sds = np.ones_like(lifetime_means)
    # initiate lifetime with most recent lifetime
    lifetime_means = np.einsum('trg,rg->trg', lifetime_means, past_lifetime_means[-1])
    lifetime_means[:109] = past_lifetime_means
    lifetime_sds = np.einsum('trg,rg->trg', lifetime_sds, past_lifetime_sds[-1])
    lifetime_sds[:109] = past_lifetime_sds
    future_dsms = get_stock_based_dsms(stocks, 1900, 2100,
                                       do_scenarios=True,
                                       lt_mean=lifetime_means,
                                       lt_sd=lifetime_sds)

    # Note: future stocks might change if inflow is changed

    return future_dsms


def get_stock_based_dsms(stocks, start_year, end_year, do_scenarios, lt_mean=None, lt_sd=None):
    stocks = np.moveaxis(stocks, 0, -1)  # move time axis so that stocks are in a fitting format for list comprehension
    mean, std_dev = load_lifetimes()
    years = np.arange(start_year, end_year + 1)
    do_change_inflow = cfg.do_change_inflow and start_year == 2000 and end_year == 2100
    if do_change_inflow:
        inflow_change_timeline = calc_change_timeline(cfg.inflow_change_factor, cfg.inflow_change_base_year)

    if do_scenarios:  # decide whether to make this a parameter or check via shape dimensiosn
        dsms = [[[_create_stock_based_dsm(scenario_stocks,
                                          years,
                                          [mean[region_idx, cat_idx]] if lt_mean is None else lt_mean[:, region_idx,
                                                                                              cat_idx],
                                          [std_dev[region_idx, cat_idx]] if lt_sd is None else lt_sd[:, region_idx,
                                                                                               cat_idx],
                                          inflow_change_timeline if do_change_inflow else None)
                  for scenario_idx, scenario_stocks in enumerate(cat_stocks)]
                 for cat_idx, cat_stocks in enumerate(region_stocks)]
                for region_idx, region_stocks in enumerate(stocks)]
    else:  # do not iterate through scenarios
        dsms = [[_create_stock_based_dsm(cat_stocks, years, [mean[region_idx, cat_idx]], [std_dev[region_idx, cat_idx]],
                                         inflow_change_timeline if do_change_inflow else None)
                 for cat_idx, cat_stocks in enumerate(region_stocks)]
                for region_idx, region_stocks in enumerate(stocks)]
    return dsms


def _create_stock_based_dsm(stocks, years, lifetime_mean, lifetime_sd, inflow_change=None):
    steel_stock_dsm = MultiDim_DynamicStockModel(t=years,
                                                 s=stocks,
                                                 lt={'Type': 'Normal', 'Mean': lifetime_mean,
                                                     'StdDev': lifetime_sd})

    steel_stock_dsm.compute_all_stock_driven()
    if inflow_change is not None:
        inflows = steel_stock_dsm.i
        inflows = inflows * inflow_change
        steel_stock_dsm = MultiDim_DynamicStockModel(t=years,
                                                     i=inflows,
                                                     lt={'Type': 'Normal', 'Mean': [lifetime_mean],
                                                         'StdDev': [lifetime_sd]})
        steel_stock_dsm.compute_all_inflow_driven()
    return steel_stock_dsm


def _get_dsms_file_name(country_specific, do_past_not_future, model_type, do_econ_model):
    time_param = 'PAST' if do_past_not_future else 'FUTURE'
    area_param = 'COUNTRY' if country_specific else cfg.region_data_source.upper()
    econ_param = 'ECON' if do_econ_model else 'BASE'
    if do_past_not_future:
        # in the past, econ base_model is never applied
        econ_param = ''
    else:
        econ_param = '_' + econ_param
    model_param = model_type.upper()
    file_name = f'model_dsms_{model_param}_{time_param}_{area_param}{econ_param}.p'
    return file_name
