import numpy as np
from math import e, pi, sqrt
from src.base_model.load_params import get_cullen_fabrication_yield
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data
from src.modelling_approaches.load_data_for_approaches import get_past_production_trade_forming_fabrication, \
    get_past_stocks
from src.odym_extension.MultiDim_DynamicStockModel import MultiDim_DynamicStockModel
from src.read_data.load_data import load_lifetimes


def get_change_driven_model_upper_cycle(country_specific=False):
    production, trade, forming_fabrication, indirect_trade = \
        get_past_production_trade_forming_fabrication(country_specific)

    past_dsms = load_model_dsms(country_specific=country_specific,
                                do_past_not_future=True,
                                model_type='change',
                                do_econ_model=False,
                                recalculate=False,
                                forming_fabrication=forming_fabrication,
                                indirect_trade=indirect_trade)
    inflows, stocks, outflows = get_dsm_data(past_dsms)
    fabrication_use = inflows - indirect_trade

    return production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows


def _calc_stock_change(stocks):
    stock_change = np.zeros_like(stocks)
    stock_change[0] = stocks[0]
    stock_change[1:] = np.diff(stocks, axis=0)
    return stock_change


def get_change_driven_past_dsms(country_specific, fabrication=None, indirect_trade=None):
    if fabrication is None or indirect_trade is None:
        production, trade, forming_fabrication, indirect_trade = \
            get_past_production_trade_forming_fabrication(country_specific)
        fabrication = forming_fabrication
    stocks = get_past_stocks(country_specific)
    stock_change = _calc_stock_change(stocks)
    fabrication_yield = get_cullen_fabrication_yield()
    normal_lifetime_mean = load_lifetimes()[0]
    max_lifetime_mean = 5 * normal_lifetime_mean
    min_lifetime_lambda = _switch_initial_lambda_mean(max_lifetime_mean)
    max_lifetime_lambda = _switch_initial_lambda_mean(normal_lifetime_mean * 0.2)

    # calculation year 0 (1900)

    lifetime_matrix_shape = (stocks.shape[0],) + stocks.shape
    lambda_matrix = np.zeros(lifetime_matrix_shape)

    mean = np.zeros_like(stocks)
    mean[0] = normal_lifetime_mean / 2
    _update_lambda_matrix(lambda_matrix, mean[0], 0)

    inflows = np.zeros_like(stocks)
    inflows_1900 = stock_change[0] / (1 - lambda_matrix[0, 0])
    direct_inflows_1900 = inflows_1900 - indirect_trade[0]
    fabrication_by_category_1900 = np.einsum('rg,g->rg', direct_inflows_1900,
                                             1 / fabrication_yield)
    fabrication[0] = np.sum(fabrication_by_category_1900, axis=1)
    inflows[0] = inflows_1900

    # iterative calculation up to 2008

    for t in range(1, 109):
        outflow_past_cohorts_t = _calc_outflow_past_cohorts_t(inflows, lambda_matrix, t)

        mean_t, inflow_t, stock_change_t = _calc_time_step(t, outflow_past_cohorts_t,
                                                           stock_change[t],
                                                           fabrication[t], indirect_trade[t],
                                                           fabrication_yield,
                                                           min_lifetime_lambda,
                                                           max_lifetime_lambda,
                                                           lambda_matrix)

        mean[t] = mean_t
        inflows[t] = inflow_t
        stock_change[t] = stock_change_t

    outflows = inflows - stock_change
    stocks = np.cumsum(stock_change, axis=0)

    inflows = np.moveaxis(inflows, 0, 2)  # move time axis to the end to iterate more easily through inflows

    years = np.arange(1900, 2009)
    dsms = [[_create_change_driven_past_dsm(cat_inflows,
                                            stocks[:, region_idx, cat_idx],
                                            outflows[:, region_idx, cat_idx],
                                            mean[:, region_idx, cat_idx],
                                            years)
             for cat_idx, cat_inflows in enumerate(region_inflows)]
            for region_idx, region_inflows in enumerate(inflows)]

    return dsms


def _calc_time_step(t, outflow_past_cohorts_t, stock_change_t, fabrication_t, indirect_trade_t, fabrication_yield,
                    min_lifetime_lambda, max_lifetime_lambda, lambda_matrix):
    fx_for_x_calculation = (stock_change_t + outflow_past_cohorts_t - indirect_trade_t) / fabrication_yield
    x_t = _calc_x_as_share_from_fx(fx_for_x_calculation)
    inflow_t = np.einsum('r,rg,g->rg', fabrication_t, x_t, fabrication_yield) + indirect_trade_t

    stock_change_t = _adapt_stock_change_t(inflow_t, stock_change_t, outflow_past_cohorts_t, min_lifetime_lambda,
                                           max_lifetime_lambda)

    lambda_t = 1 - np.divide((stock_change_t + outflow_past_cohorts_t), inflow_t, out=np.ones_like(stock_change_t),
                             where=inflow_t != 0)

    mean_t = _switch_initial_lambda_mean(lambda_t)
    _update_lambda_matrix(lambda_matrix, mean_t, t)
    return mean_t, inflow_t, stock_change_t


def _calc_outflow_past_cohorts_t(inflows, lambda_matrix, t):
    outflow_past_cohorts_t = inflows[:t] * lambda_matrix[t, :t]
    outflow_past_cohorts_t = np.sum(outflow_past_cohorts_t, axis=0)
    return outflow_past_cohorts_t


def _switch_initial_lambda_mean(mean_or_lambda):
    # if either mean or lambda is zero, this means that the inflow is zero too and hence the other can be zero as well
    # to avoid nan values

    lambda_or_mean = np.divide(np.ones_like(mean_or_lambda, dtype='float64'),
                               mean_or_lambda * sqrt(2) * pi * 0.3 * e ** (1 / (2 * 0.3 ** 2)),
                               out=np.zeros_like(mean_or_lambda, dtype='float64'),
                               where=mean_or_lambda != 0)
    return lambda_or_mean


def _calc_x_as_share_from_fx(fx):
    fx[np.sign(fx) < 0] = 0
    fx_max_divisor = np.expand_dims(np.sum(fx, axis=-1), axis=-1)
    x = np.divide(fx, fx_max_divisor, out=np.ones_like(fx) * 0.25, where=fx_max_divisor != 0)
    return x


def _adapt_stock_change_t(inflow_t, stock_change_t, outflow_past_cohorts_t, min_lifetime_lambda, max_lifetime_lambda):
    max_stock_change_t = inflow_t * (1 - min_lifetime_lambda) - outflow_past_cohorts_t
    min_stock_change_t = inflow_t * (1 - max_lifetime_lambda) - outflow_past_cohorts_t

    stock_change_t = np.minimum(max_stock_change_t, stock_change_t)
    stock_change_t = np.maximum(min_stock_change_t, stock_change_t)

    stock_change_t[inflow_t == 0] = -outflow_past_cohorts_t[inflow_t == 0]

    return stock_change_t


def _update_lambda_matrix(lambda_matrix, mean_t, t):
    sd_t = 0.3 * mean_t
    future_t = np.arange(t, 109)
    t_dash = t

    factor = np.divide(1, sqrt(2) * pi * sd_t, out=np.zeros_like(mean_t, dtype='float64'), where=mean_t != 0)
    prep_exponent_divident = np.expand_dims(np.expand_dims(future_t - t_dash, axis=1), axis=1)
    exponent_divident = - (prep_exponent_divident - mean_t) ** 2
    exponent_divisor = np.divide(1, 2 * sd_t ** 2, out=np.zeros_like(mean_t, dtype='float64'), where=mean_t != 0)
    exponent = np.einsum('trg,rg->trg', exponent_divident, exponent_divisor)

    new_lambdas = np.einsum('rg,trg->trg', factor, e ** exponent)
    lambda_matrix[t:, t] = new_lambdas


def _create_change_driven_past_dsm(inflows, stocks, outflows, lifetime_mean, years):
    steel_stock_dsm = MultiDim_DynamicStockModel(t=years,
                                                 i=inflows,
                                                 s=stocks,
                                                 o=outflows,
                                                 lt={'Type': 'Normal', 'Mean': lifetime_mean,
                                                     'StdDev': lifetime_mean * 0.3})
    steel_stock_dsm.compute_s_c_inflow_driven()
    steel_stock_dsm.compute_o_c_from_s_c()
    return steel_stock_dsm


if __name__ == '__main__':
    from src.modelling_approaches.compute_upper_cycle import test

    test(model_type='change')
