import numpy as np
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data, get_stock_based_dsms
from src.modelling_approaches.load_data_for_approaches import get_past_stocks
from src.calc_trade.calc_trade import get_scaled_past_trade
from src.calc_trade.calc_indirect_trade import get_scaled_past_indirect_trade
from src.base_model.load_params import get_cullen_fabrication_yield
from src.tools.config import cfg


def get_stock_driven_model_upper_cycle(country_specific=False):
    past_dsms = load_model_dsms(country_specific=country_specific,
                                do_past_not_future=True,
                                model_type='stock',
                                do_econ_model=False)
    inflows, stocks, outflows = get_dsm_data(past_dsms)
    scaler = np.sum(inflows, axis=2)
    trade = get_scaled_past_trade(country_specific=country_specific, scaler=scaler)[:109]
    indirect_trade = get_scaled_past_indirect_trade(country_specific=country_specific, scaler=scaler)[:109]
    fabrication_use = inflows - indirect_trade
    fabrication_yield = np.array(get_cullen_fabrication_yield())
    fabrication_by_category = np.einsum('trg,g->trg', fabrication_use, 1 / fabrication_yield)
    forming_fabrication = np.sum(fabrication_by_category, axis=2)
    production_plus_trade = forming_fabrication * (1 / cfg.forming_yield)
    production = production_plus_trade - trade

    return production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows


def get_stock_driven_past_dsms(country_specific):
    stocks = get_past_stocks(country_specific=country_specific)
    dsms = get_stock_based_dsms(stocks, 1900, 2008, do_scenarios=False)
    return dsms


if __name__ == '__main__':
    from src.modelling_approaches.compute_upper_cycle import test

    test(model_type='stock')
