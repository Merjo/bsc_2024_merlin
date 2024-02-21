import numpy as np
from src.tools.config import cfg
from src.modelling_approaches.model_3_change_driven import get_change_driven_model_upper_cycle
from src.modelling_approaches.model_2_stock_driven import get_stock_driven_model_upper_cycle
from src.modelling_approaches.model_1_inflow_driven import get_inflow_driven_model_upper_cycle
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data
from src.calc_trade.calc_trade_tools import scale_trade
from src.base_model.load_params import get_cullen_fabrication_yield


def compute_upper_cycle(model_type=cfg.model_type, country_specific=False):  # decide country_specific
    if model_type == 'change':
        get_upper_cycle_function = get_change_driven_model_upper_cycle
    elif model_type == 'stock':
        get_upper_cycle_function = get_stock_driven_model_upper_cycle
    elif model_type == 'inflow':
        get_upper_cycle_function = get_inflow_driven_model_upper_cycle
    past_production, past_trade, past_forming_fabrication, past_fabrication_use, past_indirect_trade, past_inflows, \
    past_stocks, past_outflows = \
        get_upper_cycle_function(country_specific=country_specific)

    past_production = _add_scenario_dimension(past_production)
    past_trade = _add_scenario_dimension(past_trade)
    past_forming_fabrication = _add_scenario_dimension(past_forming_fabrication)
    past_fabrication_use = _add_scenario_dimension(past_fabrication_use)
    past_indirect_trade = _add_scenario_dimension(past_indirect_trade)
    past_inflows = _add_scenario_dimension(past_inflows)
    past_stocks = _add_scenario_dimension(past_stocks)
    past_outflows = _add_scenario_dimension(past_outflows)

    future_dsms = load_model_dsms(country_specific=country_specific,
                                  do_past_not_future=False,
                                  model_type=model_type,
                                  do_econ_model=False)

    inflows, stocks, outflows = get_dsm_data(future_dsms)

    inflows[:109] = past_inflows
    stocks[:109] = past_stocks
    outflows[:109] = past_outflows
    future_indirect_trade = scale_trade(past_indirect_trade, scaler=inflows[109 - 1:],
                                        do_past_not_future=False)
    indirect_trade = np.concatenate((past_indirect_trade, future_indirect_trade), axis=0)
    future_trade = scale_trade(past_trade, scaler=np.sum(inflows[109 - 1:], axis=2),
                               do_past_not_future=False)
    trade = np.concatenate((past_trade, future_trade), axis=0)

    fabrication_yield = get_cullen_fabrication_yield()
    fabrication_use = inflows - indirect_trade

    fabrication_by_category = np.einsum('trgs,g->trgs', fabrication_use, 1 / np.array(fabrication_yield))
    forming_fabrication = np.sum(fabrication_by_category, axis=2)
    production_plus_trade = forming_fabrication * (1 / cfg.forming_yield)
    production = production_plus_trade - trade

    return production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows


def _add_scenario_dimension(data):
    new_shape = data.shape + (cfg.n_scenarios,)
    data = np.expand_dims(data, axis=-1)
    data = np.broadcast_to(data, new_shape)
    return data


def test(model_type=cfg.model_type):
    production, trade, forming_fabrication, fabrication_use, indirect_trade, inflow, stocks, outflow = \
        compute_upper_cycle(model_type)

    print(f'{model_type.capitalize()}-based model upper cycle loading succesful. \n'
          f'\nCheck shapes\n')
    print(f'Production shape: {production.shape}')
    print(f'Trade shape: {trade.shape}')
    print(f'Forming-Fabrication shape: {forming_fabrication.shape}')
    print(f'Fabrication-Use shape: {fabrication_use.shape}')
    print(f'Indirect trade shape: {indirect_trade.shape}')
    print(f'Inflow shape: {inflow.shape}')
    print(f'Stocks shape: {stocks.shape}')
    print(f'Outflows shape: {outflow.shape}')


if __name__ == '__main__':
    test('change')
