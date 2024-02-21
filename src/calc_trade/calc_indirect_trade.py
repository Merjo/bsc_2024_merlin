import numpy as np
from src.calc_trade.calc_trade_tools import get_trade_category_percentages, scale_trade, \
    expand_trade_to_past_and_future, get_imports_and_exports_from_net_trade, get_trade_test_data, visualize_trade
from src.tools.tools import get_np_from_df
from src.read_data.load_data import load_indirect_trade_2001_2019, load_indirect_trade_category_quantities


def get_indirect_trade(country_specific, scaler, inflows, outflows, split_categories_by_real_data=True):
    net_indirect_trade_2001_2019 = _get_net_indirect_trade_2001_2019(country_specific)
    net_indirect_trade = expand_trade_to_past_and_future(net_indirect_trade_2001_2019,
                                                         scaler=scaler,
                                                         first_available_year=2001,
                                                         last_available_year=2019)
    indirect_imports, indirect_exports = get_imports_and_exports_from_net_trade(net_indirect_trade)

    indirect_imports, indirect_exports = _split_indirect_trade_into_use_categories(split_categories_by_real_data,
                                                                                   country_specific,
                                                                                   indirect_imports,
                                                                                   indirect_exports,
                                                                                   inflows, outflows)
    return indirect_imports, indirect_exports


def get_scaled_past_indirect_trade(country_specific, scaler, split_categories_by_real_data=True):
    scaler = scaler[:102]  # only use scaler up to 2001
    net_indirect_trade_2001_2019 = _get_net_indirect_trade_2001_2019(country_specific)
    net_indirect_trade_1900_2000 = scale_trade(trade=net_indirect_trade_2001_2019,
                                               scaler=scaler,
                                               do_past_not_future=True)

    net_indirect_trade = np.concatenate((net_indirect_trade_1900_2000, net_indirect_trade_2001_2019), axis=0)

    indirect_imports, indirect_exports = get_imports_and_exports_from_net_trade(net_indirect_trade)

    indirect_imports, indirect_exports = _split_indirect_trade_into_use_categories(split_categories_by_real_data,
                                                                                   country_specific,
                                                                                   indirect_imports,
                                                                                   indirect_exports,
                                                                                   do_scenarios=False)
    net_indirect_trade = indirect_imports - indirect_exports

    return net_indirect_trade


def _get_net_indirect_trade_2001_2019(country_specific):
    df_indirect_imports, df_indirect_exports = load_indirect_trade_2001_2019(country_specific=country_specific)
    indirect_imports = get_np_from_df(df_indirect_imports, data_split_into_categories=False)
    indirect_exports = get_np_from_df(df_indirect_exports, data_split_into_categories=False)

    net_indirect_trade = indirect_imports - indirect_exports
    net_indirect_trade = net_indirect_trade.transpose()

    return net_indirect_trade


def _split_indirect_trade_into_use_categories(split_categories_by_real_data, country_specific,
                                              indirect_imports, indirect_exports,
                                              inflows=None, outflows=None, do_scenarios=True):
    if not split_categories_by_real_data and (inflows is None or outflows is None):
        raise RuntimeWarning('With no inflows and outflows given, '
                             'indirect trade can not be split by real category data.')
    if split_categories_by_real_data:
        indirect_trade = indirect_imports - indirect_exports
        df_shares = load_indirect_trade_category_quantities(country_specific=country_specific)
        df_shares = df_shares.divide(df_shares.sum(axis=1), axis=0)
        df_shares['Construction'] = 0  # to not allow negative zeros, not necessary but more elegant
        shares = get_np_from_df(df_shares, data_split_into_categories=False)
        scenario_dim = ''
        if do_scenarios:
            scenario_dim = 's'
        indirect_trade = np.einsum(f'tr{scenario_dim},rg->trg{scenario_dim}', indirect_trade, shares)
        indirect_imports, indirect_exports = get_imports_and_exports_from_net_trade(indirect_trade)
    else:  # split exports by in-use shares in exporting countries and import by the resulting global exports shares
        inflow_category_share = get_trade_category_percentages(inflows, category_axis=2)
        indirect_imports = np.einsum('trs,trgs->trgs', indirect_imports, inflow_category_share)

        outflow_category_share = get_trade_category_percentages(outflows, category_axis=2)
        indirect_exports = np.einsum('trs,trgs->trgs', indirect_exports, outflow_category_share)
    return indirect_imports, indirect_exports


def _test():
    from src.base_model.load_dsms import load_dsms
    from src.base_model.model_tools import get_dsm_data

    country_specific = False
    production, demand, available_scrap_by_category = get_trade_test_data(country_specific)
    dsms = load_dsms(country_specific, recalculate=False)
    stocks, inflows, outflows = get_dsm_data(dsms)
    indirect_imports, indirect_exports = get_indirect_trade(country_specific, scaler=demand, inflows=inflows,
                                                            outflows=outflows)
    indirect_trade = indirect_imports - indirect_exports

    print(f'Indirect trade is loaded with shape: {indirect_trade.shape}')
    visualize_trade(indirect_trade, steel_type='indirect')


if __name__ == '__main__':
    _test()
