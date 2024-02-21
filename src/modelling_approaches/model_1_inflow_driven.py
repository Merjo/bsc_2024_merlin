from math import sqrt, pi, e
import numpy as np
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data
from src.odym_extension.MultiDim_DynamicStockModel import MultiDim_DynamicStockModel
from src.read_data.load_data import load_lifetimes, load_region_names_list
from src.modelling_approaches.load_data_for_approaches import get_past_production_trade_forming_fabrication
from src.base_model.load_params import get_cullen_fabrication_yield
from src.modelling_approaches.load_region_sector_splits import get_region_sector_splits
from src.tools.config import cfg


def get_inflow_driven_model_upper_cycle(country_specific=False):
    production, trade, forming_fabrication, indirect_trade = \
        get_past_production_trade_forming_fabrication(country_specific)

    past_dsms = load_model_dsms(country_specific=country_specific,
                                do_past_not_future=True,
                                model_type='inflow',
                                do_econ_model=False,
                                recalculate=True,
                                forming_fabrication=forming_fabrication,
                                indirect_trade=indirect_trade)
    inflows, stocks, outflows = get_dsm_data(past_dsms)
    fabrication_use = inflows - indirect_trade

    return production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows


def get_inflow_driven_past_dsms(country_specific, fabrication=None, indirect_trade=None):
    if fabrication is None or indirect_trade is None:
        production, trade, forming_fabrication, indirect_trade = \
            get_past_production_trade_forming_fabrication(country_specific)
        fabrication = forming_fabrication
    fabrication_yield = np.array(get_cullen_fabrication_yield())
    sector_splits = get_region_sector_splits()[:109]
    mean, std_dev = load_lifetimes()
    inflows = _calc_inflows_via_sector_splits(fabrication, indirect_trade, fabrication_yield, sector_splits,
                                              mean, std_dev)

    inflows = np.moveaxis(inflows, 0, 2)  # move time axis to the end to iterate more easily through inflows
    years = np.arange(1900, 2009)
    dsms = [[_create_inflow_driven_past_dsm(cat_inflows, years, mean[region_idx, cat_idx], std_dev[region_idx, cat_idx])
             for cat_idx, cat_inflows in enumerate(region_inflows)]
            for region_idx, region_inflows in enumerate(inflows)]

    return dsms


def _calc_inflows_via_sector_splits(fabrication, indirect_trade, fabrication_yield, sector_splits, mean, std_dev):
    lifetime_matrix = _calc_lifetime_matrix(mean, std_dev)
    l = np.einsum('trd,turg->turgd', sector_splits, 1 - lifetime_matrix)  # here 'u' denotes t_dash / t'
    d_0_dividend = l[0, 0]
    d_0 = np.einsum('rgd,rdg->rgd', d_0_dividend, 1 / d_0_dividend)
    initial_indirect_trade = np.repeat(np.expand_dims(indirect_trade[0], axis=2), cfg.n_use_categories, axis=2)
    b_0_dividend = initial_indirect_trade * d_0 - np.swapaxes(initial_indirect_trade, 1, 2)
    b_0_divisor = np.einsum('r,g->rg', fabrication[0], fabrication_yield)
    b_0 = np.einsum('rgd,rd->rgd', b_0_dividend, 1 / b_0_divisor)
    b_0 = np.sum(b_0, axis=2)
    m_0_dividend = np.einsum('g,rgd->rgd', fabrication_yield, d_0)
    m_0 = np.einsum('rgd,d->rgd', m_0_dividend, 1 / fabrication_yield)
    m_0 = np.sum(m_0, axis=2)
    x_0 = (1 - b_0) / m_0

    i_0 = np.einsum('r,rg->rg', fabrication[0], (x_0 * fabrication_yield)) + indirect_trade[0]

    inflows = np.zeros_like(indirect_trade)
    inflows[0] = i_0
    fabrication_category_share = np.zeros_like(indirect_trade)
    fabrication_category_share[0] = x_0

    for t in range(1, 109):
        f = fabrication[t]
        s_prepare = np.einsum('trg,trg->trg', inflows[:t], lifetime_matrix[t, :t])
        s_prepare = np.sum(s_prepare, axis=0)
        c = sector_splits[t]
        it = indirect_trade[t]
        lt = lifetime_matrix[t, t]
        y = fabrication_yield

        m_1 = f * y[0] * (1 - lt[:, 0])
        m_2 = np.einsum('r,rg->rg', m_1, c)
        m_3 = np.einsum('rg,r->rg', m_2, 1 / c[:, 0])
        m_4 = np.einsum('rg,r,g->rg', (1 - lt), f, y)
        m = m_3 / m_4

        b_1 = (it[:, 0] * (1 - lt[:, 0]) - s_prepare[:, 0]) / c[:, 0]
        b_2 = np.einsum('r,rg->rg', b_1, c)
        b_3 = b_2 + s_prepare
        b_4 = m_4
        b_5 = b_3 / b_4
        b_6 = np.einsum('rg,r,g->rg', it, 1 / f, 1 / y)
        b = b_5 - b_6

        x_1 = (1 - np.sum(b, axis=1)) / np.sum(m, axis=1)
        x_t = np.einsum('r,rg->rg', x_1, m) + b

        min_x = np.einsum('rg,r,g->rg', -it, 1 / f, 1 / y)
        min_x = np.maximum(0, min_x)  # x should also never be zero
        if np.any(x_t < min_x):
            for region_idx in range(x_t.shape[0]):
                xtr = x_t[region_idx]
                minxr = min_x[region_idx]
                if np.any(xtr < minxr):
                    diff = xtr - minxr
                    neg_pcts = np.minimum(0, diff)
                    pos_pcts = np.maximum(0, diff)
                    sum_factor = np.abs(np.sum(pos_pcts) / np.sum(neg_pcts))
                    xtr = xtr - diff / sum_factor
                    xtr[xtr < minxr] = minxr[xtr < minxr]
                    x_t[region_idx] = xtr

        fabrication_category_share[t] = x_t
        inflows[t] = np.einsum('r,rg,g->rg', fabrication[t], x_t, fabrication_yield) + indirect_trade[t]
    # fabrication_by_category = np.einsum('tr,trg->trg', fabrication, fabrication_category_share)

    inflows[np.logical_and(inflows < 0,
                           inflows > -0.1)] = 0  # make inflows which are -0 or otherwise slightly negative positive 0
    return inflows


def _calc_lifetime_matrix(mean, std_dev):
    t = np.arange(0, 109)
    t_dash = np.arange(0, 109)
    t_matrix = np.subtract.outer(t, t_dash)
    regions = load_region_names_list()
    n_regions = len(regions)
    new_t_matrix_shape = t_matrix.shape + (n_regions, cfg.n_use_categories)
    t_matrix = np.broadcast_to(np.expand_dims(t_matrix, axis=(2, 3)), new_t_matrix_shape)
    exponent = -(t_matrix - mean) ** 2 / (2 * std_dev ** 2)
    lifetime_matrix = 1 / (sqrt(2) * pi * std_dev) * e ** exponent

    # only use lower triangle of lifetime matrix as 'past' lifetimes of inflows are irrelevant
    tri = np.tri(*lifetime_matrix.shape[:2])
    lifetime_matrix = np.einsum('turg,tu->turg', lifetime_matrix, tri)

    return lifetime_matrix


def _create_inflow_driven_past_dsm(inflows, years, lifetime_mean, lifetime_sd):
    steel_stock_dsm = MultiDim_DynamicStockModel(t=years,
                                                 i=inflows,
                                                 lt={'Type': 'Normal', 'Mean': [lifetime_mean],
                                                     'StdDev': [lifetime_sd]})

    steel_stock_dsm.compute_all_inflow_driven()
    return steel_stock_dsm


if __name__ == '__main__':
    from src.modelling_approaches.compute_upper_cycle import test

    test(model_type='inflow')
