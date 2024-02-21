import numpy as np
import pandas as pd
import os
import sys
import pickle
from ODYM.odym.modules.ODYM_Classes import MFAsystem, Classification, Process, Parameter
from src.odym_extension.SimDiGraph_MFAsystem import SimDiGraph_MFAsystem
from src.tools.config import cfg
from src.base_model.model_tools import get_dsm_data, get_stock_data_country_specific_areas, calc_change_timeline
from src.base_model.load_dsms import load_dsms
from src.base_model.load_params import get_cullen_fabrication_yield, get_wittig_distributions
from src.calc_trade.calc_trade import get_trade
from src.calc_trade.calc_scrap_trade import get_scrap_trade
from src.calc_trade.calc_indirect_trade import get_indirect_trade
from src.calc_trade.calc_trade_tools import get_imports_and_exports_from_net_trade
from src.modelling_approaches.compute_upper_cycle import compute_upper_cycle

#  constants: MFA System process IDs

ENV_PID = 0
BOF_PID = 1
EAF_PID = 2
FORM_PID = 3
FABR_PID = 4
USE_PID = 5
SCRAP_PID = 6
RECYCLE_PID = 7
WASTE_PID = 8
DISNOTCOL_PID = 9


def load_simson_base_model(country_specific=False, recalculate=False, recalculate_dsms=False) -> SimDiGraph_MFAsystem:
    file_name_end = 'countries' if country_specific else f'{cfg.region_data_source}_regions'
    file_name = f'main_model_{file_name_end}.p'
    file_path = os.path.join(cfg.data_path, 'models', file_name)
    do_load_existing = os.path.exists(file_path) and not recalculate
    if do_load_existing:
        model = pickle.load(open(file_path, "rb"))
    else:
        model = create_base_model(country_specific, recalculate_dsms)
        pickle.dump(model, open(file_path, "wb"))
    return model


def create_base_model(country_specific, recalculate_dsms):
    dsms = load_dsms(country_specific, recalculate_dsms)
    model, balance_message = create_model(country_specific, dsms)
    print(balance_message)
    return model


def create_model(country_specific, dsms, scrap_share_in_production=None):
    n_regions = len(dsms)
    max_scrap_share_in_production = _calc_max_scrap_share(scrap_share_in_production, n_regions)
    # load data
    areas = get_stock_data_country_specific_areas(country_specific)
    main_model = set_up_model(areas)
    stocks, inflows, outflows = get_dsm_data(dsms)
    # Load base_model
    initiate_model(main_model)

    # compute stocks and flows
    inflows, outflows = compute_flows(main_model, country_specific, inflows, outflows,
                                      max_scrap_share_in_production)
    compute_stocks(main_model, inflows, outflows)

    # check base_model
    balance_message = mass_balance_plausible(main_model)

    return main_model, balance_message


def initiate_model(main_model):
    initiate_processes(main_model)
    initiate_parameters(main_model)
    initiate_flows(main_model)
    initiate_stocks(main_model)
    main_model.Initialize_FlowValues()
    main_model.Initialize_StockValues()
    check_consistency(main_model)


def set_up_model(regions):
    model_classification = {'Time': Classification(Name='Time', Dimension='Time', ID=1,
                                                   Items=cfg.years),
                            'Element': Classification(Name='Elements', Dimension='Element', ID=2, Items=['Fe']),
                            'Region': Classification(Name='Regions', Dimension='Region', ID=3, Items=regions),
                            'Good': Classification(Name='Goods', Dimension='Material', ID=4,
                                                   Items=cfg.in_use_categories),
                            'Waste': Classification(Name='Waste types', Dimension='Material', ID=5,
                                                    Items=cfg.recycling_categories),
                            'Scenario': Classification(Name='Scenario', Dimension='Scenario', ID=6,
                                                       Items=cfg.scenarios)}
    model_time_start = cfg.start_year
    model_time_end = cfg.end_year
    index_table = pd.DataFrame({'Aspect': ['Time', 'Element', 'Region', 'Good', 'Waste', 'Scenario'],
                                'Description': ['Model aspect "Time"', 'Model aspect "Element"',
                                                'Model aspect "Region"', 'Model aspect "Good"',
                                                'Model aspect "Waste"', 'Model aspect "Scenario"'],
                                'Dimension': ['Time', 'Element', 'Region', 'Material', 'Material', 'Scenario'],
                                'Classification': [model_classification[Aspect] for Aspect in
                                                   ['Time', 'Element', 'Region', 'Good', 'Waste', 'Scenario']],
                                'IndexLetter': ['t', 'e', 'r', 'g', 'w', 's']})
    index_table.set_index('Aspect', inplace=True)

    main_model = SimDiGraph_MFAsystem(Name='World Steel Economy',
                                      Geogr_Scope='World',
                                      Unit='t',
                                      ProcessList=[],
                                      FlowDict={},
                                      StockDict={},
                                      ParameterDict={},
                                      Time_Start=model_time_start,
                                      Time_End=model_time_end,
                                      IndexTable=index_table,
                                      Elements=index_table.loc['Element'].Classification.Items)

    return main_model


def initiate_processes(main_model):
    main_model.ProcessList = []

    def add_process(name, p_id):
        main_model.ProcessList.append(Process(Name=name, ID=p_id))

    add_process('Primary Production / Environment', ENV_PID)
    add_process('BF/BOF Production', BOF_PID)
    add_process('EAF Production', EAF_PID)
    add_process('Forming', FORM_PID)
    add_process('Fabrication', FABR_PID)
    add_process('Using', USE_PID)
    add_process('End of Life', SCRAP_PID)
    add_process('Recycling', RECYCLE_PID)
    add_process('Waste', WASTE_PID)
    add_process('Dissipative/Not collectable', DISNOTCOL_PID)


def initiate_parameters(main_model):
    parameter_dict = {}

    use_recycling_params, recycling_usable_params = get_wittig_distributions()
    fabrication_yield = get_cullen_fabrication_yield()

    parameter_dict['Fabrication_Yield'] = Parameter(Name='Fabrication_Yield', ID=0,
                                                    P_Res=FABR_PID, MetaData=None, Indices='g',
                                                    Values=np.array(fabrication_yield), Unit='1')

    parameter_dict['Use-EOL_Distribution'] = Parameter(Name='End-of-Life_Distribution', ID=1, P_Res=USE_PID,
                                                       MetaData=None, Indices='gw',
                                                       Values=np.array(use_recycling_params).transpose(), Unit='1')

    parameter_dict['EOL-Recycle_Distribution'] = Parameter(Name='EOL-Recycle_Distribution', ID=2,
                                                           P_Res=SCRAP_PID,
                                                           MetaData=None, Indices='w',
                                                           Values=np.array(recycling_usable_params), Unit='1')

    main_model.ParameterDict = parameter_dict


def initiate_flows(main_model):
    main_model.init_flow('Iron scaler - BOF scaler', ENV_PID, BOF_PID, 't,e,r,s')
    main_model.init_flow('Recycling - BOF scaler', RECYCLE_PID, BOF_PID, 't,e,r,s')
    main_model.init_flow('BOF scaler - Forming', BOF_PID, FORM_PID, 't,e,r,s')
    main_model.init_flow('Recycling - EAF scaler', RECYCLE_PID, EAF_PID, 't,e,r,s')
    main_model.init_flow('EAF scaler - Forming', EAF_PID, FORM_PID, 't,e,r,s')

    main_model.init_flow('Forming - Fabrication', FORM_PID, FABR_PID, 't,e,r,s')
    main_model.init_flow('Forming - Scrap', FORM_PID, SCRAP_PID, 't,e,r,w,s')
    main_model.init_flow('Fabrication - In-Use', FABR_PID, USE_PID, 't,e,r,g,s')
    main_model.init_flow('Fabrication - Scrap', FABR_PID, SCRAP_PID, 't,e,r,w,s')

    main_model.init_flow('In-Use - Reuse', USE_PID, USE_PID, 't,e,r,g,s')
    main_model.init_flow('In-Use - Scrap', USE_PID, SCRAP_PID, 't,e,r,g,w,s')
    main_model.init_flow('In-Use - Dis./Not col.', USE_PID, DISNOTCOL_PID, 't,e,r,g,w,s')
    main_model.init_flow('Scrap - Recycling', SCRAP_PID, RECYCLE_PID, 't,e,r,s')
    main_model.init_flow('Scrap - Waste', SCRAP_PID, WASTE_PID, 't,e,r,s')

    main_model.init_flow('Crude Imports', ENV_PID, FORM_PID, 't,e,r,s')
    main_model.init_flow('Crude Exports', FORM_PID, ENV_PID, 't,e,r,s')
    main_model.init_flow('Scrap Imports', ENV_PID, SCRAP_PID, 't,e,r,w,s')
    main_model.init_flow('Scrap Exports', SCRAP_PID, ENV_PID, 't,e,r,w,s')
    main_model.init_flow('Indirect Imports', ENV_PID, USE_PID, 't,e,r,g,s')
    main_model.init_flow('Indirect Exports', USE_PID, ENV_PID, 't,e,r,g,s')


def initiate_stocks(main_model):
    main_model.add_stock(USE_PID, 'in_use', 't,e,r,g,s')
    main_model.add_stock(WASTE_PID, 'waste', 't,e,r,s')
    main_model.add_stock(DISNOTCOL_PID, 'dissipative/not_collectable', 't,e,r,w,s')


def check_consistency(main_model: MFAsystem):
    """
    Uses ODYM consistency checks to see if base_model dimensions and structure are well
    defined. Raises RuntimeError if not.

    :param main_model: The MFA System
    :return:
    """
    consistency = main_model.Consistency_Check()
    for consistencyCheck in consistency:
        if not consistencyCheck:
            raise RuntimeError("A consistency check failed: " + str(consistency))


def compute_flows(model: MFAsystem, country_specific: bool,
                  inflows: np.ndarray, outflows: np.ndarray, max_scrap_share_in_production: np.ndarray):
    """

    :param model: The MFA system
    :param country_specific:
    :param inflows:
    :param outflows:
    :param max_scrap_share_in_production:
    :return:
    """

    # Compute upper cycle
    # production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows

    use_eol_distribution, eol_recycle_distribution, fabrication_yield = _get_params(model)

    reuse = None
    if cfg.do_change_reuse and not cfg.do_model_approaches:
        # one is substracted as one was added to multiply scenario and category reuse changes
        reuse_factor_timeline = calc_change_timeline(cfg.reuse_factor, cfg.reuse_change_base_year) - 1
        reuse = np.einsum('trgs,tgs->trgs', outflows, reuse_factor_timeline)
        inflows -= reuse
        outflows -= reuse

    production, forming_fabrication, imports, exports, fabrication_use, indirect_imports, indirect_exports, \
    inflows, outflows = compute_upper_cycle_base_model(country_specific, inflows, outflows, fabrication_yield) \
        if not cfg.do_model_approaches \
        else compute_upper_cycle_modelling_approaches()

    direct_demand = np.sum(fabrication_use, axis=2)
    fabrication_scrap = forming_fabrication - direct_demand
    production_plus_trade = production + imports - exports
    forming_scrap = production_plus_trade - forming_fabrication

    outflows_by_waste = np.einsum('trgs,gw->trgws', outflows, use_eol_distribution)
    use_eol = np.zeros_like(outflows_by_waste)
    use_env = np.zeros_like(outflows_by_waste)

    dis_idx = cfg.recycling_categories.index('Dis')
    use_eol[:, :, :, :dis_idx, :] = outflows_by_waste[:, :, :, :dis_idx, :]
    use_env[:, :, :, dis_idx:, :] = outflows_by_waste[:, :, :, dis_idx:, :]
    eol_scrap = np.sum(use_eol, axis=2)

    available_scrap = eol_scrap.copy()
    available_scrap[:, :, cfg.recycling_categories.index('Form'), :] = forming_scrap
    available_scrap[:, :, cfg.recycling_categories.index('Fabr'), :] = fabrication_scrap

    scrap_imports, scrap_exports = get_scrap_trade(country_specific=country_specific, scaler=production,
                                                   available_scrap_by_category=available_scrap)

    total_scrap = available_scrap + scrap_imports - scrap_exports

    max_scrap_in_production = production * max_scrap_share_in_production
    recyclable_scrap = np.einsum('trwe,w->trwe', total_scrap, eol_recycle_distribution)
    recyclable_scrap = np.sum(recyclable_scrap, axis=2)
    scrap_in_production = np.minimum(recyclable_scrap, max_scrap_in_production)

    scrap_share = np.divide(scrap_in_production, production,
                            out=np.zeros_like(scrap_in_production), where=production != 0)
    eaf_share_production = _calc_eaf_share_production(scrap_share)
    eaf_production = production * eaf_share_production
    bof_production = production - eaf_production
    max_scrap_in_bof = cfg.scrap_in_BOF_rate * bof_production
    scrap_in_bof = np.minimum(max_scrap_in_bof, scrap_in_production)
    iron_production = bof_production - scrap_in_bof

    scrap_in_production = scrap_in_bof + eaf_production
    waste = np.sum(total_scrap, axis=2) - scrap_in_production

    edit_flows(model, iron_production, scrap_in_bof, bof_production, eaf_production, forming_fabrication, forming_scrap,
               imports, exports, fabrication_use, reuse, fabrication_scrap, use_eol, use_env, scrap_imports,
               scrap_exports, scrap_in_production, waste, indirect_imports, indirect_exports)

    return inflows, outflows


def compute_upper_cycle_modelling_approaches():
    production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows = \
        compute_upper_cycle(model_type=cfg.model_type)
    imports, exports = get_imports_and_exports_from_net_trade(trade)
    indirect_imports, indirect_exports = get_imports_and_exports_from_net_trade(indirect_trade)

    return production, forming_fabrication, imports, exports, fabrication_use, indirect_imports, indirect_exports, \
           inflows, outflows


def compute_upper_cycle_base_model(country_specific, inflows, outflows, fabrication_yield):
    total_demand = np.sum(inflows, axis=2)

    indirect_imports, indirect_exports = get_indirect_trade(country_specific=country_specific,
                                                            scaler=total_demand,
                                                            inflows=inflows,
                                                            outflows=outflows)
    fabrication_use = inflows - indirect_imports + indirect_exports

    inverse_fabrication_yield = 1 / fabrication_yield
    fabrication_by_category = np.einsum('trgs,g->trgs', fabrication_use, inverse_fabrication_yield)
    forming_fabrication = np.sum(fabrication_by_category, axis=2)

    imports, exports = get_trade(country_specific=country_specific, scaler=total_demand)

    production_plus_trade = forming_fabrication * (1 / cfg.forming_yield)
    production = production_plus_trade + exports - imports

    return production, forming_fabrication, imports, exports, fabrication_use, indirect_imports, indirect_exports, \
           inflows, outflows


def edit_flows(model, iron_production, scrap_in_bof, bof_production, eaf_production, forming_fabrication, forming_scrap,
               imports, exports, fabrication_use, reuse, fabrication_scrap, use_eol, use_env, scrap_imports,
               scrap_exports, scrap_in_production, waste, indirect_imports, indirect_exports):
    model.get_flowV(ENV_PID, BOF_PID)[:, 0] = iron_production
    model.get_flowV(RECYCLE_PID, BOF_PID)[:, 0] = scrap_in_bof
    model.get_flowV(BOF_PID, FORM_PID)[:, 0] = bof_production
    model.get_flowV(RECYCLE_PID, EAF_PID)[:, 0] = eaf_production
    model.get_flowV(EAF_PID, FORM_PID)[:, 0] = eaf_production
    model.get_flowV(FORM_PID, FABR_PID)[:, 0] = forming_fabrication
    model.get_flowV(FORM_PID, SCRAP_PID)[:, 0, :, cfg.recycling_categories.index('Form')] = forming_scrap
    model.get_flowV(ENV_PID, FORM_PID)[:, 0] = imports
    model.get_flowV(FORM_PID, ENV_PID)[:, 0] = exports
    model.get_flowV(FABR_PID, USE_PID)[:, 0] = fabrication_use
    model.get_flowV(ENV_PID, USE_PID)[:, 0] = indirect_imports
    model.get_flowV(USE_PID, ENV_PID)[:, 0] = indirect_exports
    model.get_flowV(FABR_PID, SCRAP_PID)[:, 0, :, cfg.recycling_categories.index('Fabr')] = fabrication_scrap
    if reuse is not None:
        model.get_flowV(USE_PID, USE_PID)[:, 0] = reuse
    model.get_flowV(USE_PID, SCRAP_PID)[:, 0] = use_eol
    model.get_flowV(USE_PID, DISNOTCOL_PID)[:, 0] = use_env
    model.get_flowV(ENV_PID, SCRAP_PID)[:, 0] = scrap_imports
    model.get_flowV(SCRAP_PID, ENV_PID)[:, 0] = scrap_exports
    model.get_flowV(SCRAP_PID, RECYCLE_PID)[:, 0] = scrap_in_production
    model.get_flowV(SCRAP_PID, WASTE_PID)[:, 0] = waste


def _get_params(model):
    params = model.ParameterDict
    use_eol_distribution = params['Use-EOL_Distribution'].Values
    eol_recycle_distribution = params['EOL-Recycle_Distribution'].Values
    fabrication_yield = params['Fabrication_Yield'].Values

    return use_eol_distribution, eol_recycle_distribution, fabrication_yield


def _calc_eaf_share_production(scrap_share):
    eaf_share_production = (scrap_share - cfg.scrap_in_BOF_rate) / (1 - cfg.scrap_in_BOF_rate)
    eaf_share_production[eaf_share_production <= 0] = 0
    return eaf_share_production


def _calc_max_scrap_share(scrap_share_in_production, n_regions):
    max_scrap_share_in_production = np.ones(
        [cfg.n_years, n_regions, cfg.n_scenarios]) * cfg.max_scrap_share_production_base_model
    if scrap_share_in_production is not None:
        max_scrap_share_in_production[cfg.econ_start_index:, :] = scrap_share_in_production
    return max_scrap_share_in_production


def compute_stocks(model, inflows, outflows):
    in_use_stock_change = model.get_stock_changeV(USE_PID)
    in_use_stock_change[:, 0, :, :] = inflows - outflows
    model.calculate_stock_values_from_stock_change(
        USE_PID)

    inflow_waste = model.get_flowV(SCRAP_PID, WASTE_PID)
    model.get_stock_changeV(WASTE_PID)[:] = inflow_waste
    model.calculate_stock_values_from_stock_change(WASTE_PID)

    inflow_disnotcol = model.get_flowV(USE_PID, DISNOTCOL_PID)
    inflow_disnotcol = np.sum(inflow_disnotcol, axis=3)
    model.get_stock_changeV(DISNOTCOL_PID)[:] = inflow_disnotcol
    model.calculate_stock_values_from_stock_change(DISNOTCOL_PID)

    return model


def mass_balance_plausible(main_model):
    """
    Checks if a given mass balance is plausible.
    :return: True if the mass balance for all processes is below 1t of steel, False otherwise.
    """

    fabrication_use = main_model.get_flowV(FABR_PID, USE_PID)[:, 0]
    indirect_imports = main_model.get_flowV(ENV_PID, USE_PID)[:, 0]
    indirect_exports = main_model.get_flowV(USE_PID, ENV_PID)[:, 0]
    use_disnotcol = np.sum(main_model.get_flowV(USE_PID, DISNOTCOL_PID)[:, 0], axis=-2)
    use_scrap = np.sum(main_model.get_flowV(USE_PID, SCRAP_PID)[:, 0], axis=-2)
    stock_change = main_model.get_stock_changeV(USE_PID)[:, 0]
    inflows = fabrication_use + indirect_imports - indirect_exports
    outflows = use_disnotcol + use_scrap
    stock_change2 = inflows - outflows

    balance = main_model.MassBalance()

    balance = np.abs(np.sum(balance, axis=(0, 2)))
    error = balance > 1
    if np.any(error):
        error_message = f"Error in mass balance of model\n"
        for idx, error_occured in enumerate(error):
            if idx == 0:  # Environment will always have an error
                continue
            if error_occured:
                error_message += f"\nError in process {idx} '{main_model.ProcessList[idx].Name}': {balance[idx]}"
        error_message += f"\n\nBalance summary: {balance}"
        raise RuntimeError(error_message)
    else:
        return f"Success - Model loaded and checked. \nBalance: {balance}.\n"


def main():
    """
    Recalculates the DMFA dict based on the dynamic stock models and trade_all_areas data.
    Checks the Mass Balance and raises a runtime error if the mass balance is too big.
    Prints success statements otherwise
    :return: None
    """
    load_simson_base_model(country_specific=False, recalculate=True)


if __name__ == "__main__":
    # overwrite config with values given in a config file,
    # if the path to this file is passed as the last argument of the function call.
    if sys.argv[-1].endswith('.yml'):
        cfg.customize(sys.argv[-1])
    main()
