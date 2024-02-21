import numpy as np
import os
from src.tools.config import cfg
from src.base_model.simson_base_model import load_simson_base_model, FABR_PID, USE_PID, ENV_PID, FORM_PID, BOF_PID, \
    EAF_PID
from matplotlib import pyplot as plt
from src.read_data.load_data import load_production, load_region_names_list
from src.tools.tools import get_np_from_df

regions = load_region_names_list()
colors = ['lightgreen', 'orangered', 'dodgerblue', 'brown', 'greenyellow',
          'crimson', 'olive', 'mediumseagreen', 'black', 'mediumblue', 'orange', 'magenta']

cfg.model_type = 'change'
cfg.curve_strategy = 'LSTM'
model = load_simson_base_model(country_specific=False, recalculate=True, recalculate_dsms=True)

df_production = load_production(country_specific=False).transpose()
actual_production = get_np_from_df(df_production, data_split_into_categories=False)
actual_production = actual_production[80:123, :]

inflow = model.get_flowV(FABR_PID, USE_PID)[80:123, 0, :, :, 1]
inflow = np.sum(inflow, axis=2)
indirect_imports = model.get_flowV(ENV_PID, USE_PID)
indirect_exports = model.get_flowV(USE_PID, ENV_PID)
indirect = indirect_imports - indirect_exports
indirect = np.sum(indirect[80:123, 0, :, :, 1], axis=2)
trade_imports = model.get_flowV(ENV_PID, FORM_PID)
trade_exports = model.get_flowV(FORM_PID, ENV_PID)
trade = trade_imports - trade_exports
trade = trade[80:123, 0, :, 1]

stocks = model.get_stockV(USE_PID)
stocks = np.sum(stocks[80:123, 0, :, :, 1], axis=2) / 20

bof_production = model.get_flowV(BOF_PID, FORM_PID)[:, 0]
eaf_production = model.get_flowV(EAF_PID, FORM_PID)[:, 0]
production = bof_production + eaf_production
production = production[80:123, :, 1]

years = np.arange(1980, 2023)

for i, region in enumerate(regions):
    plt.plot(years, inflow[:, i], label='inflow')
    plt.plot(years, production[:, i], label='production', color='red')
    plt.plot(years, actual_production[:, i], '--', label='actual production', color='red')
    plt.plot(years, trade[:, i], label='trade')
    plt.plot(years, indirect[:, i], label='indirect')
    plt.plot(years, stocks[:, i], label='stocks')
    plt.legend()
    plt.xlabel('Time (y)')
    plt.ylabel('Steel (t)')
    plt.title(f'{region} {cfg.model_type} {cfg.curve_strategy}')
    base_path = os.path.join(cfg.data_path, 'models', 'lstm_models')
    pic_path = os.path.join(base_path, f'{region}_{cfg.model_type}_{cfg.curve_strategy}.png')
    plt.savefig(pic_path, dpi=400)
    plt.clf()
