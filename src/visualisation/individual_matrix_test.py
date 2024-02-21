import numpy as np
from src.read_data.load_data import load_production, load_region_names_list
from src.tools.tools import get_np_from_df
from src.tools.config import cfg
from src.base_model.simson_base_model import create_base_model, BOF_PID, EAF_PID, FORM_PID
from matplotlib import pyplot as plt

# until 2023
colors = ['lightgreen', 'orangered', 'dodgerblue', 'brown', 'greenyellow',
          'crimson', 'olive', 'mediumseagreen', 'black', 'mediumblue', 'orange', 'magenta']
regions = load_region_names_list()

df_production = load_production(country_specific=False).transpose()
actual_production = get_np_from_df(df_production, data_split_into_categories=False)
actual_production = actual_production[:, :]

regions = load_region_names_list()

model_type = 'change'  # Options: ['change', 'stock', 'inflow']
curve_strategy = 'LSTM'  # Options: ['LSTM', 'Duerrwaechter', 'Pauliuk']
years = np.arange(1900, 2101)
for i, region in enumerate(regions):
    a = 0
    # plt.plot(years, actual_production[:, i], '--', label=f'WS_{region}', color=colors[i])

print(f"\nBegin {curve_strategy}, {model_type.capitalize()}-based\n")
cfg.curve_strategy = curve_strategy
cfg.model_type = model_type
model = create_base_model(country_specific=False, recalculate_dsms=True)
bof_production = model.get_flowV(BOF_PID, FORM_PID)[:, 0]
eaf_production = model.get_flowV(EAF_PID, FORM_PID)[:, 0]
production = bof_production + eaf_production
production = production[:, :, 1]

for i, region in enumerate(regions):
    plt.plot(years, production[:, i], '-', label=f'Predicted_{region}', color=colors[i])
plt.xlabel('Time (y)')
plt.ylabel('Steel (t)')
plt.legend(regions)
plt.title(f'World Steel production, SSP2,\n{curve_strategy} prediction, {model_type}-based')

plt.show()
