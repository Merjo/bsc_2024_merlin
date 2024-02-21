import numpy as np
from src.read_data.load_data import load_production, load_region_names_list
from src.tools.tools import get_np_from_df
from src.tools.config import cfg
from src.base_model.simson_base_model import create_base_model, BOF_PID, EAF_PID, FORM_PID
from matplotlib import pyplot as plt

# until 2023
colors = ['orange', 'red', 'purple', 'darkblue', 'black']
regions = load_region_names_list()

model_type = 'inflow'  # Options: ['change', 'stock', 'inflow']
curve_strategy = 'Duerrwaechter'  # Options: ['LSTM', 'Duerrwaechter', 'Pauliuk']
print(f"\nBegin {curve_strategy}, {model_type.capitalize()}-based\n")
cfg.curve_strategy = curve_strategy
cfg.model_type = model_type
model = create_base_model(country_specific=False, recalculate_dsms=False)
bof_production = model.get_flowV(BOF_PID, FORM_PID)[:, 0]
eaf_production = model.get_flowV(EAF_PID, FORM_PID)[:, 0]
production = bof_production + eaf_production
production = np.sum(production, axis=1)

years = np.arange(1900, 2101)
for i, scenario in enumerate(cfg.scenarios):
    plt.plot(years, production[:, i], '-', color=colors[i], label=f'{scenario}')
plt.xlabel('Time (y)')
plt.ylabel('Steel (t)')
plt.legend(cfg.scenarios)
plt.title(f'World Steel production, {model_type}-driven, {curve_strategy} prediction')

plt.show()
