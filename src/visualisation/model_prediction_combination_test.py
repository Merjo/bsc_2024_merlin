import numpy as np
from src.read_data.load_data import load_production
from src.tools.tools import get_np_from_df
from src.tools.config import cfg
from src.base_model.simson_base_model import create_base_model, BOF_PID, EAF_PID, FORM_PID
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape
import time

# until 2023

df_production = load_production(country_specific=False).transpose()
production = get_np_from_df(df_production, data_split_into_categories=False)
actual_production = production[109:123]
actual_production_world = np.sum(actual_production, axis=1)  # make sum across regions

model_types = ['change', 'stock', 'inflow']
curve_strategies = ['LSTM', 'Duerrwaechter', 'Pauliuk']
years = np.arange(2009, 2023)
plt.plot(years, actual_production_world, color='orange', label='WorldSteel production data')
model_type_colors = ['r', 'g', 'b']
curve_strategy_shapes = ['--', '.', '-']

mape_dict = {}

labels = []
times = []

do_global_not_regional = False

for c, curve_strategy in enumerate(curve_strategies):
    for m, model_type in enumerate(model_types):
        print(f"\nBegin {curve_strategy}, {model_type.capitalize()}-based\n")
        cfg.curve_strategy = curve_strategy
        cfg.model_type = model_type
        start = time.time()
        model = create_base_model(country_specific=False, recalculate_dsms=False)
        end = time.time()
        bof_production = model.get_flowV(BOF_PID, FORM_PID)[:, 0]
        eaf_production = model.get_flowV(EAF_PID, FORM_PID)[:, 0]
        production = bof_production + eaf_production
        production = production[109:123, :, 1]
        production_world = np.sum(production, axis=1)
        mape_value = mape(actual_production_world, production_world) * 100 \
            if do_global_not_regional \
            else mape(actual_production, production) * 100
        mape_dict[f'{curve_strategy}, {model_type}'] = mape_value
        label = f"{model_type.capitalize()}, {curve_strategy}"
        labels.append(f'{label}, MAPE: {mape_value:.2f}')
        times.append(f'{label}, Time: {(end - start):.2f}')
        plt.plot(years, production_world, curve_strategy_shapes[c],
                 label=f'{label}: {mape_value:.2f}%',
                 color=model_type_colors[m])

print('\nMAPE dict:\n')

for k, v in mape_dict.items():
    print(f'{k}: {v}')

print('\nTimes\n')

for time in times:
    print(time)

plt.xlabel('Time (y)')
plt.ylabel('Steel (t)')
plt.legend()
plt.title('Global steel production, SSP2')

plt.show()
