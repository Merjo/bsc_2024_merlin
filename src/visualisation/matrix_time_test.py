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

time_dict = {}

epochs = 100

for c, curve_strategy in enumerate(curve_strategies):
    for m, model_type in enumerate(model_types):
        cfg.curve_strategy = curve_strategy
        cfg.model_type = model_type
        total_time = 0
        for e in range(epochs):
            print(f"\n\nBegin Epoch {e + 1}/{epochs} {curve_strategy}, {model_type.capitalize()}-based\n\n")
            start = time.time()
            model = create_base_model(country_specific=False, recalculate_dsms=False)
            end = time.time()
            test_time = end - start
            total_time += test_time
        time_dict[(curve_strategy, model_type)] = total_time / epochs

print('\n\n\n')

for c, curve_strategy in enumerate(curve_strategies):
    strategy_value = 0
    for m, model_type in enumerate(model_types):
        strategy_value += time_dict[(curve_strategy, model_type)]
    strategy_value /= 3
    print(f'{curve_strategy} = {strategy_value:.2f}')

print('\n\n')

for m, model_type in enumerate(model_types):
    model_value = 0
    for c, curve_strategy in enumerate(curve_strategies):
        model_value += time_dict[(curve_strategy, model_type)]
    model_value /= 3
    print(f'{model_type} = {model_value:.2f}')

print('\n\n')

for k, v in time_dict.items():
    print(f'{k}: {v:.2f}')
