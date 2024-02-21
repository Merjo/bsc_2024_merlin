import numpy as np
from src.tools.config import cfg
from src.base_model.simson_base_model import create_base_model, BOF_PID, EAF_PID, FORM_PID
from matplotlib import pyplot as plt


def get_combo_production(curve_strategy, model_type):
    cfg.curve_strategy = curve_strategy
    cfg.model_type = model_type
    model = create_base_model(country_specific=False, recalculate_dsms=True)
    bof_production = model.get_flowV(BOF_PID, FORM_PID)[:, 0]
    eaf_production = model.get_flowV(EAF_PID, FORM_PID)[:, 0]
    production = bof_production + eaf_production
    production = production[:, :, 1]
    production = np.sum(production, axis=1)
    return production


# until 2023
colors = ['lightgreen', 'orangered', 'dodgerblue', 'brown', 'greenyellow',
          'crimson', 'olive', 'mediumseagreen', 'black', 'mediumblue', 'orange', 'magenta']

years = np.arange(1900, 2101)
production_1 = get_combo_production('LSTM', 'inflow')
production_2 = get_combo_production('Pauliuk', 'stock')

total_lstm = np.sum(production_1[109:])
total_paulik = np.sum(production_2[109:])

print(production_1)
print(production_2)

percentages = np.cumsum(production_1[109:]) / np.cumsum(production_2[109:])
for i, pct in enumerate(percentages):
    print(f'{i + 2009}: {pct}')

plt.plot(years, production_1, '-', label=f'Predicted_LSTM_Inflow', color=colors[0])
plt.plot(years, production_2, '-', label=f'Predicted_Pauliuk_Stock', color=colors[1])
plt.xlabel('Time (y)')
plt.ylabel('Steel (t)')
plt.legend()
plt.title(f'Global steel production comparison')

plt.show()
