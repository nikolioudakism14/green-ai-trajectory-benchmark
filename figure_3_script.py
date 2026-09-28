import matplotlib.pyplot as plt
import numpy as np

models = ['DLinear', 'SOFTS', 'LSTM', 'Informer']
energy = [0.000452, 0.000909, 0.000968, 0.001196] # kWh
val_mse = [0.070112, 0.069691, 0.068323, 0.070225]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b']
markers = ['s', 'o', '^', 'd']

plt.figure(figsize=(9, 6), dpi=300)

for m, e, v, col, mark in zip(models, energy, val_mse, colors, markers):
    plt.scatter(e, v, color=col, marker=mark, s=120, zorder=3, label=m)

plt.annotate('DLinear', (energy[0], val_mse[0]), textcoords="offset points", xytext=(-25,10), fontweight='bold', fontsize=10)
plt.annotate('SOFTS', (energy[1], val_mse[1]), textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold', fontsize=10)
plt.annotate('LSTM', (energy[2], val_mse[2]), textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold', fontsize=10)
plt.annotate('Informer', (energy[3], val_mse[3]), textcoords="offset points", xytext=(-25,10), fontweight='bold', fontsize=10)

pareto_e = [0.000452, 0.000968]
pareto_v = [0.070112, 0.068323]
plt.plot(pareto_e, pareto_v, linestyle='--', color='gray', linewidth=1.5, label='Pareto Front (DLinear - LSTM)')

plt.title('Figure 5.3: Accuracy-Energy Pareto Front (Four Architectures)', fontsize=13, fontweight='bold')
plt.xlabel('Energy Consumption (kWh)', fontsize=11)
plt.ylabel('Validation MSE', fontsize=11)
plt.ylim(0.0680, 0.0707)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()

plt.savefig('figure_5_3_four_models_pareto.png')
plt.show()