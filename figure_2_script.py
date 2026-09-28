import matplotlib.pyplot as plt
import numpy as np

models = ['DLinear\n(Lightweight)', 'SOFTS\n(Lightweight)', 'LSTM\n(Medium)', 'Informer\n(Heavy)']
total_energy = [0.4520, 0.9090, 0.9681, 1.1960]  # Wh

gpu_pct = [0.695, 0.700, 0.711, 0.714]
ram_pct = [0.240, 0.236, 0.225, 0.223]
cpu_pct = [0.065, 0.064, 0.064, 0.063]

gpu_energy = [t * g for t, g in zip(total_energy, gpu_pct)]
ram_energy = [t * r for t, r in zip(total_energy, ram_pct)]
cpu_energy = [t * c for t, c in zip(total_energy, cpu_pct)]

x = np.arange(len(models))
width = 0.55

plt.figure(figsize=(10, 6), dpi=300)

color_gpu = '#1f4e78'  # Dark blue
color_ram = '#4f81bd'  # Medium blue
color_cpu = '#c65911'  # Brown/Orange

p1 = plt.bar(x, gpu_energy, width, label='GPU (Tesla T4)', color=color_gpu)
p2 = plt.bar(x, ram_energy, width, bottom=gpu_energy, label='RAM', color=color_ram)
p3 = plt.bar(x, cpu_energy, width, bottom=np.array(gpu_energy) + np.array(ram_energy), label='CPU (Xeon 2.0GHz)',
             color=color_cpu)

for i in range(len(models)):
    total = total_energy[i]
    plt.text(x[i], total + 0.035, f"{total:.4f} Wh", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.text(x[i], gpu_energy[i] / 2, f"{gpu_pct[i] * 100:.1f}%", ha='center', va='center', color='white',
             fontweight='bold', fontsize=9)
    plt.text(x[i], gpu_energy[i] + ram_energy[i] / 2, f"{ram_pct[i] * 100:.1f}%", ha='center', va='center',
             color='white', fontweight='bold', fontsize=9)
    plt.text(x[i], gpu_energy[i] + ram_energy[i] + cpu_energy[i] / 2, f"{cpu_pct[i] * 100:.1f}%", ha='center',
             va='center', color='white', fontweight='bold', fontsize=9)

plt.title('Figure 5.2: Energy Consumption Disaggregation by Hardware Component\n(All Four Models — 5 Epochs)',
          fontsize=13, fontweight='bold')
plt.ylabel('Energy Consumed (Wh)', fontsize=11)
plt.xticks(x, models, fontsize=10)
plt.ylim(0, 1.45)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

plt.savefig('figure_5_2_four_models_energy.png')
plt.show()