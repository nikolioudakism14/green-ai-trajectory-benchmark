import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
dlinear_loss = [0.080124, 0.073112, 0.071985, 0.071204, 0.070891]
softs_loss = [0.079713, 0.073955, 0.072891, 0.072407, 0.072581]
lstm_loss = [0.081007, 0.072019, 0.071382, 0.071417, 0.070947]
informer_loss = [0.079738, 0.074624, 0.073587, 0.073121, 0.072719]

plt.figure(figsize=(9, 6), dpi=300)
plt.plot(epochs, softs_loss, marker='o', color='#2ca02c', linewidth=2.5, label='SOFTS (Lightweight)')
plt.plot(epochs, dlinear_loss, marker='s', color='#1f77b4', linewidth=2.5, label='DLinear (Lightweight)')
plt.plot(epochs, lstm_loss, marker='^', color='#ff7f0e', linewidth=2.5, label='LSTM (Medium)')
plt.plot(epochs, informer_loss, marker='d', color='#8c564b', linewidth=2.5, label='Informer (Heavy)')

plt.title('Figure 5.1: Training Loss Convergence Across the Four Architectures', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training MSE', fontsize=12)
plt.xticks(epochs, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()

plt.savefig('figure_5_1_four_models.png')
plt.show()