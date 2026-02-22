#!/usr/bin/env python3
import re
import sys
import matplotlib.pyplot as plt

log_file = "/mnt/storage/UniAD/projects/work_dirs/loki/base_loki_perception/train_20260221_100938.log"

print(f"Reading log file: {log_file}")
with open(log_file) as f:
    log_content = f.read()

# Parse the log to extract loss values
pattern = r'Epoch \[(\d+)\]\[(\d+)/\d+\].*?loss: ([\d.]+)'
matches = re.findall(pattern, log_content)

epochs = []
iterations = []
losses = []

for match in matches:
    epoch = int(match[0])
    iteration = int(match[1])
    loss = float(match[2])
    epochs.append(epoch)
    iterations.append(iteration)
    losses.append(loss)

print(f"Found {len(losses)} loss values")
print(f"Epochs: {min(epochs)} to {max(epochs)}")
print(f"Latest loss: {losses[-1]:.4f}")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(range(len(losses)), losses, linewidth=1, alpha=0.7)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)

# Add epoch markers
epoch_changes = []
for i in range(1, len(epochs)):
    if epochs[i] != epochs[i-1]:
        epoch_changes.append(i)
        
for idx in epoch_changes:
    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
# Add moving average
window = 50
if len(losses) >= window:
    moving_avg = []
    for i in range(len(losses) - window + 1):
        moving_avg.append(sum(losses[i:i+window]) / window)
    plt.plot(range(window-1, len(losses)), moving_avg, 'r-', linewidth=2, label=f'{window}-step Moving Average')
    plt.legend()

plt.tight_layout()
out_path = '/mnt/storage/UniAD/projects/work_dirs/loki/base_loki_perception/training_loss.png'
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")

# Print statistics
print(f"\nLoss Statistics:")
print(f"  Min loss: {min(losses):.4f}")
print(f"  Max loss: {max(losses):.4f}")
print(f"  Current loss: {losses[-1]:.4f}")
print(f"  Average loss (last 50): {sum(losses[-50:])/50:.4f}")
