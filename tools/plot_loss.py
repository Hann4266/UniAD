#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import subprocess
import sys

# Get the log file from the pod
pod_name = "tingji-dual-mount-deployment-69cc799cdf-m7fdz"
log_file = "/mnt/storage/UniAD/projects/work_dirs/loki/base_loki_perception/train_20260215_224353.log"

print("Fetching log file from pod...")
result = subprocess.run(
    ["kubectl", "exec", pod_name, "--", "cat", log_file],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error fetching log: {result.stderr}")
    sys.exit(1)

log_content = result.stdout

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
plt.savefig('/Users/macbook/Desktop/AVL/training_loss.png', dpi=150)
print(f"\nPlot saved to /Users/macbook/Desktop/AVL/training_loss.png")

# Print statistics
print(f"\nLoss Statistics:")
print(f"  Min loss: {min(losses):.4f}")
print(f"  Max loss: {max(losses):.4f}")
print(f"  Current loss: {losses[-1]:.4f}")
print(f"  Average loss (last 50): {sum(losses[-50:])/50:.4f}")
