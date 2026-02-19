"""
Plot training loss curves from LOKI/UniAD JSON logs.
Run: python tools/plot_training_curves.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

matplotlib.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

WORK_DIR = "projects/work_dirs/loki/base_loki_perception"
STEPS_PER_EPOCH = 18084

# ── Load only the latest JSON log (the current successful run) ──
log_files = sorted(glob.glob(os.path.join(WORK_DIR, "*.log.json")))
log_files = [log_files[-1]]  # use only the most recent log
print(f"Using log file: {log_files[0]}")

records = []
for lf in log_files:
    with open(lf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("mode") == "train":
                    records.append(rec)
            except json.JSONDecodeError:
                continue

print(f"Loaded {len(records)} training records")

# ── Extract data ──
steps = []
total_loss = []
avg_cls_loss = []
avg_bbox_loss = []
lr_list = []
grad_norms = []
epoch_boundaries = []
prev_epoch = None

for r in records:
    epoch = r["epoch"]
    iteration = r["iter"]
    global_step = (epoch - 1) * STEPS_PER_EPOCH + iteration
    steps.append(global_step)
    total_loss.append(r["loss"])
    lr_list.append(r["lr"])
    grad_norms.append(r.get("grad_norm", 0))

    if prev_epoch is not None and epoch != prev_epoch:
        epoch_boundaries.append(global_step)
    prev_epoch = epoch

    # Average cls and bbox across all frames and decoder layers
    cls_vals, bbox_vals = [], []
    for key, val in r.items():
        if "loss_cls" in key and "track" in key:
            cls_vals.append(val)
        elif "loss_bbox" in key and "track" in key:
            bbox_vals.append(val)
    avg_cls_loss.append(np.mean(cls_vals) if cls_vals else 0)
    avg_bbox_loss.append(np.mean(bbox_vals) if bbox_vals else 0)


# ── Smoothing helper ──
def smooth(values, weight=0.95):
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def add_epoch_lines(ax, boundaries):
    for b in boundaries:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, linewidth=1)


# ── Figure 1: 4-panel dashboard ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LOKI + UniAD Perception Training Progress",
             fontsize=16, fontweight='bold', y=0.98)

# Total Loss
ax = axes[0, 0]
ax.plot(steps, total_loss, alpha=0.12, color='steelblue', linewidth=0.5)
ax.plot(steps, smooth(total_loss), color='steelblue', linewidth=2.5, label='Smoothed')
add_epoch_lines(ax, epoch_boundaries)
ax.set_xlabel("Global Step")
ax.set_ylabel("Total Loss")
ax.set_title("Total Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Cls vs BBox
ax = axes[0, 1]
ax.plot(steps, smooth(avg_cls_loss), color='coral', linewidth=2.5, label='Classification')
ax.plot(steps, smooth(avg_bbox_loss), color='teal', linewidth=2.5, label='BBox Regression')
add_epoch_lines(ax, epoch_boundaries)
ax.set_xlabel("Global Step")
ax.set_ylabel("Avg Loss per Head")
ax.set_title("Classification vs BBox Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Learning Rate
ax = axes[1, 0]
ax.plot(steps, lr_list, color='darkorange', linewidth=2)
add_epoch_lines(ax, epoch_boundaries)
ax.set_xlabel("Global Step")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule (Cosine Annealing)")
ax.grid(True, alpha=0.3)

# Gradient Norm
ax = axes[1, 1]
ax.plot(steps, grad_norms, alpha=0.12, color='mediumpurple', linewidth=0.5)
ax.plot(steps, smooth(grad_norms), color='mediumpurple', linewidth=2.5, label='Smoothed')
add_epoch_lines(ax, epoch_boundaries)
ax.set_xlabel("Global Step")
ax.set_ylabel("Gradient Norm")
ax.set_title("Gradient Norm")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(WORK_DIR, "training_curves.png")
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# ── Figure 2: Per-frame BBox loss comparison ──
fig2, ax2 = plt.subplots(figsize=(10, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50']
labels = ['Frame 0 (current)', 'Frame 1 (t-1)', 'Frame 2 (t-2)']
for frame_id in [0, 1, 2]:
    frame_bbox = []
    for r in records:
        vals = [r[k] for k in r if f"frame_{frame_id}_loss_bbox" in k]
        frame_bbox.append(np.mean(vals) if vals else 0)
    ax2.plot(steps, smooth(frame_bbox, 0.97), linewidth=2.5,
             color=colors[frame_id], label=labels[frame_id])
add_epoch_lines(ax2, epoch_boundaries)
ax2.set_xlabel("Global Step")
ax2.set_ylabel("Avg BBox Loss")
ax2.set_title("BBox Loss by Temporal Frame (queue_length=3)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
out2 = os.path.join(WORK_DIR, "training_curves_perframe.png")
plt.savefig(out2, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()

# ── Figure 3: Loss summary table-style plot ──
fig3, ax3 = plt.subplots(figsize=(10, 5))
# Show loss at key milestones
milestones_idx = [0, len(records)//4, len(records)//2, 3*len(records)//4, len(records)-1]
milestone_steps = [steps[i] for i in milestones_idx]
milestone_loss = [total_loss[i] for i in milestones_idx]
milestone_cls = [avg_cls_loss[i] for i in milestones_idx]
milestone_bbox = [avg_bbox_loss[i] for i in milestones_idx]

x = np.arange(len(milestones_idx))
width = 0.25
ax3.bar(x - width, milestone_loss, width, label='Total Loss', color='steelblue', alpha=0.8)
ax3.bar(x, [c * 18 for c in milestone_cls], width, label='Cls Loss (×18 heads)', color='coral', alpha=0.8)
ax3.bar(x + width, [b * 18 for b in milestone_bbox], width, label='BBox Loss (×18 heads)', color='teal', alpha=0.8)
ax3.set_xlabel("Training Milestone")
ax3.set_ylabel("Loss Value")
ax3.set_title("Loss at Key Training Milestones")
ax3.set_xticks(x)
ax3.set_xticklabels([f"Step {s}" for s in milestone_steps], rotation=15)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
out3 = os.path.join(WORK_DIR, "training_milestones.png")
plt.savefig(out3, bbox_inches='tight')
print(f"Saved: {out3}")
plt.close()

print("\nDone! All plots saved.")
