import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load training log
with open('checkpoints/training_log.json') as f:
    log = json.load(f)

epochs      = [r['epoch'] for r in log]
total_loss  = [r.get('train_total', r.get('total', 0)) for r in log]
data_loss   = [r.get('train_data',  r.get('data',  0)) for r in log]
phys_loss   = [r.get('train_physics', r.get('physics', 0)) for r in log]
val_csi     = [r.get('val_csi', 0) for r in log]
phys_weight = [r.get('physics_weight', 0) for r in log]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
fig.patch.set_facecolor('white')

# ── Left: Loss curves ─────────────────────────────────────────────
ax1.plot(epochs, total_loss, color='#1565C0', lw=2,   label='Total loss')
ax1.plot(epochs, data_loss,  color='#2E7D32', lw=1.5,
         linestyle='--', label='Data loss')
ax1.plot(epochs, phys_loss,  color='#B71C1C', lw=1.5,
         linestyle=':', label='Physics loss')

# Shade warmup region (first 10 epochs)
warmup_end = next((e for e, w in zip(epochs, phys_weight) if w >= 0.99), 10)
ax1.axvspan(1, warmup_end, alpha=0.08, color='gray', label=f'Warmup (ep 1–{warmup_end})')

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min(epochs), max(epochs))
ax1.set_ylim(bottom=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Right: Validation CSI ─────────────────────────────────────────
ax2.fill_between(epochs, val_csi, alpha=0.15, color='#FF8F00')
ax2.plot(epochs, val_csi, color='#FF8F00', lw=2.5, label='Val CSI')
ax2.axhline(y=0.65, color='gray', linestyle='--', lw=1,
            label='Skillful threshold (0.65)')

# Mark best CSI epoch
best_epoch = epochs[np.argmax(val_csi)]
best_csi   = max(val_csi)
ax2.annotate(f'Best: {best_csi:.3f}\n(ep {best_epoch})',
             xy=(best_epoch, best_csi),
             xytext=(best_epoch + max(epochs)*0.05, best_csi - 0.08),
             fontsize=8, color='#E65100',
             arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2))

ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('CSI', fontsize=11)
ax2.set_title('Validation CSI', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.set_xlim(min(epochs), max(epochs))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Neural Weather Twin — Training Dynamics (60 Epochs)',
             fontsize=11, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
print("✅ Saved: loss_curves.png")