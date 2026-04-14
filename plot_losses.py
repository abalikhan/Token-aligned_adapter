import re
import matplotlib.pyplot as plt
from collections import defaultdict

log_file_path = "d3_dp_spades_3606762.out"

train_epoch_losses = defaultdict(dict)  # {epoch: {loss_name: value}}
val_epoch_losses = defaultdict(dict)    # {epoch: {loss_name: value}}
pck_epoch = defaultdict(dict)           # {epoch: {"pck@10": v, "pck@5": v}}

epochs = []
current_epoch = None  # track current epoch for val lines

with open(log_file_path, 'r') as file:
    for line in file:
        # --- Training line ---
        match_epoch = re.search(r'Train \| epoch (\d+)', line)
        if match_epoch:
            current_epoch = int(match_epoch.group(1))
            if current_epoch not in epochs:
                epochs.append(current_epoch)

            # collect all loss_xxx values in that line
            for name, val in re.findall(r'(\w+)\s+([0-9.]+)', line):
                if name.startswith("loss"):
                    train_epoch_losses[current_epoch][name] = float(val)

        # --- Validation losses (use last seen epoch) ---
        match_val = re.search(r'val_loss_(\w+):\s*([0-9.]+)', line)
        if match_val and current_epoch is not None:
            name = "val_loss_" + match_val.group(1)
            val = float(match_val.group(2))
            val_epoch_losses[current_epoch][name] = val

        # --- PCK values (also use last seen epoch) ---
        match_pck = re.search(r'pck@10_img: ([0-9.]+) \| pck@5_img: ([0-9.]+)', line)
        if match_pck and current_epoch is not None:
            pck_epoch[current_epoch]["pck@10"] = float(match_pck.group(1))
            pck_epoch[current_epoch]["pck@5"] = float(match_pck.group(2))

# --- Collect all metric names ---
all_loss_names = sorted(set(
    [ln for e in train_epoch_losses.values() for ln in e.keys()] +
    [ln for e in val_epoch_losses.values() for ln in e.keys()]
))

epoch_list = sorted(epochs)

# --- Plot each loss in its own figure ---
for loss_name in all_loss_names:
    train_vals = [train_epoch_losses[e].get(loss_name, None) for e in epoch_list]
    val_key = loss_name if loss_name.startswith("val_loss") else "val_" + loss_name
    val_vals = [val_epoch_losses[e].get(val_key, None) for e in epoch_list]

    plt.figure(figsize=(8, 5))
    if any(v is not None for v in train_vals):
        plt.plot(epoch_list, train_vals, label="Train", color="blue", marker="o")
    if any(v is not None for v in val_vals):
        plt.plot(epoch_list, val_vals, label="Val", color="orange", marker="x")
    plt.title(loss_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{loss_name}.png", dpi=300)
    plt.close()

# --- Plot PCK metrics separately ---
for metric in ["pck@10", "pck@5"]:
    vals = [pck_epoch[e].get(metric, None) for e in epoch_list]
    if any(v is not None for v in vals):
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_list, vals, label=metric,
                 color="green" if metric == "pck@10" else "red", marker="o")
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric}.png", dpi=300)
        plt.close()