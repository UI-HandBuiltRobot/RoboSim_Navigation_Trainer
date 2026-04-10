import json
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Interpretation text ---
LOSS_GUIDE = (
    "Loss Quality Guide\n\n"
    "> 1.0  → Poor\n"
    "0.7–1.0 → Learning, but weak\n"
    "0.3–0.7 → Decent\n"
    "< 0.3  → Strong\n\n"
    "(depends on scaling)\n\n"
    "Lower is better, but\n"
    "relative improvement matters most."
)

def plot_data(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # --- Plot losses ---
    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2)

    # --- Final value annotations ---
    ax1.scatter(epochs[-1], train_loss[-1], color="blue")
    ax1.scatter(epochs[-1], val_loss[-1], color="orange")

    ax1.text(
        epochs[-1],
        train_loss[-1],
        f"{train_loss[-1]:.3f}",
        fontsize=9,
        color="blue",
        ha="left",
        va="bottom"
    )

    ax1.text(
        epochs[-1],
        val_loss[-1],
        f"{val_loss[-1]:.3f}",
        fontsize=9,
        color="orange",
        ha="left",
        va="top"
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # --- Guide panel ---
    LOSS_GUIDE = (
        "Loss Quality Guide\n\n"
        "> 1.0  → Poor\n"
        "0.7–1.0 → Learning, but weak\n"
        "0.3–0.7 → Decent\n"
        "< 0.3  → Strong\n\n"
        "(depends on scaling)\n\n"
        "Lower is better, but\n"
        "relative improvement matters most."
    )

    ax2.axis("off")
    ax2.text(
        0, 1,
        LOSS_GUIDE,
        fontsize=10,
        va="top",
        family="monospace"
    )

    plt.tight_layout()
    plt.show()


def load_and_plot():
    file_path = filedialog.askopenfilename(
        title="Select JSON Metrics File",
        filetypes=[("JSON Files", "*.json")]
    )

    if not file_path:
        return

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        train_loss = data["train_loss"]
        val_loss = data["val_loss"]

        plot_data(train_loss, val_loss)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file:\n{e}")


# --- GUI ---
root = tk.Tk()
root.title("Metrics Plotter")
root.geometry("320x180")

btn = tk.Button(
    root,
    text="Load JSON & Plot",
    command=load_and_plot,
    height=2,
    width=20
)
btn.pack(expand=True)

root.mainloop()