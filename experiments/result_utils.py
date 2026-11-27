"""Save federated experiment metrics and generate comparison plots."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def save_results(split_type, rows):
    """Write one CSV row per client and round, then refresh all plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{split_type}_results.csv"
    fields = [
        "round", "client_id", "dataset_size", "train_loss", "test_size",
        "test_loss", "accuracy", "global_loss", "global_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    generate_plots()


def _read_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def generate_plots():
    """Generate clear, consistently formatted PNG plots from result CSV files."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; plots were skipped.")
        return

    iid = _read_csv("iid_results.csv")
    noniid = _read_csv("noniid_results.csv")
    centralized = _read_csv("centralized_results.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"IID": "#2563eb", "Non-IID": "#dc2626", "Centralized": "#16a34a"}

    def by_round(rows, field):
        """Return one mean value per round, sorted by round number."""
        values = {}
        for row in rows:
            value = row.get(field, "")
            if value not in (None, ""):
                values.setdefault(int(row["round"]), []).append(float(value))
        ordered = sorted(values.items())
        return [number for number, _ in ordered], [sum(v) / len(v) for _, v in ordered]

    def save_plot(filename):
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
        plt.close()

    def add_value_labels(x, y, formatter):
        for number, value in zip(x, y):
            plt.annotate(formatter(value), (number, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=8)

    datasets = (("IID", iid), ("Non-IID", noniid))

    # Accuracy is displayed as a percentage so the result is immediately readable.
    plt.figure(figsize=(9, 5.5))
    plotted = False
    for label, rows in datasets:
        x, y = by_round(rows, "global_accuracy")
        if x:
            plotted = True
            y_percent = [value * 100 for value in y]
            plt.plot(x, y_percent, marker="o", linewidth=2, color=colors[label], label=label)
            add_value_labels(x, y_percent, lambda value: f"{value:.1f}%")
    plt.xlabel("Federated training round")
    plt.ylabel("Global accuracy (%)")
    plt.title("Global accuracy during federated training")
    plt.ylim(0, 100)
    if plotted:
        plt.legend(title="Data split")
    save_plot("accuracy_per_round.png")

    # Keep the historical comparison filename, but make its meaning explicit.
    plt.figure(figsize=(9, 5.5))
    for label, rows in datasets:
        x, y = by_round(rows, "global_accuracy")
        if x:
            plt.plot(x, [value * 100 for value in y], marker="o", linewidth=2,
                     color=colors[label], label=label)
    plt.xlabel("Federated training round")
    plt.ylabel("Global accuracy (%)")
    plt.title("Effect of IID versus Non-IID data distribution")
    plt.ylim(0, 100)
    plt.legend(title="Data split")
    save_plot("iid_vs_noniid.png")

    # Loss complements accuracy and makes convergence problems easier to spot.
    plt.figure(figsize=(9, 5.5))
    for label, rows in datasets:
        x, y = by_round(rows, "global_loss")
        if x:
            plt.plot(x, y, marker="o", linewidth=2, color=colors[label], label=label)
    plt.xlabel("Federated training round")
    plt.ylabel("Global cross-entropy loss (lower is better)")
    plt.title("Global loss during federated training")
    plt.legend(title="Data split")
    save_plot("loss_per_round.png")

    labels, values = [], []
    for label, rows in (("IID", iid), ("Non-IID", noniid), ("Centralized", centralized)):
        if rows:
            labels.append(label)
            values.append(float(
                rows[-1].get("global_accuracy", rows[-1].get("val_acc", 0.0))
            ))
    plt.figure(figsize=(8, 5.5))
    if values:
        bars = plt.bar(labels, [value * 100 for value in values],
                       color=[colors[label] for label in labels])
        for bar, value in zip(bars, values):
            plt.annotate(f"{value * 100:.1f}%", (bar.get_x() + bar.get_width() / 2, value * 100),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    plt.ylabel("Final global accuracy (%)")
    plt.title("Final accuracy: federated versus centralized training")
    plt.ylim(0, 100)
    save_plot("federated_vs_centralized.png")

    plt.figure(figsize=(9, 5.5))
    for label, rows in (("IID", iid), ("Non-IID", noniid)):
        if rows:
            costs = {}
            for row in rows:
                costs.setdefault(int(row["round"]), 0)
                costs[int(row["round"])] += int(row["dataset_size"])
            x = sorted(costs)
            plt.plot(x, [costs[k] for k in x], marker="o", linewidth=2,
                     color=colors[label], label=label)
    plt.xlabel("Federated training round")
    plt.ylabel("Client samples processed in that round")
    plt.title("Approximate data processed per federated round")
    plt.legend(title="Data split")
    save_plot("communication_cost.png")

    plt.figure(figsize=(8, 5.5))
    labels, values = [], []
    for label, rows in (("IID", iid), ("Non-IID", noniid)):
        if rows:
            labels.append(label)
            values.append(sum(int(row["dataset_size"]) for row in rows) / len(rows))
    if values:
        bars = plt.bar(labels, values, color=[colors[label] for label in labels])
        for bar, value in zip(bars, values):
            plt.annotate(f"{value:,.0f}", (bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    plt.ylabel("Average samples per client")
    plt.title("Average client dataset size by data split")
    save_plot("client_comparison.png")
