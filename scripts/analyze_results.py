import pickle as pkl
import numpy as np
from rich.console import Console
from rich.table import Table
from bci_disc_models.conf import METRICS, MODELS
from bci_disc_models.utils import PROJECT_ROOT

def summarize_confusion_matrix(cm):
    """Summarize key statistics from the confusion matrix."""
    # This function can be expanded to provide more detailed summaries
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    return f"Acc: {accuracy:.3f}"

results_dir = PROJECT_ROOT / "results"

with open(results_dir / "parsed_results.pkl", "rb") as f:
    parsed_results = pkl.load(f)

columns = ["Model"] + METRICS + ["Confusion Matrix Summary"]
rows = []
for model in MODELS:
    row = [model]
    for metric in METRICS:
        mean = np.mean(parsed_results[model][metric])
        std = np.std(parsed_results[model][metric])
        row.append(f"{mean:.3f} ± {std:.3f}")
    
    # Add confusion matrix summary
    cm_summaries = [summarize_confusion_matrix(cm) for cm in parsed_results[model].get('confusion_matrix', [])]
    cm_summary = ', '.join(cm_summaries) if cm_summaries else 'N/A'
    row.append(cm_summary)
    
    rows.append(row)

table = Table(title="Model Comparison")
for col in columns:
    table.add_column(col, no_wrap=True)
for row in rows:
    table.add_row(*map(str, row))
console = Console()
console.print(table)

# Optionally, print detailed confusion matrices after the table
console.print("\nDetailed Confusion Matrices:")
for model in MODELS:
    if 'confusion_matrix' in parsed_results[model]:
        console.print(f"\n{model}:")
        for cm in parsed_results[model]['confusion_matrix']:
            console.print(cm)

