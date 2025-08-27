from pathlib import Path
from twister.io import load_from_csv
from twister.statistics.statistics import extract
from twister.plotting.plotting import plot

base = Path(__file__).resolve().parent
out_root = base / "outputs"
csv_dir = out_root / "csv_predictions"
plots_dir = out_root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# 1) Rebuild PatientCollection from CSVs
pc = load_from_csv(folder=str(csv_dir) + "/")  # io.py expects trailing slash

# 2) Compute statistics
results = extract(pc, n_workers=1)

# 3) Plot PDFs per video (per patient_vK)
plotting_args = {"plotting_folder": str(plots_dir) + "/", "ext": ".svg"}
plot(results, pc, plotting_args)

print(f"Plotted reports to {plots_dir}")
