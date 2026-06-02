"""Corre los 6 experimentos y compara resultados."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import train_one_experiment

targets = ["GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]
results_summary = []

for target in targets:
    print(f"\nEjecutando experimento: {target} ...")
    res = train_one_experiment(
        data_path="dataset/deterioro_cognitivo.sav",
        target_name=target,
        epochs=50,
    )
    results_summary.append({
        "target": target,
        "hamming_mean": res["summary"]["mean_hamming_loss"],
        "hamming_std":  res["summary"]["std_hamming_loss"],
        "n_clases": len(res["classes"]),
    })

print("\n=== Comparacion de experimentos ===")
print(f"{'Target':8s} | {'Clases':6s} | {'Hamming Mean':12s} | {'Hamming Std':11s}")
print("-" * 50)
for r in results_summary:
    print(f"{r['target']:8s} | {r['n_clases']:6d} | {r['hamming_mean']:12.4f} | {r['hamming_std']:11.4f}")
