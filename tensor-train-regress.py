"""Fit symbolic log-compression models from the shared TT analysis pickle."""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

if TYPE_CHECKING:
  from pysr import PySRRegressor

type FloatArray = NDArray[np.floating[Any]]
type Metrics = dict[str, float | int | None]
type Summary = dict[str, Any]

INPUT_FILE = pathlib.Path("results/tt-analysis.pkl")
OUTPUT_DIR = pathlib.Path("results/regression")
FEATURE_NAMES = ("mach_number", "max_mach_number_so_far", "epsilon")
TEST_FRACTION = 0.2
N_ITERATIONS = 500
MAX_SIZE = 15
RANDOM_STATE = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-file", type=pathlib.Path, default=INPUT_FILE)
  parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
  args = parser.parse_args(argv)
  args.input_file = args.input_file.expanduser()
  args.output_dir = args.output_dir.expanduser()
  return args


def load_data(input_file: pathlib.Path) -> pd.DataFrame:
  """Load and sort the shared snapshot table."""
  with input_file.open("rb") as handle:
    data = pd.DataFrame(pickle.load(handle))
  data["simulation_path"] = data["snapshot"].map(lambda path: str(pathlib.Path(path).parent))
  return data.sort_values(
    ["simulation", "simulation_path", "time"], kind="stable"
  ).reset_index(drop=True)


def split_by_time(data: pd.DataFrame) -> pd.DataFrame:
  """Label the latest 20% of each simulation as test data, keeping equal times together."""
  data = data.copy()
  data["split"] = "train"
  for _, group in data.groupby(["simulation", "simulation_path"], sort=False):
    count = max(1, int(np.ceil(len(group) * TEST_FRACTION)))
    boundary = group["time"].sort_values().iloc[-count]
    data.loc[group.index[group["time"] >= boundary], "split"] = "test"
  return data


def build_regressor(output_dir: pathlib.Path) -> PySRRegressor:
  # Import here so data preparation and plotting do not initialize Julia.
  from pysr import PySRRegressor

  return PySRRegressor(
    niterations=N_ITERATIONS,
    populations=15,
    maxsize=MAX_SIZE,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log"],
    nested_constraints={"sqrt": {"sqrt": 0}, "log": {"log": 0}},
    model_selection="best",
    precision=64,
    parallelism="serial",
    deterministic=True,
    random_state=RANDOM_STATE,
    output_directory=str(output_dir / "pysr"),
    run_id="compression",
  )


def fit_regressor(train_data: pd.DataFrame, output_dir: pathlib.Path) -> PySRRegressor:
  """Select an equation using training loss and complexity, never test error."""
  model = build_regressor(output_dir)
  model.fit(
    train_data[list(FEATURE_NAMES)].to_numpy(),
    np.log(train_data["compression_ratio"].to_numpy()),
    variable_names=list(FEATURE_NAMES),
  )
  return model


def regression_metrics(actual: FloatArray, predicted: FloatArray) -> Metrics:
  """Report errors on compression ratios; relative RMSE uses pointwise fractions."""
  residual = predicted - actual
  variance = float(np.sum((actual - np.mean(actual)) ** 2))
  return {
    "n_snapshots": len(actual),
    "mae": float(np.mean(np.abs(residual))),
    "rmse": float(np.sqrt(np.mean(residual**2))),
    "relative_rmse_percent": float(100 * np.sqrt(np.mean((residual / actual) ** 2))),
    "r2": float(1 - np.sum(residual**2) / variance) if variance else None,
    "median_relative_error": float(np.median(np.abs(residual) / actual)),
  }


def save_results(
  model: PySRRegressor, data: pd.DataFrame, output_dir: pathlib.Path
) -> Summary:
  best = model.get_best()
  log_equation = str(model.sympy())
  summary = {
    "features": list(FEATURE_NAMES),
    "target": "natural log of compression_ratio",
    "equation_log_compression": log_equation,
    "equation_compression": f"exp({log_equation})",
    "complexity": int(best["complexity"]),
    "training_log_loss": float(best["loss"]),
    "epsilons": sorted(data["epsilon"].unique().tolist()),
    "resolution": int(data["resolution"].iloc[0]),
    "simulations": sorted(data["simulation"].unique().tolist()),
    "niterations": N_ITERATIONS,
    "maxsize": MAX_SIZE,
    "random_state": RANDOM_STATE,
    "test_fraction": TEST_FRACTION,
    "split": "latest contiguous time segment within each simulation",
    "scope": "Temporal holdout within known simulations, not validation on unseen simulations.",
  }
  for split, group in data.groupby("split"):
    summary[split] = regression_metrics(
      group["compression_ratio"].to_numpy(),
      group["predicted_compression_ratio"].to_numpy(),
    )
  with (output_dir / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)

  model.equations_[["complexity", "loss", "score", "equation"]].to_csv(
    output_dir / "equations.csv", index=False
  )
  fields = [
    "simulation",
    "snapshot",
    "time",
    *FEATURE_NAMES,
    "split",
    "compression_ratio",
    "predicted_compression_ratio",
  ]
  data[fields].to_csv(output_dir / "predictions.csv", index=False)
  return summary


def plot_predictions(data: pd.DataFrame, output_dir: pathlib.Path) -> None:
  matplotlib.use("Agg")
  with matplotlib.rc_context():
    sns.set_theme(style="darkgrid", context="notebook")
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    for split, label, marker in (
      ("train", "Training", "o"),
      ("test", "Time holdout", "s"),
    ):
      group = data[data["split"] == split]
      sns.scatterplot(
        x=group["compression_ratio"],
        y=group["predicted_compression_ratio"],
        label=label,
        marker=marker,
        s=65,
        ax=ax,
      )
    values = data[["compression_ratio", "predicted_compression_ratio"]].to_numpy()
    lower, upper = values.min(), values.max()
    ax.plot([lower, upper], [lower, upper], "--", color="0.4", label="Exact prediction")
    ax.set(
      xscale="log",
      yscale="log",
      xlabel="Measured compression ratio",
      ylabel="Predicted compression ratio",
      title="Symbolic Compression Model",
    )
    ax.legend()
    fig.savefig(output_dir / "predicted-vs-measured.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  data = split_by_time(load_data(args.input_file))
  train_data = data[data["split"] == "train"]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  print(f"n_snapshots = {len(data)}")
  print(f"n_train     = {len(train_data)}")
  print(f"n_test      = {len(data) - len(train_data)}")
  print(f"features    = {', '.join(FEATURE_NAMES)}")
  print("target      = log(compression_ratio)")

  model = fit_regressor(train_data, args.output_dir)
  data["predicted_compression_ratio"] = np.exp(
    model.predict(data[list(FEATURE_NAMES)].to_numpy())
  )
  summary = save_results(model, data, args.output_dir)
  plot_predictions(data, args.output_dir)
  print(f"compression_ratio = {summary['equation_compression']}")
  print(f"Train relative RMSE = {summary['train']['relative_rmse_percent']:.2f}%")
  print(f"Test relative RMSE  = {summary['test']['relative_rmse_percent']:.2f}%")
  print(f"Saved regression results: {args.output_dir}")


if __name__ == "__main__":
  main()
