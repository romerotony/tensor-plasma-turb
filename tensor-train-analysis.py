from __future__ import annotations

import argparse
import fcntl
import gc
import pathlib
import pickle
import subprocess
import tempfile
from typing import Any

import cmasher as cmr
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import quimb.tensor as qtn
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray
from yt.loaders import load_uniform_grid
from yt.utilities.logger import ytLogger
from yt.visualization.plot_window import SlicePlot

type FloatArray = NDArray[np.floating[Any]]
type TensorTrain = tuple[FloatArray, FloatArray, FloatArray]
type Record = dict[str, Any]

DEFAULT_INPUT = pathlib.Path.home() / "data/turbulence-tensor-data/mach-0.5"
DEFAULT_EPSILON = 0.01
MACH_THRESHOLD = 0.1
SLICE_RESOLUTION = 800
DENSITY_FIELDS = (("gas", "density"), ("gas", "density_reconstructed"))
EPS = 1e-300
MACH_COMPONENTS = (
  "prim_density",
  "prim_velocity_1",
  "prim_velocity_2",
  "prim_velocity_3",
  "prim_pressure",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT)
  parser.add_argument(
    "--output-dir",
    type=pathlib.Path,
    help="Output directory (default: results/<input directory name>).",
  )
  args = parser.parse_args(argv)
  args.input_dir = args.input_dir.expanduser()
  args.output_dir = (
    args.output_dir.expanduser()
    if args.output_dir is not None
    else pathlib.Path("results") / args.input_dir.resolve().name
  )
  return args


def select_snapshots(directory: pathlib.Path) -> list[Record]:
  """Read Mach once per output and retain everything after the first crossing."""
  snapshots = []
  started = False
  max_mach = 0.0
  print("=== Calculating Mach Numbers ===")
  for time, path in discover_snapshots(directory):
    stats = read_snapshot_stats(path)
    mach = stats["sonic_mach_number_mean"]
    max_mach = max(max_mach, mach)
    print(f"t = {time:<6.2f}    Mean Mach Number = {mach:<8.4f}")
    if mach > MACH_THRESHOLD:
      started = True
    if started:
      snapshots.append(
        {
          "path": path,
          "time": time,
          **stats,
          "max_mach_number_so_far": max_mach,
        }
      )
  return snapshots


def discover_snapshots(directory: pathlib.Path) -> list[tuple[float, pathlib.Path]]:
  """Sort all primitive outputs, including the final output, by physical time."""
  snapshots = []
  for path in directory.glob("parthenon.prim.*.phdf"):
    with h5py.File(path, "r") as handle:
      snapshots.append((float(handle["Info"].attrs["Time"]), path))
  return sorted(snapshots)


def read_snapshot_stats(path: pathlib.Path) -> dict[str, float]:
  """Read mean sonic Mach and central-slice density percentiles block by block."""
  with h5py.File(path, "r") as handle:
    gamma = float(handle["Params"].attrs["Hydro/AdiabaticIndex"])
    components = primitive_components(handle["Info"].attrs)
    prim = handle["prim"]
    slice_z = int(handle["Info"].attrs["RootGridSize"][2]) // 2
    block_nz = int(handle["Info"].attrs["MeshBlockSize"][2])
    locations = handle["LogicalLocations"][:]
    indices = [components.index(name) for name in MACH_COMPONENTS]
    total = 0.0
    count = 0
    slice_planes = []
    for block in range(prim.shape[0]):
      rho, v1, v2, v3, pressure = [
        np.asarray(prim[block, index], dtype=np.float64) for index in indices
      ]
      local_z = slice_z - int(locations[block, 2]) * block_nz
      if 0 <= local_z < block_nz:
        slice_planes.append(rho[local_z].ravel())
      rho = np.maximum(rho, EPS)
      pressure = np.maximum(pressure, 0.0)
      speed = np.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
      sound_speed = np.maximum(np.sqrt(gamma * pressure / rho), EPS)
      total += float(np.sum(speed / sound_speed))
      count += rho.size
  density_low, density_high = np.percentile(np.concatenate(slice_planes), [1, 99])
  return {
    "sonic_mach_number_mean": total / count,
    "slice_density_p01": float(density_low),
    "slice_density_p99": float(density_high),
  }


def primitive_components(info: h5py.AttributeManager) -> list[str]:
  names = list(info["OutputDatasetNames"])
  counts = list(map(int, info["NumComponents"]))
  index = names.index("prim")
  offset = sum(counts[:index])
  return list(info["ComponentNames"])[offset : offset + counts[index]]


def analyze_snapshot(snapshot: Record) -> tuple[FloatArray, FloatArray, Record]:
  density, metadata = load_density(snapshot["path"])
  cores = decompose_density(density)
  reconstructed = reconstruct_density(cores)
  metrics = reconstruction_metrics(density, reconstructed, cores)
  metadata.update(metrics)
  metadata.update(
    {
      "sonic_mach_number_mean": snapshot["sonic_mach_number_mean"],
      "max_mach_number_so_far": snapshot["max_mach_number_so_far"],
      "requested_epsilon": DEFAULT_EPSILON,
    }
  )
  return density, reconstructed, metadata


def load_density(path: pathlib.Path) -> tuple[FloatArray, Record]:
  """Assemble a uniform, ghost-free Cartesian grid in (z, y, x) order."""
  with h5py.File(path, "r") as handle:
    info = handle["Info"].attrs
    density_index = primitive_components(info).index("prim_density")
    prim = handle["prim"]
    grid_zyx = np.asarray(info["RootGridSize"], dtype=int)[::-1]
    block_zyx = np.asarray(info["MeshBlockSize"], dtype=int)[::-1]
    density = np.empty(tuple(grid_zyx), dtype=prim.dtype)
    for block, location in enumerate(handle["LogicalLocations"][:]):
      starts = location[::-1] * block_zyx
      slices = tuple(slice(int(start), int(start + size)) for start, size in zip(starts, block_zyx))
      density[slices] = prim[block, density_index]
    metadata = {
      "source": str(path.resolve()),
      "time": float(info["Time"]),
      "shape": list(density.shape),
      "root_grid_domain": info["RootGridDomain"].tolist(),
      "forcing_accel_rms": float(handle["Params"].attrs["Hydro/turbulence/accel_rms"]),
    }
  return density, metadata


def decompose_density(density: FloatArray) -> TensorTrain:
  """Apply standard TT-SVD with equal absolute error budgets for the two splits."""
  mps = qtn.MatrixProductState.from_dense(
    density,
    dims=density.shape,
    method="svd",
    cutoff=svd_cutoff(density),
    cutoff_mode="sum2",
    renorm=False,
  )
  return extract_cores(mps)


def svd_cutoff(density: FloatArray) -> float:
  density_norm = np.linalg.norm(density.astype(np.float64, copy=False))
  return float((DEFAULT_EPSILON * density_norm) ** 2 / 2)


def extract_cores(mps: qtn.MatrixProductState) -> TensorTrain:
  """Reconstruct with np.einsum("za,ayb,bx->zyx", *cores)."""
  bond_zy, bond_yx = mps.bond(0, 1), mps.bond(1, 2)
  return (
    np.asarray(mps[0].transpose("k0", bond_zy).data),
    np.asarray(mps[1].transpose(bond_zy, "k1", bond_yx).data),
    np.asarray(mps[2].transpose(bond_yx, "k2").data),
  )


def reconstruct_density(cores: TensorTrain) -> FloatArray:
  return np.einsum("za,ayb,bx->zyx", *cores, optimize=True)


def reconstruction_metrics(
  density: FloatArray, reconstructed: FloatArray, cores: TensorTrain
) -> Record:
  """Measure error one z-plane at a time, accumulating in float64."""
  core_z, _, core_x = cores
  error2 = norm2 = 0.0
  for plane, reconstructed_plane in zip(density, reconstructed):
    reconstructed_plane = reconstructed_plane.astype(np.float64)
    original = plane.astype(np.float64)
    error2 += float(np.sum((reconstructed_plane - original) ** 2))
    norm2 += float(np.sum(original**2))
  core_bytes = sum(core.nbytes for core in cores)
  return {
    "bond_dimensions": [core_z.shape[1], core_x.shape[0]],
    "compression_ratio": density.nbytes / core_bytes,
    "relative_l2_error": float(np.sqrt(error2 / norm2)) if norm2 else None,
  }


def plot_density_slices(
  density: FloatArray,
  reconstructed: FloatArray,
  metadata: Record,
  output_path: pathlib.Path,
  color_limits: tuple[float, float],
) -> None:
  """Render aligned central z slices with a shared color scale for this frame."""
  ytLogger.setLevel("ERROR")
  bbox = np.column_stack((np.full(3, -0.5), np.asarray(density.shape[::-1]) - 0.5))
  # Analysis arrays are (z, y, x); yt uniform grids expect (x, y, z).
  dataset = load_uniform_grid(
    {
      DENSITY_FIELDS[0]: (density.transpose(2, 1, 0), "code_density"),
      DENSITY_FIELDS[1]: (reconstructed.transpose(2, 1, 0), "code_density"),
    },
    density.shape[::-1],
    bbox=bbox,
    sim_time=metadata["time"],
    periodicity=(True, True, True),
  )
  plot = SlicePlot(
    dataset,
    "z",
    list(DENSITY_FIELDS),
    center=bbox.mean(axis=1),
    width=(bbox[0, 1] - bbox[0, 0], bbox[1, 1] - bbox[1, 0]),
    origin="native",
    buff_size=(SLICE_RESOLUTION, SLICE_RESOLUTION),
  )
  plot.set_axes_unit("code_length")
  cmap = cmr.cosmic
  for field in DENSITY_FIELDS:
    plot.set_unit(field, "code_density")
    plot.set_log(field, False)
    plot.set_cmap(field, cmap)
    plot.set_zlim(field, *color_limits)
    plot.set_colorbar_label(field, "Density")
  fig = plot.export_to_mpl_figure((1, 2), axes_pad=0.6, cbar_mode="single", cbar_pad="3%")
  fig.set_size_inches(12, 6.2)
  fig.subplots_adjust(left=0.05, right=0.88, bottom=0.055, top=0.80)
  cax = plot.plots[DENSITY_FIELDS[0]].cax
  cax.tick_params(axis="y", which="both", labelsize=12)
  cax.yaxis.set_label_coords(
    0.945, 0.5, transform=blended_transform_factory(fig.transFigure, cax.transAxes)
  )
  font = plot.plots[DENSITY_FIELDS[0]].axes.xaxis.label.get_fontproperties()
  for field, title in zip(DENSITY_FIELDS, ("Original Data", "TT Reconstruction")):
    ax = plot.plots[field].axes
    ax.set_title(title, fontproperties=font)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
  simulation = pathlib.Path(metadata["source"]).parent.name.removeprefix("mach-")
  fig.suptitle(
    rf"Density Slices - Mach {simulation} ($\epsilon={metadata['requested_epsilon']:g}$)"
    "\n"
    rf"$t = {metadata['time']:.2f}$"
    f" - Compression Ratio: {metadata['compression_ratio']:.1f}×",
    fontproperties=font,
    y=0.96,
  )
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  del plot, dataset
  gc.collect()


def build_analysis_record(metadata: Record) -> Record:
  source = pathlib.Path(metadata["source"])
  return {
    "simulation": source.parent.name,
    "snapshot": str(source),
    "time": float(metadata["time"]),
    "mach_number": float(metadata["sonic_mach_number_mean"]),
    "max_mach_number_so_far": float(metadata["max_mach_number_so_far"]),
    "epsilon": float(metadata["requested_epsilon"]),
    "resolution": int(metadata["shape"][0]),
    "chi_1": int(metadata["bond_dimensions"][0]),
    "chi_2": int(metadata["bond_dimensions"][1]),
    "compression_ratio": float(metadata["compression_ratio"]),
  }


def print_snapshot_summary(metadata: Record) -> None:
  error = metadata["relative_l2_error"]
  error_text = f"{error:.4f}" if error is not None else "undefined"
  print(
    f"  Bond Dims = {str(metadata['bond_dimensions']):<12}    "
    f"Compression = {f'{metadata['compression_ratio']:.2f}x':<11}    "
    f"Relative L2 Error = {error_text}",
  )


def update_analysis_pickle(records: list[Record], output_path: pathlib.Path) -> None:
  """Merge records by snapshot/epsilon under a lock and replace the file atomically."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.with_suffix(".lock").open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    existing = []
    if output_path.exists():
      with output_path.open("rb") as handle:
        existing = pickle.load(handle)
    merged = {(row["snapshot"], row["epsilon"]): row for row in existing}
    merged.update({(row["snapshot"], row["epsilon"]): row for row in records})
    rows = sorted(
      merged.values(),
      key=lambda row: (row["simulation"], row["time"], row["epsilon"], row["snapshot"]),
    )
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".tt-analysis-") as tmp:
      temporary = pathlib.Path(tmp) / output_path.name
      with temporary.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
      temporary.replace(output_path)
  print(f"Saved Snapshot Records: {output_path}")


def plot_results(records: list[Record], output_dir: pathlib.Path) -> pathlib.Path:
  matplotlib.use("Agg")
  plot_dir = output_dir.parent / "plots"
  plot_dir.mkdir(parents=True, exist_ok=True)
  simulation = pathlib.Path(records[0]["source"]).parent.name
  output = plot_dir / f"tt-analysis-{simulation}.png"
  fig = make_figure(records)
  fig.savefig(output, dpi=180)
  plt.close(fig)
  print(f"Saved {output}")
  return output


def make_figure(records: list[Record]) -> Figure:
  with matplotlib.rc_context():
    sns.set_theme(
      style="darkgrid",
      context="notebook",
      rc={
        "axes.titleweight": "bold",
        "grid.alpha": 0.65,
        "grid.color": "0.65",
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
      },
    )
    return _make_figure(records)


def _make_figure(records: list[Record]) -> Figure:
  times = np.array([record["time"] for record in records])
  ranks = np.array([record["bond_dimensions"] for record in records])
  compression = np.array([record["compression_ratio"] for record in records])
  mach = np.array([record["sonic_mach_number_mean"] for record in records])
  simulation = pathlib.Path(records[0]["source"]).parent.name.removeprefix("mach-")
  epsilon = records[0]["requested_epsilon"]
  palette = sns.color_palette("deep")
  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8.5), layout="constrained")
  fig.suptitle(rf"Tensor-Train Results - Mach {simulation} ($\epsilon = {epsilon:g}$)")
  mach_ax, rank_ax, compression_ax = axes

  plot_series(mach_ax, times, mach, marker="o", color=palette[0])
  mach_ax.set_ylabel("Mean Sonic Mach Number")
  mach_ax.set_ylim(bottom=0)
  plot_series(rank_ax, times, ranks[:, 0], label=r"$\chi_1$", marker="o", color=palette[1])
  plot_series(rank_ax, times, ranks[:, 1], label=r"$\chi_2$", marker="o", color=palette[2])
  rank_ax.set_ylabel("Bond Dimension")
  rank_ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
  rank_ax.legend()
  plot_series(compression_ax, times, compression, marker="o", color=palette[3])
  compression_ax.set_yscale("log")
  compression_ax.set_ylim(bottom=1)
  compression_ax.set_ylabel("Compression Ratio")
  compression_ax.set_xlabel("Simulation Time")

  driving_end = driving_disabled_time(records)
  for ax in axes:
    if driving_end is not None:
      ax.axvline(driving_end, color="0.35", linestyle="--", linewidth=1.2)
    ax.grid(axis="both", which="major", alpha=0.65, color="0.65")
    ax.margins(x=0.02)
  sns.despine(fig=fig)
  return fig


def plot_series(ax: Axes, times: FloatArray, values: NDArray[np.number[Any]], **style: Any) -> None:
  sns.lineplot(
    x=times,
    y=values,
    ax=ax,
    estimator=None,
    errorbar=None,
    sort=False,
    linewidth=1.8,
    markersize=7,
    **style,
  )


def driving_disabled_time(records: list[Record]) -> float | None:
  was_driven = False
  for record in records:
    acceleration = record["forcing_accel_rms"]
    if acceleration > 0:
      was_driven = True
    elif was_driven and acceleration == 0:
      return record["time"]
  return None


def create_animation(frame_dir: pathlib.Path, output_path: pathlib.Path) -> None:
  """Call ffmpeg to create animation"""
  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-loglevel",
      "quiet",
      "-framerate",
      "24",
      "-i",
      str(frame_dir / "%d.png"),
      "-vf",
      "pad=ceil(iw/2)*2:ceil(ih/2)*2",
      "-vcodec",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-b:v",
      "8M",
      str(output_path),
    ],
    check=True,
  )
  print(f"Saved {output_path}")


def slice_color_limits(snapshots: list[Record]) -> list[tuple[float, float]]:
  """Smooth slice percentile limits with a centered three-frame average."""
  limits = np.array(
    [(snapshot["slice_density_p01"], snapshot["slice_density_p99"]) for snapshot in snapshots]
  )
  smoothed = [limits[max(0, index - 1) : index + 2].mean(axis=0) for index in range(len(limits))]
  return [(float(low), float(high)) for low, high in smoothed]


def main(argv: list[str] | None = None) -> None:
  matplotlib.use("Agg")
  args = parse_args(argv)
  snapshots = select_snapshots(args.input_dir)
  color_limits = slice_color_limits(snapshots)

  metadata_records = []
  analysis_records = []
  plot_dir = args.output_dir.parent / "plots"
  plot_dir.mkdir(parents=True, exist_ok=True)
  animation = plot_dir / f"density-comparison-{args.input_dir.resolve().name}.mp4"
  print("\n=== Running Tensor Train Analysis ===")
  with tempfile.TemporaryDirectory(dir=plot_dir, prefix=".density-frames-") as tmp:
    frame_dir = pathlib.Path(tmp)
    for index, snapshot in enumerate(snapshots):
      print(f"Decomposing Snapshot t = {snapshot['time']:.2f}")
      density, reconstructed, metadata = analyze_snapshot(snapshot)
      plot_density_slices(
        density, reconstructed, metadata, frame_dir / f"{index}.png", color_limits[index]
      )
      metadata_records.append(metadata)
      analysis_records.append(build_analysis_record(metadata))
      print_snapshot_summary(metadata)
      del density, reconstructed
    print("\n=== Saving Results ===")
    create_animation(frame_dir, animation)
  update_analysis_pickle(analysis_records, args.output_dir.parent / "tt-analysis.pkl")
  plot_results(metadata_records, args.output_dir)


if __name__ == "__main__":
  main()
