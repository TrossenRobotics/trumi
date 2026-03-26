"""Visualize an ORB-SLAM3 camera trajectory as a 3D plot.

:Steps:
    1. Load ORB-SLAM3 generated camera_trajectory.csv and separate tracked from lost frames.
    2. Draw the path (viridis: purple=start and yellow=end) optionally
       overlay RGB orientation frames every FRAME_STEP poses (--frames).
    3. Mark start/end positions, and display the plot.

:Usage:
    uv run python scripts/visualize_trajectory.py -t path/to/camera_trajectory.csv
    uv run python scripts/visualize_trajectory.py -t path/to/camera_trajectory.csv --frames
"""

import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

# Draw an orientation frame every N tracked poses to keep the plot readable.
FRAME_STEP = 5
# Length of each axis arrow in meters.
AXIS_LENGTH = 0.01


def load_trajectory(traj_file: pathlib.Path) -> pd.DataFrame:
    """Load a camera_trajectory CSV file produced by ORB-SLAM3.

    :param traj_file: Path to the CSV file.
    :return: Parsed trajectory DataFrame.
    """
    return pd.read_csv(traj_file)


@click.command()
@click.option(
    "-t",
    "--traj_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to camera_trajectory.csv produced by ORB-SLAM3.",
)
@click.option(
    "--frames",
    is_flag=True,
    default=False,
    help="Overlay RGB orientation frames every FRAME_STEP poses.",
)
def main(traj_file, frames):
    """Plot a 3D camera trajectory with a path line.

    :param traj_file: Path to camera_trajectory.csv produced by ORB-SLAM3.
    :param frames: If set, draw RGB orientation axes at every FRAME_STEP pose.
    """
    df = load_trajectory(pathlib.Path(traj_file))

    tracked = df[~df["is_lost"]]
    lost = df[df["is_lost"]]

    pct_tracked = 100.0 * len(tracked) / len(df) if len(df) else 0.0
    print(
        f"Frames : {len(df)} total | {len(tracked)} tracked ({pct_tracked:.1f}%) | {len(lost)} lost"
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(tracked) > 1:
        pts = tracked[["x", "y", "z"]].values
        segments = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 3)
        lc = Line3DCollection(segments, cmap="viridis", linewidth=1.5, alpha=0.8)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection3d(lc)
        fig.colorbar(lc, ax=ax, label="Time progress (start→end)", shrink=0.5, pad=0.1)

    if len(lost):
        ax.scatter(
            lost["x"],
            lost["y"],
            lost["z"],
            color="red",
            s=10,
            alpha=0.4,
            label=f"lost ({len(lost)})",
        )

    # Optional orientation frames every FRAME_STEP poses.
    if frames:
        for i in range(0, len(tracked), FRAME_STEP):
            row = tracked.iloc[i]
            pos = row["x"], row["y"], row["z"]
            rot = R.from_quat([row["q_x"], row["q_y"], row["q_z"], row["q_w"]])
            vx = rot.apply([1, 0, 0]) * AXIS_LENGTH
            vy = rot.apply([0, 1, 0]) * AXIS_LENGTH
            vz = rot.apply([0, 0, 1]) * AXIS_LENGTH
            ax.quiver(*pos, *vx, color="red", linewidth=0.8)
            ax.quiver(*pos, *vy, color="green", linewidth=0.8)
            ax.quiver(*pos, *vz, color="blue", linewidth=0.8)

    # Start / end markers.
    if len(tracked):
        p0 = tracked.iloc[0]
        on = tracked.iloc[-1]
        # Use .iloc to index by position, not by the original DataFrame index.
        ax.scatter(
            p0["x"], p0["y"], p0["z"], label="start", color="lime", s=80, marker="*"
        )
        ax.scatter(
            on["x"], on["y"], on["z"], label="end", color="orange", s=80, marker="*"
        )

    if len(tracked):
        lo = tracked[["x", "y", "z"]].values.min(axis=0)
        hi = tracked[["x", "y", "z"]].values.max(axis=0)
        half = (hi - lo).max() / 2.0
        mid = (lo + hi) / 2.0
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Trajectory  |  {pct_tracked:.0f}% tracked")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
