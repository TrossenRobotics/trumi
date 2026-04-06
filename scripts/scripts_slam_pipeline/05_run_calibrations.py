"""Run SLAM tag and gripper range calibrations for a session directory.

:Steps:
    1. Run calibrate_slam_tag.py on the mapping video to produce tx_slam_tag.json.
    2. Run calibrate_gripper_range.py on gripper_calibration directory
       to produce gripper_range.json.

:Usage:
    uv run python scripts_slam_pipeline/05_run_calibrations.py -i <session_dir>
"""

import logging
import pathlib
import subprocess
import sys

import click

logger = logging.getLogger(__name__)


@click.command(
    help="Run SLAM tag and gripper range calibrations for a session directory."
)
@click.option("-i", "--input_dir", required=True, help="Session directory.")
@click.option(
    "-t",
    "--tag_det_threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum per-finger detection rate to accept a gripper (passed to calibrate_gripper_range.py).",
)
@click.option(
    "-nz",
    "--nominal_z",
    type=float,
    default=0.072,
    show_default=True,
    help="Expected depth (m) of the finger tags when computing width (passed to calibrate_gripper_range.py).",
)
def main(input_dir, tag_det_threshold, nominal_z):
    """Run calibrate_slam_tag and calibrate_gripper_range for a session directory.

    :param input_dir: Session directory containing a demos/mapping subdirectory
        and one or more demos/gripper_calibration* subdirectories.
    :param tag_det_threshold: Minimum per-finger detection rate passed to calibrate_gripper_range.py.
    :param nominal_z: Expected finger tag depth in metres passed to calibrate_gripper_range.py.
    """
    script_dir = pathlib.Path(__file__).parent.parent.joinpath("scripts")
    session = pathlib.Path(input_dir).resolve()
    demos_dir = session.joinpath("demos")
    mapping_dirs = list(demos_dir.glob("mapping_*"))
    if not mapping_dirs:
        raise click.ClickException(f"No mapping_* directory found in: {demos_dir}")
    mapping_dir = mapping_dirs[0]

    # slam tag calibration
    script_path = script_dir.joinpath("calibrate_slam_tag.py")
    if not script_path.is_file():
        raise click.ClickException(f"Script not found: {script_path}")

    tag_path = mapping_dir.joinpath("tag_detection.pkl")
    if not tag_path.is_file():
        raise click.ClickException(f"tag_detection.pkl not found in: {mapping_dir}")

    csv_path = mapping_dir.joinpath("camera_trajectory.csv")
    if not csv_path.is_file():
        csv_path = mapping_dir.joinpath("mapping_camera_trajectory.csv")
        logger.info(
            "camera_trajectory.csv not found, falling back to mapping_camera_trajectory.csv"
        )
    if not csv_path.is_file():
        raise click.ClickException(f"No trajectory CSV found in: {mapping_dir}")

    slam_tag_path = mapping_dir.joinpath("tx_slam_tag.json")
    cmd = [
        sys.executable,
        str(script_path),
        "--tag_detection",
        str(tag_path),
        "--csv_trajectory",
        str(csv_path),
        "--output",
        str(slam_tag_path),
        "--keyframe_only",
    ]
    subprocess.run(cmd, check=True)

    # gripper range calibration
    script_path = script_dir.joinpath("calibrate_gripper_range.py")
    if not script_path.is_file():
        raise click.ClickException(f"Script not found: {script_path}")

    for gripper_dir in sorted(demos_dir.glob("gripper_calibration*")):
        tag_path = gripper_dir.joinpath("tag_detection.pkl")
        if not tag_path.is_file():
            raise click.ClickException(f"tag_detection.pkl not found in: {gripper_dir}")
        gripper_range_path = gripper_dir.joinpath("gripper_range.json")
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(tag_path),
            "--output",
            str(gripper_range_path),
            "--tag_det_threshold",
            str(tag_det_threshold),
            "--nominal_z",
            str(nominal_z),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
