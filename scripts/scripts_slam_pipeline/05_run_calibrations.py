"""Run SLAM tag and gripper range calibrations for a session directory.

:Steps:
    1. Run calibrate_slam_tag.py on the mapping video to produce tx_slam_tag.json.
    2. Run calibrate_gripper_range.py on gripper_calibration directory
       to produce gripper_range.json.

:Usage:
    uv run python scripts_slam_pipeline/05_run_calibrations.py -i <session_dir>
"""

import pathlib
import subprocess
import sys

import click


@click.command(
    help="Run SLAM tag and gripper range calibrations for a session directory."
)
@click.option("-i", "--input_dir", required=True, help="Session directory.")
def main(input_dir):
    """Run calibrate_slam_tag and calibrate_gripper_range for a session directory.

    :param input_dir: Session directory containing a demos/mapping subdirectory
        and one or more demos/gripper_calibration* subdirectories.
    """
    script_dir = pathlib.Path(__file__).parent.parent.joinpath("scripts")
    session = pathlib.Path(input_dir).resolve()
    demos_dir = session.joinpath("demos")
    mapping_dir = demos_dir.joinpath("mapping")

    # slam tag calibration
    script_path = script_dir.joinpath("calibrate_slam_tag.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    tag_path = mapping_dir.joinpath("tag_detection.pkl")
    if not tag_path.is_file():
        raise FileNotFoundError(f"tag_detection.pkl not found in: {mapping_dir}")

    csv_path = mapping_dir.joinpath("camera_trajectory.csv")
    if not csv_path.is_file():
        csv_path = mapping_dir.joinpath("mapping_camera_trajectory.csv")
        print(
            "camera_trajectory.csv not found, falling back to mapping_camera_trajectory.csv"
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"No trajectory CSV found in: {mapping_dir}")

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
    result = subprocess.run(cmd, check=True)

    # gripper range calibration
    script_path = script_dir.joinpath("calibrate_gripper_range.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    for gripper_dir in sorted(demos_dir.glob("gripper_calibration*")):
        tag_path = gripper_dir.joinpath("tag_detection.pkl")
        if not tag_path.is_file():
            raise FileNotFoundError(f"tag_detection.pkl not found in: {gripper_dir}")
        gripper_range_path = gripper_dir.joinpath("gripper_range.json")
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(tag_path),
            "--output",
            str(gripper_range_path),
            "--tag_det_threshold",
            "0.05",
            "--nominal_z",
            "0.080",
        ]
        result = subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
