"""Run the full dataset generation pipeline for one or more session directories.

:Steps:
    0. Organize raw GoPro MP4 files into the demo directory structure.
    1. Extract IMU telemetry from each raw_video.mp4.
    2. Build an ORB-SLAM3 map atlas from the mapping video.
    3. Run ORB-SLAM3 localization on all demo videos.
    4. Detect ArUco tags in all demo videos.
    5. Run SLAM tag and gripper range calibrations.
    6. Generate dataset_plan.pkl.
    7. Generate the Zarr replay buffer (dataset.zarr.zip).

:Usage:
    uv run python scripts/dataset_generation_pipeline.py <session_dir> [<session_dir> ...]
"""

import logging
import pathlib
import subprocess
import sys

import click

logger = logging.getLogger(__name__)


# TODO(abhichothani42): make this scripts import modules
@click.command()
@click.argument("session_dir", nargs=-1, required=True)
@click.option("-c", "--calibration_dir", default=None)
def main(session_dir, calibration_dir):
    """Run the full dataset generation pipeline for each session directory.

    :param session_dir: One or more session directories to process.
    :param calibration_dir: Path to the calibration directory containing
        gopro13_intrinsics_2_7k.json and aruco_config.yaml. Defaults to
        example/calibration/ relative to the repository root.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    script_dir = pathlib.Path(__file__).parent.joinpath("scripts_slam_pipeline")
    if calibration_dir is None:
        calibration_dir = (
            pathlib.Path(__file__)
            .parent.parent.joinpath("example", "calibration")
            .resolve()
        )
    else:
        calibration_dir = pathlib.Path(calibration_dir).resolve()
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    for session in session_dir:
        session = pathlib.Path(session).resolve()

        # 00 Organize raw GoPro videos into demo directory structure
        logger.info("\n%s 00_process_videos %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("00_process_videos.py")

        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")

        cmd = [sys.executable, str(script_path), str(session)]
        result = subprocess.run(cmd, check=True)

        # 01 extract IMU telemetry from each raw_video.mp4 into imu_data.json
        logger.info("\n%s 01_extract_gopro_imu %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")

        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")

        cmd = [sys.executable, str(script_path), str(session)]
        result = subprocess.run(cmd, check=True)

        # 02 create ORB-SLAM3 map atlas from the mapping video
        logger.info("\n%s 02_create_map %s", "#" * 15, "#" * 15)
        logger.info("Depending on your hardware this might take a few minutes")
        script_path = script_dir.joinpath("02_create_map.py")

        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")

        demo_dir = session.joinpath("demos")
        mapping_dirs = list(demo_dir.glob("mapping_*"))
        if not mapping_dirs:
            raise FileNotFoundError(f"No mapping directory found in: {demo_dir}")
        mapping_dir = mapping_dirs[0]

        if not mapping_dir.is_dir():
            raise FileNotFoundError(f"Mapping dir not found at: {mapping_dir}.")

        map_path = mapping_dir.joinpath("map_atlas.osa")

        if not map_path.is_file():
            cmd = [
                sys.executable,
                str(script_path),
                "--input_dir",
                str(mapping_dir),
                "--map_path",
                str(map_path),
            ]
            result = subprocess.run(cmd, check=True)

            # check if map_atlas.osa file is generated
            if not map_path.is_file():
                stdout_log = mapping_dir / "slam_stdout_mapping.txt"
                stderr_log = mapping_dir / "slam_stderr_mapping.txt"
                raise FileNotFoundError(
                    f"ORB-SLAM3 did not produce map_atlas.osa at: {map_path}\n"
                    f"Check logs for details:\n"
                    f"  stdout: {stdout_log}\n"
                    f"  stderr: {stderr_log}"
                )

        # 03 run SLAM localization on all demo videos using the map atlas
        logger.info("\n%s 03_batch_slam %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("03_batch_slam.py")

        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--map_path",
            str(map_path),
        ]
        result = subprocess.run(cmd, check=True)

        # 04 detect and localize ArUco tags in all demo videos
        logger.info("\n%s 04_detect_aruco %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("04_detect_aruco.py")

        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")

        camera_intrinsics = calibration_dir.joinpath("gopro13_intrinsics_2_7k.json")
        aruco_config = calibration_dir.joinpath("aruco_config.yaml")

        if not camera_intrinsics.is_file():
            raise FileNotFoundError(f"Camera intrinsics not found: {camera_intrinsics}")
        if not aruco_config.is_file():
            raise FileNotFoundError(f"ArUco config not found: {aruco_config}")

        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--camera_intrinsics",
            str(camera_intrinsics),
            "--aruco_yaml",
            str(aruco_config),
        ]
        result = subprocess.run(cmd, check=True)

        # 05 run slam tag and gripper range calibrations
        logger.info("\n%s 05_run_calibrations %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("05_run_calibrations.py")
        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")
        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(session),
        ]
        result = subprocess.run(cmd, check=True)

        # 06 generate dataset plan
        logger.info("\n%s 06_generate_dataset_plan %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(session),
        ]
        result = subprocess.run(cmd, check=True)

        # 07 generate replay buffer
        logger.info("\n%s 07_generate_replay_buffer %s", "#" * 15, "#" * 15)
        script_path = script_dir.joinpath("07_generate_replay_buffer.py")
        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find script at: {script_path}")
        output_path = session.joinpath("dataset.zarr.zip")
        cmd = [
            sys.executable,
            str(script_path),
            "--output",
            str(output_path),
            str(session),
        ]
        result = subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
