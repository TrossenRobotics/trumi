"""
Create an ORB-SLAM3 map atlas from a GoPro mapping video using a Docker-based SLAM runner.

:Steps:
    1. Validate that raw_video.mp4 and imu_data.json exist in input_dir.
    2. Generate a predefined SLAM mask (mirror + finger regions).
    3. Run the ORB-SLAM3 Docker container with the video, IMU data, and optional mask.
       ORB-SLAM3 Docker container automatically subsamples frames so IMU samples per frame >= 3.
       (eg. every 2nd frame at 120 fps so running SLAM at 60 fps.)
    4. Output map atlas (*.osa) and camera trajectory CSV.

:Usage:
    uv run python scripts/scripts_slam_pipeline/02_create_map.py --input_dir <mapping_dir> [--map_path <output.osa>]
"""

import logging
import pathlib
import subprocess

import click
import cv2
import numpy as np

from trumi.utils.cv_util import GOPRO_2_7K_RESOLUTION, draw_predefined_mask

logger = logging.getLogger(__name__)

# TODO(abhichothani42): Add Camera setting.yaml file for difference resolution
# and different camera models, make it configurable
# ORB-SLAM3 camera/IMU settings file path (inside the Docker container)
SLAM_SETTING = (
    "/ORB_SLAM3/Examples/Monocular-Inertial/gopro13_maxlens_fisheye_setting_v1_720.yaml"
)


@click.command(help="Create an ORB-SLAM3 map atlas from a GoPro mapping video.")
@click.option("-i", "--input_dir", required=True, help="Directory for mapping video")
@click.option(
    "-m",
    "--map_path",
    default=None,
    help="Output ORB_SLAM3 *.osa map atlas. Defaults to <input_dir>/map_atlas.osa",
)
@click.option("-d", "--docker_image", default="orb_slam3")
@click.option(
    "-nm",
    "--no_mask",
    is_flag=True,
    default=False,
    help="Whether to mask out finger and mirrors. Set if map is created with bare GoPro not on gripper.",
)
def main(input_dir, map_path, docker_image, no_mask):
    """Run ORB-SLAM3 map creation for a mapping video directory.

    Generates a SLAM mask from the predefined mirror and finger polygons, mounts the
    video directory into a Docker container, and runs the monocular-inertial SLAM pipeline.

    :param input_dir: Directory containing raw_video.mp4 (mapping video) and imu_data.json.
    :param map_path: Output path for the ORB-SLAM3 map atlas (*.osa). Defaults to
        <input_dir>/map_atlas.osa.
    :param docker_image: Name of the Docker image containing the ORB-SLAM3 binary.
    :param no_mask: If True, skip mask generation and run SLAM on the raw video.
    """
    video_dir = pathlib.Path(input_dir).resolve()

    for fn in ["raw_video.mp4", "imu_data.json"]:
        file_path = video_dir / fn
        if not file_path.is_file():
            raise click.ClickException(f"Required file missing: {file_path}")

    if map_path is None:
        map_path = video_dir.joinpath("map_atlas.osa")
    else:
        map_path = pathlib.Path(map_path).resolve()

    # ensure the output directory exists whether map_path is default or user-provided
    map_path.parent.mkdir(parents=True, exist_ok=True)

    mount_target = pathlib.Path("/data")
    csv_path = mount_target.joinpath("mapping_camera_trajectory.csv")
    video_path = mount_target.joinpath("raw_video.mp4")
    json_path = mount_target.joinpath("imu_data.json")
    mask_path = mount_target.joinpath("slam_mask.png")
    if not no_mask:
        mask_write_path = video_dir.joinpath("slam_mask.png")
        slam_mask = np.zeros(GOPRO_2_7K_RESOLUTION, dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True
        )
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path("/map").joinpath(map_mount_source.name)

    # run SLAM
    cmd = [
        "docker",
        "run",
        "--rm",  # delete after finish
        "--volume",
        str(video_dir) + ":" + "/data",
        "--volume",
        str(map_mount_source.parent) + ":" + str(map_mount_target.parent),
        docker_image,
        "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
        "--vocabulary",
        "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting",
        SLAM_SETTING,
        "--input_video",
        str(video_path),
        "--input_imu_json",
        str(json_path),
        "--output_trajectory_csv",
        str(csv_path),
        "--save_map",
        str(map_mount_target),
    ]
    if not no_mask:
        cmd.extend(["--mask_img", str(mask_path)])

    stdout_path = video_dir.joinpath("slam_stdout_mapping.txt")
    stderr_path = video_dir.joinpath("slam_stderr_mapping.txt")

    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        result = subprocess.run(
            cmd, cwd=str(video_dir), stdout=stdout_f, stderr=stderr_f
        )

    if result.returncode != 0:
        raise click.ClickException(
            f"SLAM failed (exit code {result.returncode}). Check {stderr_path}"
        )

    logger.info("Done. Map saved to: %s", map_path)
    logger.info("Logs: %s, %s", stdout_path.name, stderr_path.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
