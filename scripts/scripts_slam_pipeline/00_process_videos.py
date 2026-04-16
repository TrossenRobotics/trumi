"""
Organize raw GoPro MP4 videos into a structured demo directory for downstream processing.

:Steps:
    1. If raw_videos/ does not exist, create it and move all MP4s found directly
       in <session_dir> into it.
    2. If raw_videos/mapping.mp4 does not exist, rename the largest MP4 in
       raw_videos/ to mapping.mp4.
    3. If raw_videos/gripper_calibration/ does not exist, create it and move the
       earliest-recorded MP4 (excluding mapping.mp4) per camera serial into it.
    4. Move all MP4s in raw_videos/ to <session_dir>/demos/ with metadata-based
       directory names.
    5. Remove raw_videos/ if it is empty after moving all files.

:Usage:
    uv run python scripts/scripts_slam_pipeline/00_process_videos.py <session_dir> [<session_dir> ...]
"""

import logging
import pathlib
import shutil

import click
from exiftool import ExifToolHelper

from trumi.utils.timecode_util import mp4_get_start_datetime

logger = logging.getLogger(__name__)

MP4_GLOB = "**/*.[mM][pP]4"
EXIF_CAM_SERIAL_KEY = "QuickTime:CameraSerialNumber"
DEMO_DATETIME_FMT = r"%Y.%m.%d_%H.%M.%S.%f"


@click.command(
    help="Organize raw GoPro MP4s for one or more SESSION_DIR paths. "
    "Expects MP4s under <session_dir>/raw_videos/. "
)
@click.argument("session_dir", nargs=-1, required=True)
def main(session_dir):
    """Process raw GoPro videos for one or more session directories.

    Expects MP4s to be in <session_dir>/raw_videos/. Moves and renames them into
    <session_dir>/demos/ with a metadata-based directory name per video.

    :param session_dir: One or more session directory paths containing raw MP4 videos.
    """
    for session in session_dir:
        session = pathlib.Path(session).resolve()

        if not session.is_dir():
            raise click.ClickException(f"Session directory not found: '{session}'")

        # hardcode subdirs
        input_dir = session.joinpath("raw_videos")
        output_dir = session.joinpath("demos")

        # create raw_videos dir if doesn't exist
        if not input_dir.is_dir():
            mp4_paths = [p for p in session.glob("*.[mM][pP]4") if p.is_file()]
            if not mp4_paths:
                logger.info(
                    f"No MP4 files found directly in '{session}'. "
                    "Already processed or no videos to process."
                )
                continue
            input_dir.mkdir()
            logger.info(
                f"{input_dir.name} subdir doesn't exist! Creating one and moving all MP4 videos inside."
            )
            for mp4_path in mp4_paths:
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)

        # create mapping video if doesn't exist
        mapping_vid_path = input_dir.joinpath("mapping.mp4")
        if not mapping_vid_path.exists():
            mp4_paths = list(input_dir.glob(MP4_GLOB))
            if not mp4_paths:
                raise click.ClickException(
                    f"No MP4 files found in '{input_dir}'. Cannot create mapping.mp4."
                )
            max_path = max(mp4_paths, key=lambda p: p.stat().st_size)
            shutil.move(max_path, mapping_vid_path)
            logger.info(
                f"raw_videos/mapping.mp4 doesn't exist! Renaming largest file {max_path.name}."
            )

        # create gripper calibration dir if doesn't exist
        gripper_cal_dir = input_dir.joinpath("gripper_calibration")
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            logger.info(
                "raw_videos/gripper_calibration/ doesn't exist! Creating one with the first video of each camera serial."
            )

            serial_start_dict = dict()
            serial_path_dict = dict()
            mp4_paths = [
                p for p in input_dir.glob(MP4_GLOB) if not p.name.startswith("mapping")
            ]
            if not mp4_paths:
                raise click.ClickException(
                    f"No MP4 files found in '{input_dir}' (excluding mapping). Cannot create gripper_calibration/."
                )
            with ExifToolHelper() as et:
                for mp4_path in mp4_paths:
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta.get(EXIF_CAM_SERIAL_KEY)
                    if cam_serial is None:
                        raise click.ClickException(
                            f"Missing '{EXIF_CAM_SERIAL_KEY}' tag in '{mp4_path}'. "
                            f"Available keys: {list(meta.keys())}"
                        )

                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path

            for serial, path in serial_path_dict.items():
                out_path = gripper_cal_dir.joinpath(path.name)
                logger.info(f"Selected {path.name} for camera serial {serial}")
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        # create dir for each video with video specific name
        input_mp4_paths = list(input_dir.glob(MP4_GLOB))
        logger.info(f"Found {len(input_mp4_paths)} MP4 videos")

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    logger.debug(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta.get(EXIF_CAM_SERIAL_KEY)
                if cam_serial is None:
                    raise click.ClickException(
                        f"Missing '{EXIF_CAM_SERIAL_KEY}' tag in '{mp4_path}'. "
                        f"Available keys: {list(meta.keys())}"
                    )
                out_dname = (
                    "demo_"
                    + cam_serial
                    + "_"
                    + start_date.strftime(DEMO_DATETIME_FMT)
                    + "_"
                    + mp4_path.name
                )

                # special folders
                if mp4_path.name.startswith("mapping"):
                    out_dname = (
                        "mapping_"
                        + cam_serial
                        + "_"
                        + start_date.strftime(DEMO_DATETIME_FMT)
                        + "_"
                        + mp4_path.name
                    )
                elif mp4_path.parent.name.startswith("gripper_calibration"):
                    out_dname = (
                        "gripper_calibration_"
                        + cam_serial
                        + "_"
                        + start_date.strftime(DEMO_DATETIME_FMT)
                        + "_"
                        + mp4_path.name
                    )

                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)

                # move videos
                vfname = "raw_video.mp4"
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

        # remove raw_videos dir if empty
        if input_dir.is_dir():
            for subdir in input_dir.iterdir():
                if subdir.is_dir() and not any(subdir.iterdir()):
                    subdir.rmdir()
                    logger.info(f"Removed empty directory: {subdir}")
            if not any(input_dir.iterdir()):
                input_dir.rmdir()
                logger.info(f"Removed empty directory: {input_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
