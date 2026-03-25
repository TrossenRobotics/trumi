"""
Organise raw GoPro MP4 videos into a structured demo directory for downstream processing.

:Steps:
    1. Create raw_videos dir and move all videos inside it.
    2. Rename the largest video to mapping.mp4 if not present.
    3. Move the earliest video per camera serial to gripper_calibration dir.
    4. Move all remaining videos to demos dir with metadata-based names.
    5. Create a symlink at the original raw_videos location pointing to the moved file.

:Usage:
    uv run python scripts_slam_pipeline/00_process_videos.py <session_dir> [<session_dir> ...]
"""

import pathlib
import shutil

import click
from exiftool import ExifToolHelper

from utils.common.timecode_util import mp4_get_start_datetime


@click.command(
    help="Organise raw GoPro MP4s for one or more SESSION_DIR paths. "
    "Expects MP4s under <session_dir>/raw_videos/. "
)
@click.argument("session_dir", nargs=-1, required=True)
def main(session_dir):
    """Process raw GoPro videos for one or more session directories.

    Expects MP4s to be in ``<session_dir>/raw_videos/``. Moves and renames them into
    ``<session_dir>/demos/`` with a metadata-based directory name per video, leaving
    a relative symlink at the original location for tools that reference raw paths.

    :param session_dir: One or more session directory paths containing raw MP4 videos.
    """
    for session in session_dir:
        session = pathlib.Path(session).resolve()

        # hardcode subdirs
        input_dir = session.joinpath("raw_videos")
        output_dir = session.joinpath("demos")

        # create raw_videos dir if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(
                f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside."
            )
            for mp4_path in session.glob("**/*.[mM][pP]4"):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)

        # create mapping video if don't exist
        mapping_vid_path = input_dir.joinpath("mapping.mp4")
        if (not mapping_vid_path.exists()) and not (mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in input_dir.glob("**/*.[mM][pP]4"):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path
            shutil.move(max_path, mapping_vid_path)
            print(
                f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}."
            )

        # create gripper calibration video if don't exist
        # TODO: (Abhishek) Could be done by checking video file name
        gripper_cal_dir = input_dir.joinpath("gripper_calibration")
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print(
                "raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial."
            )

            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in input_dir.glob("**/*.[mM][pP]4"):
                    if mp4_path.name.startswith("map"):
                        continue

                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta["QuickTime:CameraSerialNumber"]

                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path

            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        # create dir for each video with video specific name
        input_mp4_paths = list(input_dir.glob("**/*.[mM][pP]4"))
        print(f"Found {len(input_mp4_paths)} MP4 videos")

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta["QuickTime:CameraSerialNumber"]
                out_dname = (
                    "demo_"
                    + cam_serial
                    + "_"
                    + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                )

                # special folders
                if mp4_path.name.startswith("mapping"):
                    out_dname = "mapping"
                elif mp4_path.name.startswith(
                    "gripper_cal"
                ) or mp4_path.parent.name.startswith("gripper_cal"):
                    out_dname = (
                        "gripper_calibration_"
                        + cam_serial
                        + "_"
                        + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                    )

                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)

                # move videos
                vfname = "raw_video.mp4"
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                rel_link = out_video_path.relative_to(mp4_path.parent, walk_up=True)
                mp4_path.symlink_to(rel_link)


if __name__ == "__main__":
    main()
