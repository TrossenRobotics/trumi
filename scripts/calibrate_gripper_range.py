"""Calibrate the min/max opening width of the gripper from a gripper calibration video.

:Steps:
    1. Load tag_detection.pkl from a gripper calibration video.
    2. Auto-detect which gripper is present by finding the (left_id, right_id) ArUco
       pair with the highest joint detection rate.
    3. For each frame compute the lateral separation between the two finger tags.
    4. Save min_width (closed) and max_width (open) to a JSON file.

:Usage:
    uv run python scripts/calibrate_gripper_range.py \\
        -i path/to/tag_detection.pkl \\
        -o path/to/gripper_range.json
"""

import collections
import json
import logging
import pathlib
import pickle

import click
import numpy as np

from trumi.utils.cv_util import get_gripper_width

logger = logging.getLogger(__name__)


@click.command(help="Calibrate gripper min/max opening width from a calibration video.")
@click.option(
    "-i",
    "--input_path",
    required=True,
    type=click.Path(exists=True),
    help="tag_detection.pkl from the gripper calibration video.",
)
@click.option("-o", "--output", required=True, help="Output gripper_range.json path.")
@click.option(
    "-t",
    "--tag_det_threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum per-finger detection rate to accept a gripper.",
)
@click.option(
    "-nz",
    "--nominal_z",
    type=float,
    default=0.072,
    show_default=True,
    help="Expected depth (m) of the finger tags when computing width.",
)
def main(input_path, output, tag_det_threshold, nominal_z):
    """Calibrate gripper min/max opening width.

    :param input_path: Path to tag_detection.pkl from the gripper calibration video.
    :param output: Output path for gripper_range.json.
    :param tag_det_threshold: Minimum per-finger detection rate; exits with error if not met.
    :param nominal_z: Expected tag depth in metres used by get_gripper_width.
    """
    with open(input_path, "rb") as f:
        tag_detection_results = pickle.load(f)

    n_frames = len(tag_detection_results)
    tag_counts: dict = collections.defaultdict(int)
    for frame in tag_detection_results:
        for key in frame["tag_dict"]:
            tag_counts[key] += 1
    tag_stats: dict = {k: v / n_frames for k, v in tag_counts.items()}

    if not tag_stats:
        raise click.ClickException(
            "No ArUco tags detected in any frame of tag_detection.pkl."
        )

    # Each gripper occupies 6 consecutive tag IDs; left finger = id*6, right = id*6+1.
    tag_per_gripper = 6
    max_gripper_id = max(tag_stats) // tag_per_gripper
    gripper_prob_map = {}
    for gripper_id in range(max_gripper_id + 1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        gripper_prob = min(tag_stats.get(left_id, 0.0), tag_stats.get(right_id, 0.0))
        if gripper_prob > 0:
            gripper_prob_map[gripper_id] = gripper_prob

    if not gripper_prob_map:
        raise click.ClickException("No grippers detected in tag_detection.pkl.")

    gripper_id, gripper_prob = max(gripper_prob_map.items(), key=lambda x: x[1])
    logger.info(
        f"Detected gripper id: {gripper_id} (min per-finger detection rate: {gripper_prob:.3f})"
    )
    if gripper_prob < tag_det_threshold:
        raise click.ClickException(
            f"Detection rate {gripper_prob:.3f} < threshold {tag_det_threshold}."
        )

    left_id = gripper_id * tag_per_gripper
    right_id = left_id + 1
    gripper_widths = []
    for dt in tag_detection_results:
        width = get_gripper_width(
            dt["tag_dict"], left_id, right_id, nominal_z=nominal_z
        )
        gripper_widths.append(float("nan") if width is None else width)
    gripper_widths = np.array(gripper_widths)

    result = {
        "gripper_id": gripper_id,
        "left_finger_tag_id": left_id,
        "right_finger_tag_id": right_id,
        "max_width": float(np.nanmax(gripper_widths)),
        "min_width": float(np.nanmin(gripper_widths)),
    }
    pathlib.Path(output).write_text(json.dumps(result, indent=2))
    logger.info(f"Saved gripper range to {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
