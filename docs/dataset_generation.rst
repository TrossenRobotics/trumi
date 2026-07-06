===========================
Dataset Generation Pipeline
===========================

Once you have recorded a session, the dataset generation pipeline turns the raw GoPro videos into a structured dataset.
It extracts IMU telemetry, runs visual-inertial SLAM to estimate the gripper trajectory, detects ArUco tags, calibrates gripper width, and packages everything into an MCAP or Zarr dataset.

Before You Run
==============

1. Make sure the Docker daemon is running:

   .. code-block:: bash

       docker info

2. Organize your session directory with the appropriate videos:

   .. code-block:: text

       <session_dir>/
       └── raw_videos/
           ├── mapping.mp4               # rename your mapping video to this
           ├── gripper_calibration/      # place gripper calibration video(s) here
           │   └── *.mp4                 # one video per camera serial
           └── *.mp4                     # remaining videos are treated as demonstrations

Run the Pipeline
================

Run the full pipeline with:

.. code-block:: bash

    uv run python scripts/dataset_generation_pipeline.py <session_dir>

For example, using the downloaded example dataset:

.. code-block:: bash

    uv run python scripts/dataset_generation_pipeline.py example_gopro13_dataset

You can pass more than one session directory to process several sessions in sequence.
By default the pipeline writes an MCAP dataset; pass ``-f zarr`` to write Zarr instead (see :ref:`dataset_generation-dataset-formats`).

Example output (truncated to the final steps):

.. code-block:: text

    ...
    ############### 06_generate_dataset_plan ###############
    INFO: Found following cameras:
    camera_serial
    C3534250760071    2
    INFO: Assigned camera_idx: right=0; left=1; non_gripper=2,3...
                 camera_serial  gripper_hw_idx                                                  example_vid
    camera_idx
    0           C3534250760071               0  demo_C3534250760071_2026.03.31_20.59.03.643175_GX010168.MP4
    INFO: 99% of raw data are used.
    INFO: Dropped demos: 0
    INFO: Saved dataset plan (2 episodes) to example_gopro13_dataset/dataset_plan.pkl
    INFO:
    ############### 07_generate_dataset (mcap) ###############
    INFO: Collected 2 episodes, 1 grippers, 1 cameras.
    INFO: Writing 2 episode MCAP files to example_gopro13_dataset/dataset_mcap
    Episodes: 100%|████████████████████████████████████████| 2/2 [00:19<00:00,  9.98s/it]
    INFO: Done! 2 episode MCAP files written to example_gopro13_dataset/dataset_mcap

For this dataset, 99% of the data are usable (successful SLAM), with 0 demonstrations dropped.
If your dataset has a low SLAM success rate, revisit the :doc:`/data_collection` guidance.

Pipeline Stages
===============

The pipeline runs eight stages in sequence.
Each stage is an individual script under ``scripts/scripts_slam_pipeline/`` and can also be run on its own.

.. list-table::
    :align: center
    :header-rows: 1
    :class: centered-table

    * - Stage
      - Script
      - What it does
    * - 00
      - ``00_process_videos.py``
      - Organizes raw GoPro MP4 files into the demo directory structure.
    * - 01
      - ``01_extract_gopro_imu.py``
      - Extracts IMU telemetry from each ``raw_video.mp4`` into ``imu_data.json``.
    * - 02
      - ``02_create_map.py``
      - Builds an ORB-SLAM3 map atlas (``map_atlas.osa``) from the mapping video.
    * - 03
      - ``03_batch_slam.py``
      - Runs ORB-SLAM3 localization on all demo videos using the map atlas.
    * - 04
      - ``04_detect_aruco.py``
      - Detects and localizes ArUco tags in all demo videos.
    * - 05
      - ``05_run_calibrations.py``
      - Runs SLAM tag and gripper range calibrations.
    * - 06
      - ``06_generate_dataset_plan.py``
      - Generates ``dataset_plan.pkl``.
    * - 07
      - ``07_generate_mcap_dataset.py`` / ``07_generate_zarr_dataset.py``
      - Packages the demonstrations into the chosen output format (MCAP or Zarr).

.. _dataset_generation-slam-frame-stride:

SLAM Frame Stride
=================

SLAM processes frames at a lower rate than the recorded video to ensure enough IMU samples accumulate between consecutive SLAM frames (a minimum of 3 samples/frame at a 200 Hz IMU rate).
For a 120 fps recording, this means SLAM runs at 60 fps (``skip=2``).

Check the ``skip`` value for your recording in the log file produced during the mapping SLAM step (``<session_dir>/demos/mapping_*/slam_stdout_mapping.txt``):

.. code-block:: text

    Video: 119.88 fps  |  SLAM: 59.9401 fps (skip=2)  |  IMU/frame: 3.33667

Pass this value as ``--slam_frame_stride`` to ``dataset_generation_pipeline.py``:

.. code-block:: bash

    uv run python scripts/dataset_generation_pipeline.py <session_dir> --slam_frame_stride 2

The same value must be passed to ``04_detect_aruco.py``, ``06_generate_dataset_plan.py``, ``07_generate_mcap_dataset.py``, and ``07_generate_zarr_dataset.py`` when running those steps manually.
The default is ``2`` (120 fps → 60 fps SLAM).

Inspecting Results
==================

Two visualization scripts help you inspect intermediate results.

**SLAM trajectory** — plots the ORB-SLAM3 camera trajectory from a session's ``camera_trajectory.csv`` as a 3D path:

.. code-block:: bash

    uv run python scripts/visualize_trajectory.py -t <demo_dir>/camera_trajectory.csv

**ArUco tag detections** — renders tag detections overlaid on the source video and writes an annotated MP4:

.. code-block:: bash

    uv run python scripts/visualize_aruco_video.py \
        -i <demo_dir> \
        -ci <path/to/intrinsics.json> \
        -sfs <slam_frame_stride> \
        -o <output.mp4>

.. _dataset_generation-dataset-formats:

Dataset Formats
===============

The pipeline supports two output formats, selected with ``--format`` (``-f``):

.. code-block:: bash

    # MCAP (default)
    uv run python scripts/dataset_generation_pipeline.py <session_dir>

    # Zarr
    uv run python scripts/dataset_generation_pipeline.py -f zarr <session_dir>

**MCAP** writes one ``.mcap`` file per episode into a ``dataset_mcap/`` directory.
Each file contains time-aligned robot state, JPEG-compressed camera images, and IMU telemetry (accelerometer and gyroscope) as typed, self-describing messages.
MCAP files can be inspected with `Foxglove <https://foxglove.dev/>`_.

**Zarr** writes a single ``dataset.zarr.zip`` archive containing all episodes in a flat, NumPy-backed replay buffer with JpegXL-compressed images.

Both formats store per-step end-effector pose (position and axis-angle rotation), gripper width, demo start/end poses, and camera images.
MCAP additionally includes the raw IMU samples.
Alongside the dataset, the pipeline also writes ``dataset_plan.pkl`` — the plan describing which demonstrations are used.
