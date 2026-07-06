===============
Data Collection
===============

This page describes how to record a data collection session and the best practices that keep your SLAM success rate high.
Recording quality is the single biggest factor in how much of your data is usable, so it is worth following these steps carefully.

Sessions and Datasets
=====================

A dataset consists of one or more **sessions**.
A session is a distinct unit of data collection, all originating from the same location.
Each session includes three key components:

- A **SLAM mapping video** — used to build a 3D map of the environment.
- A **gripper calibration video** — used to calibrate gripper finger tag detection.
- A series of **task demonstration videos** — the demonstrations themselves.

Record these in order for each session.

.. _aruco-markers:

Markers and Identifiers
=======================

TRumi uses two kinds of visual markers, both included in the kit:

- **Mapping marker** — a printed ArUco target placed in the scene to anchor the SLAM map and set metric scale.
- **Gripper finger identifiers** — embedded multicolor identifiers built into the finger mounts, used to measure gripper width.

Printable PDFs of every marker are available in the table below, in case you ever need a replacement.

.. important::

    Whenever you do print a marker, the physical printed size must match the table below **exactly**.
    Incorrect sizes produce inaccurate poses and gripper widths.

.. list-table::
    :align: center
    :header-rows: 1
    :class: centered-table

    * - Marker
      - Dictionary
      - ID
      - Size
      - PDF
    * - Mapping
      - DICT_4X4_50
      - 13
      - 0.16 m
      - :download:`download </_downloads/marker_13_mapping.pdf>`
    * - Gripper 0 left
      - DICT_4X4_50
      - 0
      - 0.016 m
      - :download:`download </_downloads/marker_0_finger_left_gripper_0.pdf>`
    * - Gripper 0 right
      - DICT_4X4_50
      - 1
      - 0.016 m
      - :download:`download </_downloads/marker_1_finger_right_gripper_0.pdf>`
    * - Gripper 1 left
      - DICT_4X4_50
      - 6
      - 0.016 m
      - :download:`download </_downloads/marker_6_finger_left_gripper_1.pdf>`
    * - Gripper 1 right
      - DICT_4X4_50
      - 7
      - 0.016 m
      - :download:`download </_downloads/marker_7_finger_right_gripper_1.pdf>`

Step 1: Mapping Video
=====================

The mapping video is used by ORB-SLAM3 to build a 3D map of the environment.
SLAM success rate is **highly sensitive** to the scene and to how you record this video.

**Scene selection tips:**

- Prefer environments with enough visual texture.
- Avoid large plain surfaces (white walls, bare ceilings, empty corners).

Place the printed mapping marker on the table and follow the mapping process carefully.
Correct marker placement is critical for SLAM success rate.

.. TODO: add photo/video demonstrating the mapping motion and marker placement.

Step 2: Gripper Calibration Video
=================================

Record a short video of opening and closing the gripper **5 times**.
This is used to calibrate gripper finger tag detection.

.. TODO: add photo/video of the gripper calibration recording.

Step 3: Demonstrations
======================

Record *N* demonstration videos.
The number of demonstrations needed depends on task complexity and environment variability.

.. tip::

    We recommend **100 demonstrations** for a single task in a fixed environment.
    Increase this for more complex tasks or greater environment variation.

.. TODO: add photo/video of a demonstration being collected.

Best Practices
==============

- **Texture matters.** SLAM relies on visual features.
  Richly textured scenes track far better than plain ones.
- **Keep the mapping marker visible and correctly placed.** It anchors the map and sets scale.
- **Move smoothly.** Avoid rapid jerks and motion blur, which degrade both SLAM and ArUco detection.
- **Keep lighting consistent.** Avoid strong glare and heavy shadows on the markers.
- **Verify your printed marker sizes** before collecting a large batch of data.
- **For bimanual collection,** confirm timecode sync (see :ref:`timecode sync <gopro_setup:timecode sync (bimanual only)>`) before recording.

If your dataset has a low SLAM success rate, revisit these practices — most low-yield sessions trace back to a poor mapping video or an incorrectly sized/placed marker.

Once you have recorded a session, continue to :doc:`/dataset_generation` to process it.
