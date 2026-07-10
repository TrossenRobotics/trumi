===================
TRumi Documentation
===================

.. image:: images/trumi_action_shot.png
    :align: center
    :width: 600px

**TRumi** (Trossen Robotics Universal Manipulation Interface) is a handheld manipulation data collection system.
A parallel-jaw gripper with a wrist-mounted GoPro lets you capture manipulation demonstrations anywhere — no full robot setup required — and turn them into structured datasets for imitation learning.

These docs walk through the complete process: from setting up the hardware, to configuring the GoPro, to running the dataset generation pipeline that uses visual-inertial SLAM on the GoPro video and IMU data to estimate the gripper's motion through space.
The pipeline outputs a structured ``.zarr`` or ``.mcap`` dataset ready for downstream policy training.

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    overview.rst
    specifications.rst

.. toctree::
    :maxdepth: 1
    :caption: Setup

    hardware_setup.rst
    gopro_setup.rst
    software_setup.rst

.. toctree::
    :maxdepth: 1
    :caption: Collect & Generate

    data_collection.rst
    dataset_generation.rst
