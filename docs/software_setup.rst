==============
Software Setup
==============

This page describes how to install the TRumi dataset generation pipeline on a Linux PC.

Prerequisites
=============

- **Python 3.12**
- `uv <https://docs.astral.sh/uv/>`_ package manager
- `Docker <https://docs.docker.com/get-started/get-docker/>`_ (required for ORB-SLAM3)
- `ExifTool <https://exiftool.org/>`_

On Ubuntu, install ExifTool with:

.. code-block:: bash

    sudo apt install libimage-exiftool-perl

Install the Package
===================

Clone the repository and install the package:

.. code-block:: bash

    cd ~
    git clone https://github.com/TrossenRobotics/trumi.git
    cd trumi
    uv sync

This installs all dependencies declared in ``pyproject.toml`` into a local ``.venv``.

To activate the environment manually:

.. code-block:: bash

    source .venv/bin/activate

Or prefix any command with ``uv run`` to automatically use the environment without activating it.

Build the ORB-SLAM3 Docker Image
================================

The SLAM pipeline runs ORB-SLAM3 inside a Docker container.
Build the image once before running the pipeline.

Follow the setup instructions in `TrossenRobotics/ORB_SLAM3 — DOCKER.md <https://github.com/TrossenRobotics/ORB_SLAM3/blob/master/DOCKER.md>`_.

Verify Docker is running before you use the pipeline:

.. code-block:: bash

    docker info

Developer Setup (Optional)
==========================

If you plan to contribute, install the pre-commit hooks:

.. code-block:: bash

    uv run pre-commit install

Try It Without Recording (Optional)
===================================

An example dataset is available on `Hugging Face <https://huggingface.co/datasets/TrossenRoboticsCommunity/example_trumi_dataset>`_ so you can try the pipeline before recording your own data.

Download it with the `Hugging Face CLI <https://huggingface.co/docs/huggingface_hub/guides/cli>`_ (downloads into the current directory, preserving the ``example_trumi_dataset/`` folder structure):

.. code-block:: bash

    cd ~/trumi
    hf download TrossenRoboticsCommunity/example_trumi_dataset \
        --repo-type dataset \
        --local-dir .

With the software installed, continue to :doc:`/data_collection` to record a session, or jump straight to :doc:`/dataset_generation` if you are using the example dataset.
