========
Overview
========

What is TRumi?
==============

TRumi (Trossen Robotics Universal Manipulation Interface) is a **handheld manipulation data collection system**.
It is a handheld parallel-jaw gripper with a wrist-mounted GoPro that records what the gripper sees and how it moves.

Instead of teleoperating a robot to gather demonstrations, an operator performs the task directly with the handheld gripper.
This lets teams collect more manipulation demonstrations — across more objects, environments, and task variations — without setting up a full robot rig for every demo.

TRumi follows the `Universal Manipulation Interface (UMI) <https://umi-gripper.github.io/>`_ approach: it captures demonstrations in **end-effector space**, rather than being tied to one specific robot's joint configuration.
This makes the collected data well suited to large-scale data collection across many environments, before any robot-specific training and validation.

Why collect data this way?
==========================

- **Intuitive.** Operators perform tasks directly with the gripper, instead of reasoning about a robot's joint configuration during teleoperation.
- **Fast, portable, anywhere.** Scene setup is minimal — no robot, no fixed workcell.
  Collect in the environments where the task actually happens.
- **Scalable.** Supports single-gripper and bimanual (two-gripper) collection, as well as multi-site collection across many operators.
- **Not tied to one robot.** Because demonstrations are captured in end-effector space, the data is not tied to a specific robot's joint configuration.

Things to keep in mind
======================

TRumi is a **complementary** data collection tool, not a replacement for leader-follower teleoperation.
Use TRumi to scale and diversify data collection; use leader-follower teleoperation for final tuning and validation on a specific robot.

Because TRumi captures data in end-effector space, it is not directly tied to any robot hardware.
Deploying a trained policy on a real robot still requires integration with the target robot — converting policy outputs into robot motion, typically through interpolation, inverse kinematics, and validation on the actual hardware.

How it works
============

At its core, TRumi answers a single question: *where was the gripper, and how open was it, at every moment of a demonstration?* Because the GoPro is rigidly mounted to the gripper, tracking the camera through space is equivalent to tracking the gripper itself.

The workflow has three phases:

.. mermaid::

    flowchart LR
        A["<b>1 · Collect</b><br/>Perform the task with TRumi grippers<br/>GoPro records video + IMU data"]
        B["<b>2 · Process</b><br/>Visual-inertial SLAM estimates motion<br/>Extract pose and gripper width"]
        C["<b>3 · Output</b><br/>Package into a .zarr / .mcap dataset<br/>Use for downstream policy training"]
        A --> B --> C

        classDef box fill:#f2f2f2,stroke:#1e1d22,color:#1e1d22;
        class A,B,C box;

**1. Collect.** An operator performs the task while holding the TRumi gripper.
The GoPro records the scene along with its onboard IMU (accelerometer and gyroscope) data.
A one-time mapping video of the workspace is also recorded, giving SLAM a map to localize against.

**2. Process.** The dataset generation pipeline runs visual-inertial SLAM (ORB-SLAM3) to estimate the camera's 6-DoF trajectory through the mapped scene, combining the video with the IMU data for robust, metric-scale motion.
For each frame it then extracts the end-effector pose and the gripper width (how open the fingers are, measured from the finger identifiers).

**3. Output.** The synchronized frames, poses, and gripper widths are packaged into a structured ``.zarr`` or ``.mcap`` dataset, ready to feed into downstream policy-training workflows.

The rest of these docs cover each part of this workflow in detail.
If this is your first time, continue to :doc:`/specifications` and then work through the setup pages in order.

.. seealso::

    Product page: `trossenrobotics.com/trumi <https://www.trossenrobotics.com/trumi>`_
