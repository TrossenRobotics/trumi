==============
Hardware Setup
==============

This page walks you through setting up the TRumi hardware before you record any data.

Assembly
========

Download :download:`this guide </_downloads/TRumi Assembly Guide.pdf>` or watch the YouTube video below for assembly instructions:

.. youtube:: do4-RMIDbxc
    :align: center

Single vs. Bimanual
===================

TRumi supports both single-gripper and bimanual (two-gripper) collection.
The two setups differ in the gripper finger identifiers: in a bimanual setup each gripper must carry its own set of identifiers so the pipeline can tell the two grippers apart.
Gripper 0 and Gripper 1 use different finger identifiers, and spare finger holders are included to configure a gripper as Gripper 0 or Gripper 1.
Their printable ArUco equivalents also use different marker IDs — see :ref:`aruco-markers`.

What's Next
===========

Continue to :doc:`/gopro_setup` to load the required camera settings, then :doc:`/software_setup` to install the pipeline.
