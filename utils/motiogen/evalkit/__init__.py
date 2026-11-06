"""evalkit: standard evaluation toolset (config/evalkit_v1_20261107.json).

Dimensions share the input contract (io), the floor parameter (floor) and the IK pass (ik) and never gate each other:
residual (whole-body residual force, hover / penetration depth), ergo (RULA/REBA frame classes), muscle (static
optimization effort), kinematic (reach, jerk, bone length, foot skating) and distribution (native evaluators).
Modules that need OpenSim run in the OpenSim python; the others in the motiogen environments.
"""
