"""No-op unitree_sdk2py stub for Phase 1 headless tests without cyclonedds.

Allows importing unitree_sim_isaaclab task modules. DDS calls are inert.
Real DDS requires cyclonedds + unitree_sdk2_python (Phase 1 GUI / teleop).
"""


def ChannelFactoryInitialize(*_args, **_kwargs) -> None:
    return None


class ChannelPublisher:
    def __init__(self, *_args, **_kwargs):
        pass

    def Init(self, *_args, **_kwargs):
        pass

    def Write(self, *_args, **_kwargs):
        pass


class ChannelSubscriber:
    def __init__(self, *_args, **_kwargs):
        pass

    def Init(self, *_args, **_kwargs):
        pass
