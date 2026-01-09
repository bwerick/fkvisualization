# model/elements.py
# Holds all *data* objects that represent the robot structure (no UI, no OpenGL).

from __future__ import annotations  # Allows forward references in type hints

import math  # Used for default joint ranges in radians
from dataclasses import dataclass, field  # dataclass for clean data containers
from enum import Enum  # EEType enum
from typing import Union  # Element = Joint | Link

import numpy as np  # transforms + axis vectors

from .kinematics import rot_axis_angle, T_from_Rp  # pure math motion transforms


@dataclass
class Joint:
    """
    A joint advances one frame.

    - joint_type: "revolute" or "prismatic"
    - axis: joint-local axis (3,)
    - q: radians for revolute, meters for prismatic
    - T_mount: fixed transform applied before the joint motion (alignment / offset)
    """

    name: str  # human-readable name shown in the UI
    joint_type: str  # "revolute" or "prismatic"
    axis: np.ndarray  # (3,) axis in joint-local coordinates
    q: float  # joint value (rad for revolute, meters for prismatic)
    q_min: float  # slider minimum
    q_max: float  # slider maximum
    T_mount: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )  # fixed pre-motion transform

    def motion_T(self) -> np.ndarray:
        """
        Returns only the motion part of the joint as a 4x4 transform.
        """
        a = np.asarray(self.axis, dtype=float)  # ensure float array
        if self.joint_type == "revolute":
            # Revolute: rotate about axis by q radians
            R = rot_axis_angle(a, self.q)  # Rodrigues rotation
            return T_from_Rp(R, np.zeros(3, dtype=float))  # rotation + zero translation
        else:
            # Prismatic: translate along axis by q meters
            a = a / (np.linalg.norm(a) + 1e-12)  # normalize axis safely
            p = a * float(self.q)  # translation vector
            return T_from_Rp(
                np.eye(3, dtype=float), p
            )  # identity rotation + translation

    def local_T(self) -> np.ndarray:
        """
        Full local transform for this element: mount alignment, then motion.
        """
        return self.T_mount @ self.motion_T()  # apply mount first, then motion


@dataclass
class Link:
    """
    A fixed transform between frames (pure translation/rotation offset).
    In your UI, you typically use this as the "link segment" between joints.
    """

    name: str  # human-readable name shown in the UI
    T: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )  # fixed transform

    def local_T(self) -> np.ndarray:
        """
        Links contribute a fixed transform to the chain.
        """
        return self.T  # already a 4x4 transform


class EEType(str, Enum):
    """
    Visual-only end effector types (no kinematics constraints; just drawing).
    """

    FLANGE = "Flange"
    SUCTION = "Suction"
    CLAW = "Claw"


@dataclass
class EndEffector:
    """
    Visual-only end effector object.

    - ee_type: which mesh to draw
    - T: optional offset from final robot frame to the end effector root
    """

    ee_type: EEType = EEType.FLANGE  # default EE type
    T: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )  # safe default factory


# "Element" is anything in the chain that advances one frame
Element = Union[Joint, Link]
