# model/robot.py
# Holds the RobotChain class which manages elements and computes FK frames.

from __future__ import annotations  # forward refs in typing

import math  # for default joint ranges
from typing import List, Optional  # typing helpers

import numpy as np  # transforms

from .elements import (
    Joint,
    Link,
    EndEffector,
    EEType,
    Element,
)  # structure objects


class RobotChain:
    """
    A simple serial chain made of:
      - Joint elements (revolute/prismatic)
      - Link elements (fixed transforms)
    Each element advances the current transform by multiplying its local_T().
    """

    def __init__(self):
        # Ordered list of joints/links describing the chain
        self.elements: List[Element] = []

        # Visual-only end effector, optional
        self.end_effector: Optional[EndEffector] = EndEffector(
            EEType.FLANGE, np.eye(4, dtype=float)
        )

    # ---- element creation helpers (used by UI buttons) ----

    def add_revolute_joint(self):
        # Number joints so they look nice in the UI list
        jn = sum(isinstance(e, Joint) for e in self.elements) + 1

        # Append a default revolute joint about +Z
        self.elements.append(
            Joint(
                name=f"J{jn} (rev)",
                joint_type="revolute",
                axis=np.array([0.0, 0.0, 1.0], dtype=float),
                q=0.0,
                q_min=-math.pi,
                q_max=math.pi,
                T_mount=np.eye(4, dtype=float),
            )
        )

    def add_prismatic_joint(self):
        # Number joints so they look nice in the UI list
        jn = sum(isinstance(e, Joint) for e in self.elements) + 1

        # Append a default prismatic joint along +Z
        self.elements.append(
            Joint(
                name=f"J{jn} (pris)",
                joint_type="prismatic",
                axis=np.array([0.0, 0.0, 1.0], dtype=float),
                q=0.0,
                q_min=-0.3,
                q_max=0.3,
                T_mount=np.eye(4, dtype=float),
            )
        )

    def add_link(self, dx=0.2, dy=0.0, dz=0.0):
        # Number links so they look nice in the UI list
        ln = sum(isinstance(e, Link) for e in self.elements) + 1

        # Create a pure translation transform for the link
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.array([dx, dy, dz], dtype=float)

        # Append the link element
        self.elements.append(Link(name=f"L{ln} (link)", T=T))

    def remove(self, idx: int):
        # Remove element by list index if valid
        if 0 <= idx < len(self.elements):
            self.elements.pop(idx)

    # ---- forward kinematics ----

    def fk_all(self) -> List[np.ndarray]:
        """
        Returns global frames: T0, T1, ..., Tn
        where n = len(elements).

        - T0 is identity (world/base frame)
        - Each element advances one frame:
              T_{k+1} = T_k @ element.local_T()
        """
        Ts = [np.eye(4, dtype=float)]  # T0
        T = np.eye(4, dtype=float)  # running product

        for e in self.elements:
            T = T @ e.local_T()  # apply element
            Ts.append(T.copy())  # store a snapshot of the global transform

        return Ts
