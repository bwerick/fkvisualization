#!/usr/bin/env python3
"""
Robot Frames Sandbox (Joints + Links + EE + Hybrid + Deg/Rad)
Python 3.11 compatible (dataclass default_factory fixes, safe QMatrix4x4 builder)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ============================
# Math / SE(3) utilities
# ============================


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = _norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=float,
    )


def rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    a = unit(axis)
    K = skew(a)
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.eye(3, dtype=float) * c + (1 - c) * np.outer(a, a) + s * K
    return R


def T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def T_trans(p: np.ndarray) -> np.ndarray:
    return T_from_Rp(np.eye(3, dtype=float), p)


def T_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    return T_from_Rp(rot_axis_angle(axis, theta), np.zeros(3, dtype=float))


def se3_valid(T: np.ndarray, tol: float = 1e-4) -> bool:
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=float), atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    return True


def rotation_matrix_from_z_to_vec(v: np.ndarray) -> np.ndarray:
    """
    Return R such that R * [0,0,1] aligns with unit(v).
    """
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    t = unit(v)
    c = float(np.dot(z, t))

    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-10:
        # 180 deg: rotate around X (or any axis orthogonal to z)
        return rot_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), math.pi)

    axis = unit(np.cross(z, t))
    angle = math.acos(max(-1.0, min(1.0, c)))
    return rot_axis_angle(axis, angle)


# ============================
# Qt transform helper (safe)
# ============================


def qmatrix_from_T(T: np.ndarray) -> QtGui.QMatrix4x4:
    """
    Safe builder for PySide6 QMatrix4x4 from numpy 4x4.
    Avoids item assignment (unsupported).
    """
    M = QtGui.QMatrix4x4()
    # QMatrix4x4 setRow expects QVector4D
    for r in range(4):
        M.setRow(
            r,
            QtGui.QVector4D(
                float(T[r, 0]),
                float(T[r, 1]),
                float(T[r, 2]),
                float(T[r, 3]),
            ),
        )
    return M


# ============================
# Mesh builders
# ============================


def meshdata_box(size: Tuple[float, float, float]) -> gl.MeshData:
    """
    Build a box centered at origin with extents size=(sx,sy,sz).
    Oriented along x,y,z.
    """
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    # 8 vertices
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )

    # 12 triangles (2 per face)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # -y
            [1, 5, 6],
            [1, 6, 2],  # +x
            [2, 6, 7],
            [2, 7, 3],  # +y
            [3, 7, 4],
            [3, 4, 0],  # -x
        ],
        dtype=int,
    )

    return gl.MeshData(vertexes=verts, faces=faces)


def meshdata_cylinder(radius: float, length: float, segments: int = 24) -> gl.MeshData:
    """
    Cylinder centered at origin, axis along +Z, length 'length'.
    """
    r = float(radius)
    L = float(length)
    hz = L / 2.0

    # circle points
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    circle = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

    # vertices: bottom ring + top ring + centers
    bottom = np.column_stack([circle, np.full((segments,), -hz)])
    top = np.column_stack([circle, np.full((segments,), +hz)])
    v_center_bottom = np.array([[0.0, 0.0, -hz]])
    v_center_top = np.array([[0.0, 0.0, +hz]])

    verts = np.vstack([bottom, top, v_center_bottom, v_center_top])
    idx_center_bottom = 2 * segments
    idx_center_top = 2 * segments + 1

    faces = []

    # side faces
    for i in range(segments):
        j = (i + 1) % segments
        # quad split into 2 tris
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])

    # bottom cap
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([idx_center_bottom, j, i])

    # top cap
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([idx_center_top, segments + i, segments + j])

    return gl.MeshData(vertexes=verts, faces=np.array(faces, dtype=int))


# ============================
# Model
# ============================


class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass
class Element:
    name: str

    def local_T(self) -> np.ndarray:
        return np.eye(4, dtype=float)


@dataclass
class Joint(Element):
    jtype: JointType = JointType.REVOLUTE
    axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    q: float = 0.0  # radians for revolute, meters for prismatic (internal)
    # fixed mount alignment before joint motion
    T_mount: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))

    def local_T(self) -> np.ndarray:
        a = unit(self.axis)
        if self.jtype == JointType.REVOLUTE:
            return self.T_mount @ T_rot(a, self.q)
        else:
            return self.T_mount @ T_trans(a * float(self.q))


@dataclass
class Link(Element):
    dx: float = 0.2
    dy: float = 0.0
    dz: float = 0.0

    def local_T(self) -> np.ndarray:
        return T_trans(np.array([self.dx, self.dy, self.dz], dtype=float))


class EndEffectorType(str, Enum):
    NONE = "None"
    FLANGE = "Flange"
    SUCTION = "Suction"
    CLAW = "Claw"


@dataclass
class RobotModel:
    elements: List[Element] = field(default_factory=list)
    ee_type: EndEffectorType = EndEffectorType.NONE

    def add_revolute(self, axis=(0, 0, 1)):
        n = sum(isinstance(e, Joint) for e in self.elements) + 1
        self.elements.append(
            Joint(
                name=f"J{n} (rev)",
                jtype=JointType.REVOLUTE,
                axis=np.array(axis, dtype=float),
            )
        )

    def add_prismatic(self, axis=(0, 0, 1)):
        n = sum(isinstance(e, Joint) for e in self.elements) + 1
        self.elements.append(
            Joint(
                name=f"J{n} (pris)",
                jtype=JointType.PRISMATIC,
                axis=np.array(axis, dtype=float),
            )
        )

    def add_link(self, dx=0.2, dy=0.0, dz=0.0):
        n = sum(isinstance(e, Link) for e in self.elements) + 1
        self.elements.append(Link(name=f"L{n} (link)", dx=dx, dy=dy, dz=dz))

    def remove_at(self, idx: int):
        if 0 <= idx < len(self.elements):
            self.elements.pop(idx)

    def fk_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns:
          - Ts_global_frames: list of frame poses (frame 0, frame 1, ... frame N)
          - Ts_step: list of step transforms (0H1,1H2,...), same length as elements
        Convention: each element advances frame i -> i+1
        """
        Ts_frames = [np.eye(4, dtype=float)]
        Ts_step = []
        T = np.eye(4, dtype=float)
        for e in self.elements:
            A = e.local_T()
            Ts_step.append(A)
            T = T @ A
            Ts_frames.append(T.copy())
        return Ts_frames, Ts_step


# ============================
# Transform viewer window
# ============================


class TransformViewMode(str, Enum):
    NUMERIC = "Numeric"
    HYBRID = "Hybrid"


class FrameTransformsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame Transforms")

        self._mode = TransformViewMode.NUMERIC

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # mode buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_numeric = QtWidgets.QPushButton("Numeric")
        self.btn_hybrid = QtWidgets.QPushButton("Hybrid")
        self.btn_numeric.setCheckable(True)
        self.btn_hybrid.setCheckable(True)
        self.btn_numeric.setChecked(True)
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.addButton(self.btn_numeric)
        self.btn_group.addButton(self.btn_hybrid)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_numeric)
        btn_row.addWidget(self.btn_hybrid)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        self.inner = QtWidgets.QWidget()
        self.scroll.setWidget(self.inner)
        self.inner_layout = QtWidgets.QVBoxLayout(self.inner)
        self.inner_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.btn_numeric.clicked.connect(
            lambda: self.set_mode(TransformViewMode.NUMERIC)
        )
        self.btn_hybrid.clicked.connect(lambda: self.set_mode(TransformViewMode.HYBRID))

        self._blocks: List[QtWidgets.QWidget] = []

    def set_mode(self, mode: TransformViewMode):
        self._mode = mode
        self.btn_numeric.setChecked(mode == TransformViewMode.NUMERIC)
        self.btn_hybrid.setChecked(mode == TransformViewMode.HYBRID)

    def clear_blocks(self):
        for w in self._blocks:
            w.setParent(None)
        self._blocks.clear()

    def update_from_robot(self, robot: RobotModel, angle_units_deg: bool):
        self.clear_blocks()
        Ts_frames, Ts_step = robot.fk_frames()

        # For hybrid we show "symbol" row + numeric row
        # (We won't do true sympy; we'll show c/s tokens as strings.)

        for i, e in enumerate(robot.elements):
            title = f"{i}H{i+1}  ({e.name})"
            header = QtWidgets.QLabel(title)
            header.setStyleSheet("font-size: 18px; font-weight: 600; padding: 8px 0;")
            self.inner_layout.addWidget(header)

            # info line
            info = QtWidgets.QLabel()
            info.setStyleSheet("font-family: monospace; padding-bottom: 6px;")
            if isinstance(e, Joint):
                axis = unit(e.axis)
                axis_name = (
                    "X"
                    if np.allclose(axis, [1, 0, 0])
                    else (
                        "Y"
                        if np.allclose(axis, [0, 1, 0])
                        else (
                            "Z"
                            if np.allclose(axis, [0, 0, 1])
                            else f"[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]"
                        )
                    )
                )
                if e.jtype == JointType.REVOLUTE:
                    th = e.q
                    th_deg = math.degrees(th)
                    shown = th_deg if angle_units_deg else th
                    unit_str = "deg" if angle_units_deg else "rad"
                    c = math.cos(th)
                    s = math.sin(th)
                    info.setText(
                        f"JOINT revolute | axis={axis_name} | θ={shown:.5f} {unit_str} | c={c:.6f} s={s:.6f}"
                    )
                else:
                    q = float(e.q)
                    info.setText(f"JOINT prismatic | axis={axis_name} | q={q:.5f} m")
            else:
                link = e  # type: ignore
                info.setText(
                    f"LINK fixed | translation=[{link.dx:.3f},{link.dy:.3f},{link.dz:.3f}] m"
                )
            self.inner_layout.addWidget(info)

            A = Ts_step[i]

            table = self._matrix_table(A, e, angle_units_deg)
            self.inner_layout.addWidget(table)

            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setStyleSheet("color: #444;")
            self.inner_layout.addWidget(sep)

        self.inner_layout.addStretch(1)

    def _matrix_table(
        self, A: np.ndarray, e: Element, angle_units_deg: bool
    ) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container)

        if self._mode == TransformViewMode.NUMERIC:
            tbl = self._make_tbl(A, None)
            v.addWidget(tbl)
        else:
            # Hybrid: top row "symbolic-ish", bottom numeric
            sym = self._symbolic_matrix(A, e, angle_units_deg)
            tbl = self._make_tbl(A, sym)
            v.addWidget(tbl)

        self._blocks.append(container)
        return container

    def _make_tbl(
        self, A: np.ndarray, sym: Optional[List[List[str]]]
    ) -> QtWidgets.QTableWidget:
        rows = 4 if sym is None else 8
        tbl = QtWidgets.QTableWidget(rows, 4)
        tbl.setStyleSheet("font-family: monospace; font-size: 14px;")
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setVisible(False)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        tbl.setShowGrid(True)

        def set_cell(r, c, text):
            it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            tbl.setItem(r, c, it)

        if sym is not None:
            for r in range(4):
                for c in range(4):
                    set_cell(r, c, sym[r][c])
            for r in range(4):
                for c in range(4):
                    set_cell(r + 4, c, f"{A[r, c]: .6f}")
        else:
            for r in range(4):
                for c in range(4):
                    set_cell(r, c, f"{A[r, c]: .6f}")

        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()
        tbl.setMinimumHeight(180 if sym is None else 300)
        return tbl

    def _symbolic_matrix(
        self, A: np.ndarray, e: Element, angle_units_deg: bool
    ) -> List[List[str]]:
        # Just a friendly symbolic-ish presentation
        if isinstance(e, Joint) and e.jtype == JointType.REVOLUTE:
            axis = unit(e.axis)
            # detect axis
            if np.allclose(axis, [0, 0, 1]):
                return [
                    ["c", "-s", "0", "0"],
                    ["s", "c", "0", "0"],
                    ["0", "0", "1", "0"],
                    ["0", "0", "0", "1"],
                ]
            if np.allclose(axis, [1, 0, 0]):
                return [
                    ["1", "0", "0", "0"],
                    ["0", "c", "-s", "0"],
                    ["0", "s", "c", "0"],
                    ["0", "0", "0", "1"],
                ]
            if np.allclose(axis, [0, 1, 0]):
                return [
                    ["c", "0", "s", "0"],
                    ["0", "1", "0", "0"],
                    ["-s", "0", "c", "0"],
                    ["0", "0", "0", "1"],
                ]
        if isinstance(e, Joint) and e.jtype == JointType.PRISMATIC:
            return [
                ["1", "0", "0", "q*ax"],
                ["0", "1", "0", "q*ay"],
                ["0", "0", "1", "q*az"],
                ["0", "0", "0", "1"],
            ]
        if isinstance(e, Link):
            return [
                ["1", "0", "0", "dx"],
                ["0", "1", "0", "dy"],
                ["0", "0", "1", "dz"],
                ["0", "0", "0", "1"],
            ]
        # fallback
        return [[f"{A[r,c]:.3f}" for c in range(4)] for r in range(4)]


# ============================
# 3D View
# ============================


class GLRobotView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setBackgroundColor((0, 0, 0, 255))
        self.opts["distance"] = 2.5
        self.opts["elevation"] = 20
        self.opts["azimuth"] = 35

        self._items: List[object] = []

        grid = gl.GLGridItem()
        grid.setSize(5, 5)
        grid.setSpacing(0.2, 0.2)
        self.addItem(grid)
        self._items.append(grid)

    def clear_dynamic(self):
        # remove all except first grid
        for it in self._items[1:]:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._items = self._items[:1]

    def draw_robot(
        self,
        Ts_frames: List[np.ndarray],
        elements: List[Element],
        ee_type: EndEffectorType,
    ):
        self.clear_dynamic()

        # Origins
        origins = [T[:3, 3].copy() for T in Ts_frames]

        # ---- Draw LINKS between origins for Link elements only ----
        link_w = 0.04
        link_h = 0.02
        link_color = (0.65, 0.65, 0.65, 1.0)

        # Each element i maps frame i -> i+1
        for i, e in enumerate(elements):
            if not isinstance(e, Link):
                continue
            p0 = origins[i]
            p1 = origins[i + 1]
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue

            md = meshdata_box((link_w, link_h, L))
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=link_color,
            )

            R = rotation_matrix_from_z_to_vec(d)
            mid = 0.5 * (p0 + p1)
            Tm = np.eye(4, dtype=float)
            Tm[:3, :3] = R
            Tm[:3, 3] = mid
            mesh.setTransform(qmatrix_from_T(Tm))

            self.addItem(mesh)
            self._items.append(mesh)

        # ---- Draw JOINTS at frame origins (always visible) ----
        rev_radius = 0.02
        rev_len = 0.08
        pris_base = 0.08  # housing length
        pris_w, pris_h = 0.05, 0.035

        for i, e in enumerate(elements):
            if not isinstance(e, Joint):
                continue

            p0 = origins[i]

            # compute joint axis in world using base (frame i) and T_mount
            T_base = Ts_frames[i] @ e.T_mount
            axis_local = unit(e.axis)
            axis_w = T_base[:3, :3] @ axis_local

            R_axis = rotation_matrix_from_z_to_vec(axis_w)

            if e.jtype == JointType.REVOLUTE:
                md = meshdata_cylinder(rev_radius, rev_len, segments=26)
                mesh = gl.GLMeshItem(
                    meshdata=md,
                    smooth=False,
                    drawFaces=True,
                    drawEdges=True,
                    color=(0.9, 0.2, 0.2, 1.0),
                )
                Tm = np.eye(4, dtype=float)
                Tm[:3, :3] = R_axis
                Tm[:3, 3] = p0
                mesh.setTransform(qmatrix_from_T(Tm))
                self.addItem(mesh)
                self._items.append(mesh)

            else:
                # housing
                housing_md = meshdata_box((pris_w, pris_h, pris_base))
                housing = gl.GLMeshItem(
                    meshdata=housing_md,
                    smooth=False,
                    drawFaces=True,
                    drawEdges=True,
                    color=(0.2, 0.45, 0.9, 1.0),
                )
                Th = np.eye(4, dtype=float)
                Th[:3, :3] = R_axis
                Th[:3, 3] = p0
                housing.setTransform(qmatrix_from_T(Th))
                self.addItem(housing)
                self._items.append(housing)

                # slider extends only in +axis direction visually
                ext = max(0.0, float(e.q))
                if ext > 1e-6:
                    slider_md = meshdata_box((pris_w * 0.75, pris_h * 0.75, ext))
                    slider = gl.GLMeshItem(
                        meshdata=slider_md,
                        smooth=False,
                        drawFaces=True,
                        drawEdges=True,
                        color=(0.35, 0.55, 0.90, 1.0),
                    )

                    # place slider in front of housing along +Z of joint-aligned frame
                    offset = pris_base / 2.0 + ext / 2.0
                    Ts = np.eye(4, dtype=float)
                    Ts[:3, :3] = R_axis
                    Ts[:3, 3] = p0 + R_axis @ np.array([0.0, 0.0, offset], dtype=float)
                    slider.setTransform(qmatrix_from_T(Ts))

                    self.addItem(slider)
                    self._items.append(slider)

        # ---- Draw frame triads ----
        axis_len = 0.06
        for T in Ts_frames:
            p = T[:3, 3]
            R = T[:3, :3]
            x = p + R[:, 0] * axis_len
            y = p + R[:, 1] * axis_len
            z = p + R[:, 2] * axis_len

            self._add_line(p, x, (1.0, 0.0, 0.0, 1.0))
            self._add_line(p, y, (0.0, 1.0, 0.0, 1.0))
            self._add_line(p, z, (0.3, 0.3, 1.0, 1.0))

        # ---- End effector (visual only) ----
        if ee_type != EndEffectorType.NONE and len(Ts_frames) > 0:
            Tee = Ts_frames[-1]
            self._draw_ee(Tee, ee_type)

    def _add_line(self, p0, p1, color):
        pts = np.vstack([p0, p1]).astype(float)
        item = gl.GLLinePlotItem(pos=pts, color=color, width=2.0, antialias=True)
        self.addItem(item)
        self._items.append(item)

    def _draw_ee(self, Tee: np.ndarray, ee_type: EndEffectorType):
        p = Tee[:3, 3]
        R = Tee[:3, :3]

        if ee_type == EndEffectorType.FLANGE:
            # ring-ish: approximate with a short fat cylinder
            md = meshdata_cylinder(0.05, 0.015, segments=32)
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.85, 0.85, 0.85, 1.0),
            )
            Tm = np.eye(4, dtype=float)
            Tm[:3, :3] = R
            Tm[:3, 3] = p
            mesh.setTransform(qmatrix_from_T(Tm))
            self.addItem(mesh)
            self._items.append(mesh)

        elif ee_type == EndEffectorType.SUCTION:
            # disk + short stem
            disk = gl.GLMeshItem(
                meshdata=meshdata_cylinder(0.05, 0.01, 32),
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.2, 0.2, 0.2, 1.0),
            )
            stem = gl.GLMeshItem(
                meshdata=meshdata_cylinder(0.01, 0.06, 20),
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.8, 0.8, 0.8, 1.0),
            )

            Tdisk = np.eye(4, dtype=float)
            Tdisk[:3, :3] = R
            Tdisk[:3, 3] = p
            disk.setTransform(qmatrix_from_T(Tdisk))

            Tstem = np.eye(4, dtype=float)
            Tstem[:3, :3] = R
            Tstem[:3, 3] = p + R @ np.array([0, 0, 0.04], dtype=float)
            stem.setTransform(qmatrix_from_T(Tstem))

            self.addItem(disk)
            self._items.append(disk)
            self.addItem(stem)
            self._items.append(stem)

        elif ee_type == EndEffectorType.CLAW:
            # two fingers
            palm = gl.GLMeshItem(
                meshdata=meshdata_box((0.08, 0.03, 0.03)),
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.9, 0.4, 0.1, 1.0),
            )
            finger = meshdata_box((0.015, 0.015, 0.08))
            f1 = gl.GLMeshItem(
                meshdata=finger,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.95, 0.8, 0.1, 1.0),
            )
            f2 = gl.GLMeshItem(
                meshdata=finger,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.95, 0.8, 0.1, 1.0),
            )

            Tp = np.eye(4, dtype=float)
            Tp[:3, :3] = R
            Tp[:3, 3] = p
            palm.setTransform(qmatrix_from_T(Tp))

            Tf1 = np.eye(4, dtype=float)
            Tf1[:3, :3] = R
            Tf1[:3, 3] = p + R @ np.array([0.025, 0.0, 0.06], dtype=float)
            f1.setTransform(qmatrix_from_T(Tf1))

            Tf2 = np.eye(4, dtype=float)
            Tf2[:3, :3] = R
            Tf2[:3, 3] = p + R @ np.array([-0.025, 0.0, 0.06], dtype=float)
            f2.setTransform(qmatrix_from_T(Tf2))

            self.addItem(palm)
            self._items.append(palm)
            self.addItem(f1)
            self._items.append(f1)
            self.addItem(f2)
            self._items.append(f2)


# ============================
# Main UI
# ============================


class AngleUnits(str, Enum):
    RADIANS = "Radians"
    DEGREES = "Degrees"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Robot Frames Sandbox (Joints + Links + EE + Hybrid + Deg/Rad)"
        )
        self.resize(1400, 800)

        self.robot = RobotModel()
        self.angle_units = AngleUnits.RADIANS

        # start demo
        self.robot.add_revolute(axis=(0, 0, 1))
        self.robot.add_link(dx=0.3, dy=0, dz=0)
        self.robot.add_revolute(axis=(1, 0, 0))
        self.robot.add_link(dx=0.3, dy=0, dz=0)
        self.robot.add_prismatic(axis=(0, 0, 1))
        self.robot.add_link(dx=0.3, dy=0, dz=0)

        self.transforms_win = FrameTransformsWindow()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left panel
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(260)
        left.addWidget(QtWidgets.QLabel("Elements (Joint / Link)"))
        left.addWidget(self.list, 1)

        btns = QtWidgets.QVBoxLayout()
        self.btn_add_rev = QtWidgets.QPushButton("Add Revolute Joint")
        self.btn_add_pris = QtWidgets.QPushButton("Add Prismatic Joint")
        self.btn_add_link = QtWidgets.QPushButton("Add Link (Fixed)")
        self.btn_remove = QtWidgets.QPushButton("Remove Selected")
        for b in (
            self.btn_add_rev,
            self.btn_add_pris,
            self.btn_add_link,
            self.btn_remove,
        ):
            btns.addWidget(b)
        left.addLayout(btns)

        left.addSpacing(12)
        left.addWidget(QtWidgets.QLabel("End Effector (visual)"))
        self.ee_combo = QtWidgets.QComboBox()
        for t in EndEffectorType:
            self.ee_combo.addItem(t.value)
        left.addWidget(self.ee_combo)
        self.btn_set_ee = QtWidgets.QPushButton("Set End Effector")
        self.btn_clear_ee = QtWidgets.QPushButton("Clear End Effector")
        left.addWidget(self.btn_set_ee)
        left.addWidget(self.btn_clear_ee)
        left.addStretch(1)

        # Center view
        self.view = GLRobotView()
        root.addWidget(self.view, 1)

        # Right inspector
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 0)

        right.addWidget(QtWidgets.QLabel("Inspector"))

        # angle units
        row_units = QtWidgets.QHBoxLayout()
        row_units.addWidget(QtWidgets.QLabel("Angle units:"))
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.addItem(AngleUnits.RADIANS.value)
        self.units_combo.addItem(AngleUnits.DEGREES.value)
        row_units.addWidget(self.units_combo, 1)
        right.addLayout(row_units)

        # selected label
        self.lbl_selected = QtWidgets.QLabel("Selected: (none)")
        self.lbl_selected.setStyleSheet("padding: 6px 0; font-weight: 600;")
        right.addWidget(self.lbl_selected)

        self.lbl_joint_info = QtWidgets.QLabel("")
        self.lbl_joint_info.setWordWrap(True)
        self.lbl_joint_info.setStyleSheet("font-family: monospace;")
        right.addWidget(self.lbl_joint_info)

        # axis preset + spinboxes
        self.axis_preset = QtWidgets.QComboBox()
        self.axis_preset.addItems(["Custom", "X", "Y", "Z"])
        right.addWidget(QtWidgets.QLabel("Axis preset:"))
        right.addWidget(self.axis_preset)

        row_axis = QtWidgets.QHBoxLayout()
        self.spin_ax = QtWidgets.QDoubleSpinBox()
        self.spin_ax.setRange(-1, 1)
        self.spin_ax.setDecimals(6)
        self.spin_ay = QtWidgets.QDoubleSpinBox()
        self.spin_ay.setRange(-1, 1)
        self.spin_ay.setDecimals(6)
        self.spin_az = QtWidgets.QDoubleSpinBox()
        self.spin_az.setRange(-1, 1)
        self.spin_az.setDecimals(6)
        row_axis.addWidget(QtWidgets.QLabel("ax"))
        row_axis.addWidget(self.spin_ax)
        row_axis.addWidget(QtWidgets.QLabel("ay"))
        row_axis.addWidget(self.spin_ay)
        row_axis.addWidget(QtWidgets.QLabel("az"))
        row_axis.addWidget(self.spin_az)
        right.addLayout(row_axis)

        # q slider
        self.q_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.q_slider.setRange(-1800, 1800)
        right.addWidget(self.q_slider)

        self.lbl_q = QtWidgets.QLabel("q = 0")
        self.lbl_q.setStyleSheet("font-family: monospace;")
        right.addWidget(self.lbl_q)

        # mount matrix table
        right.addWidget(QtWidgets.QLabel("Joint mount transform (fixed alignment)"))
        self.tbl_mount = QtWidgets.QTableWidget(4, 4)
        self.tbl_mount.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.tbl_mount.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.tbl_mount.horizontalHeader().setVisible(False)
        self.tbl_mount.verticalHeader().setVisible(False)
        self.tbl_mount.setStyleSheet("font-family: monospace; font-size: 13px;")
        right.addWidget(self.tbl_mount, 1)

        # SE3 validity indicator
        self.lbl_se3 = QtWidgets.QLabel("SE(3) looks valid")
        self.lbl_se3.setStyleSheet("color: #5fd35f; font-weight: 600; padding: 8px 0;")
        right.addWidget(self.lbl_se3)

        # transforms window button
        self.btn_open_tf = QtWidgets.QPushButton("Open Frame Transforms Window")
        right.addWidget(self.btn_open_tf)

        right.addStretch(1)

        # connections
        self.btn_add_rev.clicked.connect(self.on_add_rev)
        self.btn_add_pris.clicked.connect(self.on_add_pris)
        self.btn_add_link.clicked.connect(self.on_add_link)
        self.btn_remove.clicked.connect(self.on_remove)

        self.btn_set_ee.clicked.connect(self.on_set_ee)
        self.btn_clear_ee.clicked.connect(self.on_clear_ee)

        self.list.currentRowChanged.connect(self.on_select)
        self.axis_preset.currentIndexChanged.connect(self.on_axis_preset)
        self.spin_ax.valueChanged.connect(self.on_axis_changed)
        self.spin_ay.valueChanged.connect(self.on_axis_changed)
        self.spin_az.valueChanged.connect(self.on_axis_changed)

        self.q_slider.valueChanged.connect(self.on_q_changed)
        self.units_combo.currentIndexChanged.connect(self.on_units_changed)

        self.btn_open_tf.clicked.connect(self.on_open_transforms)

        # init
        self.refresh_list()
        self.list.setCurrentRow(0)
        self.update_view()

    def refresh_list(self):
        self.list.clear()
        for e in self.robot.elements:
            self.list.addItem(e.name)

    def selected_element(self) -> Optional[Element]:
        idx = self.list.currentRow()
        if 0 <= idx < len(self.robot.elements):
            return self.robot.elements[idx]
        return None

    def update_view(self):
        Ts_frames, _ = self.robot.fk_frames()
        self.view.draw_robot(Ts_frames, self.robot.elements, self.robot.ee_type)

        # update transforms window (if open)
        self.transforms_win.update_from_robot(
            self.robot, angle_units_deg=(self.angle_units == AngleUnits.DEGREES)
        )

    def update_inspector(self):
        e = self.selected_element()
        if e is None:
            self.lbl_selected.setText("Selected: (none)")
            self.lbl_joint_info.setText("")
            return

        if isinstance(e, Joint):
            self.lbl_selected.setText("Selected: JOINT")
            axis = unit(e.axis)
            self.spin_ax.blockSignals(True)
            self.spin_ay.blockSignals(True)
            self.spin_az.blockSignals(True)
            self.spin_ax.setValue(float(axis[0]))
            self.spin_ay.setValue(float(axis[1]))
            self.spin_az.setValue(float(axis[2]))
            self.spin_ax.blockSignals(False)
            self.spin_ay.blockSignals(False)
            self.spin_az.blockSignals(False)

            axis_label = f"[{axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f}]"
            self.lbl_joint_info.setText(
                f"{e.name} | {e.jtype.value} | axis={axis_label}"
            )

            # slider mapping
            if e.jtype == JointType.REVOLUTE:
                # internal radians; slider shows deg or rad
                if self.angle_units == AngleUnits.DEGREES:
                    val = int(round(math.degrees(e.q) * 10))
                    self.lbl_q.setText(f"q = {math.degrees(e.q):.5f} deg")
                else:
                    val = int(round(e.q * 1000))
                    self.lbl_q.setText(f"q = {e.q:.5f} rad")
                self.q_slider.blockSignals(True)
                (
                    self.q_slider.setRange(-1800, 1800)
                    if self.angle_units == AngleUnits.DEGREES
                    else self.q_slider.setRange(-4000, 4000)
                )
                self.q_slider.setValue(val)
                self.q_slider.blockSignals(False)
            else:
                # prismatic in meters
                self.q_slider.blockSignals(True)
                self.q_slider.setRange(-500, 500)  # +/-0.5m
                self.q_slider.setValue(int(round(e.q * 1000)))
                self.q_slider.blockSignals(False)
                self.lbl_q.setText(f"q = {e.q:.5f} m")

            # mount matrix
            Tm = e.T_mount
            self._fill_table(self.tbl_mount, Tm)

            ok = se3_valid(Tm)
            self.lbl_se3.setText("SE(3) looks valid" if ok else "SE(3) invalid")
            self.lbl_se3.setStyleSheet(
                "color: #5fd35f; font-weight: 600; padding: 8px 0;"
                if ok
                else "color: #ff5555; font-weight: 600; padding: 8px 0;"
            )
        else:
            self.lbl_selected.setText("Selected: LINK")
            link: Link = e  # type: ignore
            self.lbl_joint_info.setText(
                f"{link.name} | fixed | dx,dy,dz=[{link.dx:.3f},{link.dy:.3f},{link.dz:.3f}]"
            )
            self.lbl_q.setText("q = (n/a)")
            self._fill_table(self.tbl_mount, np.eye(4, dtype=float))
            self.lbl_se3.setText("SE(3) looks valid")
            self.lbl_se3.setStyleSheet(
                "color: #5fd35f; font-weight: 600; padding: 8px 0;"
            )

    def _fill_table(self, tbl: QtWidgets.QTableWidget, T: np.ndarray):
        for r in range(4):
            for c in range(4):
                it = QtWidgets.QTableWidgetItem(f"{T[r, c]: .6f}")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                tbl.setItem(r, c, it)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    # ---- callbacks ----

    def on_add_rev(self):
        self.robot.add_revolute(axis=(0, 0, 1))
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def on_add_pris(self):
        self.robot.add_prismatic(axis=(0, 0, 1))
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def on_add_link(self):
        self.robot.add_link(dx=0.3, dy=0.0, dz=0.0)
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def on_remove(self):
        idx = self.list.currentRow()
        if idx < 0:
            return
        self.robot.remove_at(idx)
        self.refresh_list()
        self.list.setCurrentRow(min(idx, len(self.robot.elements) - 1))
        self.update_view()

    def on_set_ee(self):
        txt = self.ee_combo.currentText()
        self.robot.ee_type = (
            EndEffectorType(txt)
            if txt in [t.value for t in EndEffectorType]
            else EndEffectorType.NONE
        )
        self.update_view()

    def on_clear_ee(self):
        self.robot.ee_type = EndEffectorType.NONE
        self.ee_combo.setCurrentText(EndEffectorType.NONE.value)
        self.update_view()

    def on_select(self, idx: int):
        self.update_inspector()

    def on_axis_preset(self, _):
        e = self.selected_element()
        if not isinstance(e, Joint):
            return
        preset = self.axis_preset.currentText()
        if preset == "X":
            e.axis = np.array([1.0, 0.0, 0.0], dtype=float)
        elif preset == "Y":
            e.axis = np.array([0.0, 1.0, 0.0], dtype=float)
        elif preset == "Z":
            e.axis = np.array([0.0, 0.0, 1.0], dtype=float)
        # "Custom" does nothing
        self.update_inspector()
        self.update_view()

    def on_axis_changed(self, _):
        e = self.selected_element()
        if not isinstance(e, Joint):
            return
        a = np.array(
            [self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value()],
            dtype=float,
        )
        e.axis = unit(a)
        self.axis_preset.blockSignals(True)
        self.axis_preset.setCurrentText("Custom")
        self.axis_preset.blockSignals(False)
        self.update_inspector()
        self.update_view()

    def on_q_changed(self, v: int):
        e = self.selected_element()
        if not isinstance(e, Joint):
            return
        if e.jtype == JointType.REVOLUTE:
            if self.angle_units == AngleUnits.DEGREES:
                deg = v / 10.0
                e.q = math.radians(deg)
            else:
                e.q = v / 1000.0
        else:
            e.q = v / 1000.0  # meters
        self.update_inspector()
        self.update_view()

    def on_units_changed(self, _):
        self.angle_units = AngleUnits(self.units_combo.currentText())
        self.update_inspector()
        self.update_view()

    def on_open_transforms(self):
        self.transforms_win.show()
        self.transforms_win.raise_()


# ============================
# Main
# ============================


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
