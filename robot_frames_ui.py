import os

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

import sys
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# =========================
# SE(3) utilities
# =========================


def hat(w: np.ndarray) -> np.ndarray:
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
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    a = axis / n
    K = hat(a)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def is_valid_transform(T: np.ndarray, tol: float = 5e-3) -> bool:
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    return True


def origin_of(T: np.ndarray) -> np.ndarray:
    return T[:3, 3].copy()


def axes_of(T: np.ndarray) -> np.ndarray:
    return T[:3, :3].copy()


# =========================
# Mesh / transform helpers
# =========================


def rotation_matrix_from_z_to_vec(v: np.ndarray) -> np.ndarray:
    """
    Return R such that R*[0,0,1] aligns with v_hat.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3, dtype=float)

    vhat = v / n
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    c = float(np.clip(np.dot(z, vhat), -1.0, 1.0))

    if abs(c - 1.0) < 1e-10:
        return np.eye(3, dtype=float)

    if abs(c + 1.0) < 1e-10:
        # 180° rotation about X (any axis orthogonal to z is fine)
        return rot_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi)

    axis = np.cross(z, vhat)
    s = np.linalg.norm(axis)
    axis = axis / (s + 1e-12)
    theta = math.atan2(s, c)
    return rot_axis_angle(axis, theta)


def qmatrix_from_T(T: np.ndarray) -> QtGui.QMatrix4x4:
    """
    Construct QMatrix4x4 from a 4x4 numpy array.
    (PySide6 binding does not allow M[r,c] assignment.)
    """
    T = np.asarray(T, dtype=float)
    return QtGui.QMatrix4x4(
        float(T[0, 0]),
        float(T[0, 1]),
        float(T[0, 2]),
        float(T[0, 3]),
        float(T[1, 0]),
        float(T[1, 1]),
        float(T[1, 2]),
        float(T[1, 3]),
        float(T[2, 0]),
        float(T[2, 1]),
        float(T[2, 2]),
        float(T[2, 3]),
        float(T[3, 0]),
        float(T[3, 1]),
        float(T[3, 2]),
        float(T[3, 3]),
    )


def meshdata_box(size=(1.0, 1.0, 1.0)) -> gl.MeshData:
    """
    Create a box aligned to axes with z from 0..sz (so its 'base' is at z=0),
    matching how we place cylinders (base at p0 then rotate along link direction).
    size = (sx, sy, sz)
    """
    sx, sy, sz = map(float, size)
    x0, x1 = -sx / 2, sx / 2
    y0, y1 = -sy / 2, sy / 2
    z0, z1 = 0.0, sz

    verts = np.array(
        [
            [x0, y0, z0],  # 0
            [x1, y0, z0],  # 1
            [x1, y1, z0],  # 2
            [x0, y1, z0],  # 3
            [x0, y0, z1],  # 4
            [x1, y0, z1],  # 5
            [x1, y1, z1],  # 6
            [x0, y1, z1],  # 7
        ],
        dtype=float,
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 5, 1],
            [0, 4, 5],  # front
            [3, 2, 6],
            [3, 6, 7],  # back
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=int,
    )

    return gl.MeshData(vertexes=verts, faces=faces)


def center_meshdata_z(md: gl.MeshData) -> gl.MeshData:
    """
    Centers any mesh along its local Z axis by shifting vertices so that
    z-range becomes symmetric around 0.
    Works across pyqtgraph versions (some primitives are centered, some are not).
    """
    v = md.vertexes().copy()
    zmin = float(v[:, 2].min())
    zmax = float(v[:, 2].max())
    zc = 0.5 * (zmin + zmax)
    v[:, 2] -= zc
    return gl.MeshData(vertexes=v, faces=md.faces())


# =========================
# Elements: Joint + Link + End Effector
# =========================


@dataclass
class Joint:
    name: str
    joint_type: str  # "revolute" or "prismatic"
    axis: np.ndarray  # joint-local axis
    q: float  # ALWAYS stored in radians (for revolute); meters for prismatic
    q_min: float
    q_max: float
    T_mount: np.ndarray  # fixed alignment before the motion (optional)

    def motion_T(self) -> np.ndarray:
        a = np.asarray(self.axis, dtype=float)
        if self.joint_type == "revolute":
            R = rot_axis_angle(a, self.q)
            return T_from_Rp(R, np.zeros(3))
        else:
            a = a / (np.linalg.norm(a) + 1e-12)
            p = a * self.q
            return T_from_Rp(np.eye(3), p)

    def local_T(self) -> np.ndarray:
        return self.T_mount @ self.motion_T()


@dataclass
class Link:
    name: str
    T: np.ndarray  # fixed transform

    def local_T(self) -> np.ndarray:
        return self.T


class EEType(str, Enum):
    FLANGE = "Flange"
    SUCTION = "Suction"
    CLAW = "Claw"


@dataclass
class EndEffector:
    ee_type: EEType = EEType.FLANGE
    T: np.ndarray = np.eye(4)  # offset from final frame to EE root


Element = Union[Joint, Link]


class RobotChain:
    def __init__(self):
        self.elements: List[Element] = []
        self.end_effector: Optional[EndEffector] = EndEffector(EEType.FLANGE, np.eye(4))

    def add_revolute_joint(self):
        jn = sum(isinstance(e, Joint) for e in self.elements) + 1
        self.elements.append(
            Joint(
                name=f"J{jn} (rev)",
                joint_type="revolute",
                axis=np.array([0, 0, 1], dtype=float),
                q=0.0,
                q_min=-math.pi,
                q_max=math.pi,
                T_mount=np.eye(4),
            )
        )

    def add_prismatic_joint(self):
        jn = sum(isinstance(e, Joint) for e in self.elements) + 1
        self.elements.append(
            Joint(
                name=f"J{jn} (pris)",
                joint_type="prismatic",
                axis=np.array([0, 0, 1], dtype=float),
                q=0.0,
                q_min=-0.3,
                q_max=0.3,
                T_mount=np.eye(4),
            )
        )

    def add_link(self, dx=0.2, dy=0.0, dz=0.0):
        ln = sum(isinstance(e, Link) for e in self.elements) + 1
        T = np.eye(4)
        T[:3, 3] = np.array([dx, dy, dz], dtype=float)
        self.elements.append(Link(name=f"L{ln} (link)", T=T))

    def remove(self, idx: int):
        if 0 <= idx < len(self.elements):
            self.elements.pop(idx)

    def fk_all(self) -> List[np.ndarray]:
        """
        Returns global frames: T0, T1, ..., Tn where n = len(elements).
        Each element advances one frame.
        """
        Ts = [np.eye(4)]
        T = np.eye(4)
        for e in self.elements:
            T = T @ e.local_T()
            Ts.append(T.copy())
        return Ts


# =========================
# UI widgets
# =========================


class MatrixEditor(QtWidgets.QTableWidget):
    matrixChanged = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__(4, 4)
        self._block = False
        self.itemChanged.connect(self._on_change)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.set_matrix(np.eye(4))

    def set_matrix(self, T: np.ndarray):
        self._block = True
        for r in range(4):
            for c in range(4):
                item = self.item(r, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.setItem(r, c, item)
                item.setText(f"{float(T[r, c]): .6f}")
        self._block = False

    def matrix(self) -> Optional[np.ndarray]:
        T = np.zeros((4, 4), dtype=float)
        for r in range(4):
            for c in range(4):
                item = self.item(r, c)
                if item is None:
                    return None
                txt = item.text().strip()
                try:
                    T[r, c] = float(txt)
                except ValueError:
                    return None
        return T

    def _on_change(self, _):
        if self._block:
            return
        T = self.matrix()
        if T is not None:
            self.matrixChanged.emit(T)


def _axis_basis_name(a: np.ndarray, tol: float = 1e-6) -> Optional[str]:
    a = np.asarray(a, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return None
    ah = a / n
    if np.allclose(ah, [1, 0, 0], atol=tol):
        return "X"
    if np.allclose(ah, [0, 1, 0], atol=tol):
        return "Y"
    if np.allclose(ah, [0, 0, 1], atol=tol):
        return "Z"
    # also allow negative basis as same axis (rotation direction is sign-sensitive,
    # but for learning display it's often okay to show it explicitly)
    if np.allclose(ah, [-1, 0, 0], atol=tol):
        return "-X"
    if np.allclose(ah, [0, -1, 0], atol=tol):
        return "-Y"
    if np.allclose(ah, [0, 0, -1], atol=tol):
        return "-Z"
    return None


class MatricesWindow(QtWidgets.QMainWindow):
    """
    Tabbed window:
      - Numeric: local transforms iH(i+1) and global transforms 0Hk (tables)
      - Hybrid: formulas + live values (monospace text)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Transforms")
        self.resize(660, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs)

        # -------- Numeric tab
        self.num_scroll = QtWidgets.QScrollArea()
        self.num_scroll.setWidgetResizable(True)
        self.tabs.addTab(self.num_scroll, "Numeric")

        self.num_inner = QtWidgets.QWidget()
        self.num_scroll.setWidget(self.num_inner)
        self.num_vbox = QtWidgets.QVBoxLayout(self.num_inner)
        self.num_vbox.setSpacing(10)

        self._local_items: List[Tuple[QtWidgets.QLabel, QtWidgets.QTableWidget]] = []
        self._global_items: List[Tuple[QtWidgets.QLabel, QtWidgets.QTableWidget]] = []

        # -------- Hybrid tab (tables)
        self.hy_scroll = QtWidgets.QScrollArea()
        self.hy_scroll.setWidgetResizable(True)
        self.tabs.addTab(self.hy_scroll, "Hybrid")

        self.hy_inner = QtWidgets.QWidget()
        self.hy_scroll.setWidget(self.hy_inner)
        self.hy_vbox = QtWidgets.QVBoxLayout(self.hy_inner)
        self.hy_vbox.setSpacing(10)

        self._hy_items: List[Tuple[QtWidgets.QLabel, QtWidgets.QTableWidget]] = []
        self.hy_hdr = QtWidgets.QLabel(
            "Hybrid matrices: symbol (top) + live value (bottom)"
        )
        self.hy_hdr.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.hy_vbox.addWidget(self.hy_hdr)

    def _make_hybrid_table(self) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(4, 4)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        t.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        t.setFixedHeight(190)  # taller for 2-line cells

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        t.setFont(font)

        for r in range(4):
            for c in range(4):
                it = QtWidgets.QTableWidgetItem("")
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                t.setItem(r, c, it)
        return t

    def _set_hybrid_cell(
        self, table: QtWidgets.QTableWidget, r: int, c: int, expr: str, val: float
    ):
        table.item(r, c).setText(f"{expr}\n{val: .6f}")

    def _clear_hybrid(self):
        # remove old hybrid widgets except header
        while self.hy_vbox.count() > 1:
            item = self.hy_vbox.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._hy_items.clear()

    def _axis_is_basis(self, axis: np.ndarray, tol: float = 1e-6) -> Optional[str]:
        a = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return None
        a = a / n
        if np.allclose(a, [1, 0, 0], atol=tol):
            return "X"
        if np.allclose(a, [0, 1, 0], atol=tol):
            return "Y"
        if np.allclose(a, [0, 0, 1], atol=tol):
            return "Z"
        if np.allclose(a, [-1, 0, 0], atol=tol):
            return "-X"
        if np.allclose(a, [0, -1, 0], atol=tol):
            return "-Y"
        if np.allclose(a, [0, 0, -1], atol=tol):
            return "-Z"
        return None

    def _format_theta_live(self, q_rad: float, angle_unit: str) -> str:
        if angle_unit == "deg":
            return f"{math.degrees(q_rad):.3f}° ({q_rad:.5f} rad)"
        return f"{q_rad:.5f} rad"

    def _make_table(self) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(4, 4)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        t.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        t.setFixedHeight(140)
        for r in range(4):
            for c in range(4):
                it = QtWidgets.QTableWidgetItem("0.000000")
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                t.setItem(r, c, it)
        return t

    def _set_table(self, table: QtWidgets.QTableWidget, T: np.ndarray):
        for r in range(4):
            for c in range(4):
                table.item(r, c).setText(f"{float(T[r,c]): .6f}")

    def _clear_numeric(self):
        while self.num_vbox.count():
            item = self.num_vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._local_items.clear()
        self._global_items.clear()

    def set_transforms(
        self,
        elements: List[Element],
        locals_list: List[np.ndarray],
        globals_list: List[np.ndarray],
        angle_unit: str,  # "rad" or "deg"
    ):
        n = len(locals_list)

        # ---------- Numeric rebuild if counts changed
        if len(self._local_items) != n or len(self._global_items) != len(globals_list):
            self._clear_numeric()

            hdr1 = QtWidgets.QLabel("Local transforms (iH(i+1))")
            hdr1.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.num_vbox.addWidget(hdr1)

            for i in range(n):
                lbl = QtWidgets.QLabel(f"{i}H{i+1}")
                lbl.setStyleSheet("font-weight: bold;")
                table = self._make_table()
                self.num_vbox.addWidget(lbl)
                self.num_vbox.addWidget(table)
                self._local_items.append((lbl, table))

            hdr2 = QtWidgets.QLabel("Global transforms (0Hk)")
            hdr2.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 12px;")
            self.num_vbox.addWidget(hdr2)

            for k in range(len(globals_list)):
                lbl = QtWidgets.QLabel(f"0H{k}")
                lbl.setStyleSheet("font-weight: bold;")
                table = self._make_table()
                self.num_vbox.addWidget(lbl)
                self.num_vbox.addWidget(table)
                self._global_items.append((lbl, table))

            self.num_vbox.addStretch(1)

        # ---------- Numeric update
        for i, Tl in enumerate(locals_list):
            self._set_table(self._local_items[i][1], Tl)
        for k, Tg in enumerate(globals_list):
            self._set_table(self._global_items[k][1], Tg)

        # ---------- Hybrid update (formulas + live values)
        self._set_hybrid_tables(elements, locals_list, angle_unit)

    def _set_hybrid_tables(
        self, elements: List[Element], locals_list: List[np.ndarray], angle_unit: str
    ):
        # rebuild if needed
        if len(self._hy_items) != len(locals_list):
            self._clear_hybrid()
            for i in range(len(locals_list)):
                lbl = QtWidgets.QLabel(f"{i}H{i+1}  ({elements[i].name})")
                lbl.setStyleSheet("font-weight: bold;")
                tab = self._make_hybrid_table()
                self.hy_vbox.addWidget(lbl)
                self.hy_vbox.addWidget(tab)
                self._hy_items.append((lbl, tab))
            self.hy_vbox.addStretch(1)

        # fill each 4x4 with expr + value
        rev_k = 0
        pri_k = 0

        for i, e in enumerate(elements):
            T = locals_list[i]
            table = self._hy_items[i][1]

            # default: numeric only
            def fill_numeric_only():
                for r in range(4):
                    for c in range(4):
                        self._set_hybrid_cell(table, r, c, " ", float(T[r, c]))

            if isinstance(e, Link):
                # Hybrid: show translation symbols for last column (top), numeric below
                dx, dy, dz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
                # rotation part
                for r in range(3):
                    for c in range(3):
                        expr = (
                            "1"
                            if (r == c and abs(T[r, c] - 1.0) < 1e-9)
                            else ("0" if abs(T[r, c]) < 1e-9 else "R")
                        )
                        self._set_hybrid_cell(table, r, c, expr, float(T[r, c]))
                self._set_hybrid_cell(table, 0, 3, "dx", dx)
                self._set_hybrid_cell(table, 1, 3, "dy", dy)
                self._set_hybrid_cell(table, 2, 3, "dz", dz)
                # last row
                self._set_hybrid_cell(table, 3, 0, "0", 0.0)
                self._set_hybrid_cell(table, 3, 1, "0", 0.0)
                self._set_hybrid_cell(table, 3, 2, "0", 0.0)
                self._set_hybrid_cell(table, 3, 3, "1", 1.0)
                continue

            # Joint
            if e.joint_type == "revolute":
                rev_k += 1
                sym = f"θ{rev_k}"
                q = float(e.q)
                ax = self._axis_is_basis(e.axis)

                # handle negative axes as sign flip on theta
                theta_sym = sym
                theta_val = q
                if ax and ax.startswith("-"):
                    theta_sym = f"-{sym}"
                    theta_val = -q
                    ax = ax[1:]

                c = math.cos(theta_val)
                s = math.sin(theta_val)

                # Build symbolic templates for Rx/Ry/Rz only
                if ax == "Z":
                    exprM = [
                        ["c" + str(rev_k), "-s" + str(rev_k), "0", "0"],
                        ["s" + str(rev_k), "c" + str(rev_k), "0", "0"],
                        ["0", "0", "1", "0"],
                        ["0", "0", "0", "1"],
                    ]
                    valM = [
                        [c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                elif ax == "Y":
                    exprM = [
                        ["c" + str(rev_k), "0", "s" + str(rev_k), "0"],
                        ["0", "1", "0", "0"],
                        ["-s" + str(rev_k), "0", "c" + str(rev_k), "0"],
                        ["0", "0", "0", "1"],
                    ]
                    valM = [
                        [c, 0, s, 0],
                        [0, 1, 0, 0],
                        [-s, 0, c, 0],
                        [0, 0, 0, 1],
                    ]
                elif ax == "X":
                    exprM = [
                        ["1", "0", "0", "0"],
                        ["0", "c" + str(rev_k), "-s" + str(rev_k), "0"],
                        ["0", "s" + str(rev_k), "c" + str(rev_k), "0"],
                        ["0", "0", "0", "1"],
                    ]
                    valM = [
                        [1, 0, 0, 0],
                        [0, c, -s, 0],
                        [0, s, c, 0],
                        [0, 0, 0, 1],
                    ]
                else:
                    # custom axis: numeric-only in matrix cells (still tabular)
                    fill_numeric_only()
                    # update header with live theta info
                    self._hy_items[i][0].setText(
                        f"{i}H{i+1}  ({e.name})   axis=custom   {sym}={self._format_theta_live(q, angle_unit)}"
                    )
                    continue

                # fill
                for r in range(4):
                    for c2 in range(4):
                        self._set_hybrid_cell(
                            table, r, c2, exprM[r][c2], float(valM[r][c2])
                        )

                # update header with live angle
                self._hy_items[i][0].setText(
                    f"{i}H{i+1}  ({e.name})   {ax}   {sym}={self._format_theta_live(q, angle_unit)}   c{rev_k}={c:.6f} s{rev_k}={s:.6f}"
                )
                continue

            else:
                # prismatic
                pri_k += 1
                sym = f"d{pri_k}"
                d = float(e.q)
                ax = self._axis_is_basis(e.axis)

                # For basis axes, show symbol in translation component
                if ax in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                    # use sign for -axis
                    sign = -1.0 if ax.startswith("-") else 1.0
                    base = ax[1:] if ax.startswith("-") else ax

                    # rotation is identity for pure prismatic motion
                    for r in range(3):
                        for c2 in range(3):
                            self._set_hybrid_cell(
                                table, r, c2, "1" if r == c2 else "0", float(T[r, c2])
                            )

                    # translation
                    expr_dx, expr_dy, expr_dz = "0", "0", "0"
                    val_dx, val_dy, val_dz = 0.0, 0.0, 0.0
                    if base == "X":
                        expr_dx = f"{sym}" if sign > 0 else f"-{sym}"
                        val_dx = d
                    elif base == "Y":
                        expr_dy = f"{sym}" if sign > 0 else f"-{sym}"
                        val_dy = d
                    else:
                        expr_dz = f"{sym}" if sign > 0 else f"-{sym}"
                        val_dz = d

                    self._set_hybrid_cell(table, 0, 3, expr_dx, float(T[0, 3]))
                    self._set_hybrid_cell(table, 1, 3, expr_dy, float(T[1, 3]))
                    self._set_hybrid_cell(table, 2, 3, expr_dz, float(T[2, 3]))

                    # last row
                    self._set_hybrid_cell(table, 3, 0, "0", 0.0)
                    self._set_hybrid_cell(table, 3, 1, "0", 0.0)
                    self._set_hybrid_cell(table, 3, 2, "0", 0.0)
                    self._set_hybrid_cell(table, 3, 3, "1", 1.0)

                    self._hy_items[i][0].setText(
                        f"{i}H{i+1}  ({e.name})   {ax}   {sym}={d:.5f} m"
                    )
                    continue

                # custom axis -> numeric-only table
                fill_numeric_only()
                self._hy_items[i][0].setText(
                    f"{i}H{i+1}  ({e.name})   axis=custom   d={d:.5f} m"
                )

    def _build_hybrid_text(self, elements: List[Element], angle_unit: str) -> str:
        """
        Hybrid view:
          - show function form (Rx/Ry/Rz if axis aligned; otherwise R(â,θ))
          - show live values (degrees + radians when in degrees mode)
          - show c/s values for revolute basis axes
        """
        # assign symbols per joint
        rev_count = 0
        pri_count = 0

        lines: List[str] = []
        lines.append("HYBRID VIEW (structure + live values)")
        lines.append(
            "Internals: revolute q stored in radians; prismatic q stored in meters."
        )
        lines.append("")

        for i, e in enumerate(elements):
            label = f"{i}H{i+1}"
            lines.append("=" * 70)
            lines.append(f"{label}   ({e.name})")

            if isinstance(e, Link):
                T = e.T
                dx, dy, dz = T[0, 3], T[1, 3], T[2, 3]
                lines.append("Type: LINK (fixed)")
                lines.append("Form:  T_link")
                lines.append(
                    f"Live:  translation = [dx,dy,dz] = [{dx:.4f}, {dy:.4f}, {dz:.4f}]"
                )
                lines.append("")

            else:
                if e.joint_type == "revolute":
                    rev_count += 1
                    sym = f"θ{rev_count}"
                    q_rad = float(e.q)

                    if angle_unit == "deg":
                        q_deg = math.degrees(q_rad)
                        q_str = f"{q_deg:.3f}° ({q_rad:.5f} rad)"
                    else:
                        q_str = f"{q_rad:.5f} rad"

                    axname = _axis_basis_name(e.axis)
                    lines.append("Type: JOINT (revolute)")
                    lines.append("Form:  T_mount · R(axis, θ)")
                    lines.append(f"Live:  {sym} = {q_str}")

                    if axname in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                        # show nice Rx/Ry/Rz
                        if axname == "X":
                            Rform = f"Rx({sym})"
                        elif axname == "Y":
                            Rform = f"Ry({sym})"
                        elif axname == "Z":
                            Rform = f"Rz({sym})"
                        else:
                            # negative axis
                            base = axname[1:]
                            Rform = f"R{base}(-{sym})  (since axis = {axname})"
                        lines.append(f"Axis:  {axname}")
                        lines.append(f"Rot:   {Rform}")

                        # trig aliases
                        c = math.cos(q_rad)
                        s = math.sin(q_rad)
                        lines.append(
                            f"Trig:  c{rev_count}=cos({sym})={c:.6f}   s{rev_count}=sin({sym})={s:.6f}"
                        )
                    else:
                        a = np.asarray(e.axis, dtype=float)
                        n = float(np.linalg.norm(a))
                        if n < 1e-12:
                            ah = np.array([0.0, 0.0, 0.0])
                        else:
                            ah = a / n
                        lines.append(
                            f"Axis:  â = [{ah[0]:.4f}, {ah[1]:.4f}, {ah[2]:.4f}]"
                        )
                        lines.append(f"Rot:   R(â, {sym})  (Rodrigues axis-angle)")

                    lines.append("")

                else:
                    pri_count += 1
                    sym = f"d{pri_count}"
                    d = float(e.q)
                    lines.append("Type: JOINT (prismatic)")
                    lines.append("Form:  T_mount · T(axis · d)")
                    lines.append(f"Live:  {sym} = {d:.5f} m")

                    axname = _axis_basis_name(e.axis)
                    if axname in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                        if axname == "X":
                            Tform = f"Tx({sym})"
                        elif axname == "Y":
                            Tform = f"Ty({sym})"
                        elif axname == "Z":
                            Tform = f"Tz({sym})"
                        else:
                            base = axname[1:]
                            Tform = f"T{base}(-{sym})  (since axis = {axname})"
                        lines.append(f"Axis:  {axname}")
                        lines.append(f"Trans: {Tform}")
                    else:
                        a = np.asarray(e.axis, dtype=float)
                        n = float(np.linalg.norm(a))
                        if n < 1e-12:
                            ah = np.array([0.0, 0.0, 0.0])
                        else:
                            ah = a / n
                        lines.append(
                            f"Axis:  â = [{ah[0]:.4f}, {ah[1]:.4f}, {ah[2]:.4f}]"
                        )
                        lines.append(f"Trans: T(â * {sym})")

                    lines.append("")

        lines.append("=" * 70)
        lines.append(
            "Tip: For planar X–Z motion, set revolute joint axis to world Y (use Axis preset = Y)."
        )
        return "\n".join(lines)


class GLRobotView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=1.2, elevation=25, azimuth=35)
        self._frame_items = []
        self._seg_items = []
        self._ee_items = []

        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        self.addItem(grid)

    def clear_robot(self):
        for it in self._frame_items + self._seg_items + self._ee_items:
            self.removeItem(it)
        self._frame_items.clear()
        self._seg_items.clear()
        self._ee_items.clear()

    def draw_chain(
        self,
        Ts_global: List[np.ndarray],
        elements: List[Element],
        end_effector: Optional[EndEffector],
    ):
        """
        Draw segment geometry for each element between frame i and i+1.
        - Link: gray box
        - Revolute joint: red cylinder
        - Prismatic joint: blue box
        End effector is visual-only, attached to final frame.
        """
        self.clear_robot()

        # origins[k] corresponds to Ts_global[k]
        origins = np.array([T[:3, 3] for T in Ts_global], dtype=float)

        # -------- Draw LINKS (between origins) --------
        for i in range(len(Ts_global) - 1):
            e = elements[i]
            if not isinstance(e, Link):
                continue  # links only here

            p0 = origins[i]
            p1 = origins[i + 1]
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue

            md = center_meshdata_z(meshdata_box(size=(0.02, 0.02, L)))

            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                color=(0.65, 0.65, 0.65, 1.0),
            )

            R = rotation_matrix_from_z_to_vec(d)
            mid = 0.5 * (p0 + p1)
            Tm = np.eye(4)
            Tm[:3, :3] = R
            Tm[:3, 3] = mid  # - R @ np.array([0.0, 0.0, L / 2], dtype=float)
            mesh.setTransform(qmatrix_from_T(Tm))

            self.addItem(mesh)
            self._seg_items.append(mesh)
            # -------- Draw JOINTS (at joint origin, always visible) --------
        joint_len = 0.08
        rev_radius = 0.02
        pris_w, pris_h = 0.05, 0.035  # cross-section

        for i in range(len(elements)):
            e = elements[i]
            if not isinstance(e, Joint):
                continue

            # Joint origin is frame i origin
            p0 = origins[i]

            # Compute world axis using the joint base pose (before motion)
            T_base = Ts_global[i] @ e.T_mount
            a = np.asarray(e.axis, dtype=float)
            a = a / (np.linalg.norm(a) + 1e-12)
            axis_w = T_base[:3, :3] @ a

            # Make transform that places a z-aligned mesh on the joint axis at p0
            R = rotation_matrix_from_z_to_vec(axis_w)
            mid = 0.5 * (p0 + p1)
            Tm = np.eye(4)
            Tm[:3, :3] = R
            Tm[:3, 3] = p0  # p0 - R @ np.array([0.0, 0.0, L / 4], dtype=float)

            if e.joint_type == "revolute":
                md = gl.MeshData.cylinder(
                    rows=12, cols=24, radius=[rev_radius, rev_radius], length=joint_len
                )
                md = center_meshdata_z(md)

                mesh = gl.GLMeshItem(
                    meshdata=md,
                    smooth=True,
                    drawFaces=True,
                    drawEdges=False,
                    color=(0.85, 0.35, 0.35, 1.0),
                )
                mesh.setTransform(qmatrix_from_T(Tm))
                self.addItem(mesh)
                self._seg_items.append(mesh)

            else:
                # prismatic: show a housing always, and optionally an extension
                # housing (fixed length)
                T_base = Ts_global[i] @ e.T_mount
                R_base = T_base[:3, :3]
                p_base = T_base[:3, 3]

                base_len = 0.08
                md = meshdata_box(size=(pris_w, pris_h, base_len))
                md = center_meshdata_z(md)

                housing = gl.GLMeshItem(
                    meshdata=md,
                    smooth=False,
                    drawFaces=True,
                    drawEdges=True,
                    color=(0.25, 0.35, 0.60, 1.0),
                )
                Th = np.eye(4, dtype=float)
                Th[:3, :3] = R_base
                Th[:3, 3] = p_base

                housing.setTransform(qmatrix_from_T(Th))

                self.addItem(housing)
                self._seg_items.append(housing)

                # slider extension proportional to |q| (so you can "see" motion)
                ext = max(0.0, float(e.q))  # only extend in +axis direction

                if ext > 1e-6:
                    md_slider = center_meshdata_z(
                        meshdata_box(size=(pris_w * 0.8, pris_h * 0.8, ext))
                    )

                    slider = gl.GLMeshItem(
                        meshdata=md_slider,
                        smooth=False,
                        drawFaces=True,
                        drawEdges=True,
                        color=(0.35, 0.55, 0.90, 1.0),
                    )

                    Tslide = np.eye(4, dtype=float)
                    Tslide[:3, :3] = R_base

                    # slider is centered in its own +Z because of center_meshdata_z,
                    # so place its CENTER in front of the housing:
                    offset = (base_len / 2.0) + (ext / 2.0)

                    Tslide[:3, 3] = p_base + R_base @ np.array(
                        [0.0, 0.0, offset], dtype=float
                    )

                    slider.setTransform(qmatrix_from_T(Tslide))
                    self.addItem(slider)
                    self._seg_items.append(slider)

        # frame triads
        axis_len = 0.06
        for T in Ts_global:
            o = origin_of(T)
            R = axes_of(T)
            x = o + axis_len * R[:, 0]
            y = o + axis_len * R[:, 1]
            z = o + axis_len * R[:, 2]

            lx = gl.GLLinePlotItem(pos=np.vstack([o, x]), width=3, color=(1, 0, 0, 1))
            ly = gl.GLLinePlotItem(pos=np.vstack([o, y]), width=3, color=(0, 1, 0, 1))
            lz = gl.GLLinePlotItem(pos=np.vstack([o, z]), width=3, color=(0, 0, 1, 1))

            for it in (lx, ly, lz):
                self.addItem(it)
                self._frame_items.append(it)

        # end effector (visual only)
        if end_effector is not None and len(Ts_global) > 0:
            T_end = Ts_global[-1]
            T_ee = T_end @ end_effector.T

            def add_mesh(md, color, Tlocal=np.eye(4), smooth=False, edges=True):
                mesh = gl.GLMeshItem(
                    meshdata=md,
                    smooth=smooth,
                    drawFaces=True,
                    drawEdges=edges,
                    color=color,
                )
                mesh.setTransform(qmatrix_from_T(T_ee @ Tlocal))
                self.addItem(mesh)
                self._ee_items.append(mesh)

            ee = end_effector.ee_type

            if ee == EEType.FLANGE:
                md = gl.MeshData.cylinder(
                    rows=8, cols=32, radius=[0.04, 0.04], length=0.02
                )
                add_mesh(md, (0.7, 0.7, 0.75, 1.0), smooth=True, edges=False)

            elif ee == EEType.SUCTION:
                md = gl.MeshData.cylinder(
                    rows=8, cols=32, radius=[0.03, 0.045], length=0.04
                )
                add_mesh(md, (0.2, 0.2, 0.2, 1.0), smooth=True, edges=False)

            elif ee == EEType.CLAW:
                base = meshdata_box(size=(0.06, 0.05, 0.03))
                add_mesh(base, (0.55, 0.55, 0.55, 1.0), edges=True)

                finger = meshdata_box(size=(0.015, 0.015, 0.08))
                Tl = np.eye(4)
                Tl[:3, 3] = np.array([-0.025, 0.0, 0.03])
                add_mesh(finger, (0.85, 0.75, 0.25, 1.0), Tlocal=Tl, edges=True)

                Tr = np.eye(4)
                Tr[:3, 3] = np.array([+0.025, 0.0, 0.03])
                add_mesh(finger, (0.85, 0.75, 0.25, 1.0), Tlocal=Tr, edges=True)


# =========================
# Main window
# =========================


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Robot Frames Sandbox (Joints + Links + EE + Hybrid + Deg/Rad)"
        )
        self.resize(1380, 760)

        self.robot = RobotChain()

        # starter: J, L, J, L, J, L
        self.robot.add_revolute_joint()
        self.robot.add_link(dx=0.0, dy=0.0, dz=0.2)
        self.robot.add_revolute_joint()
        self.robot.add_link(dx=0.2)
        self.robot.add_prismatic_joint()
        self.robot.add_link(dx=0.2)

        self.angle_unit = "rad"  # "rad" or "deg"

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left panel
        left = QtWidgets.QVBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.btn_rev = QtWidgets.QPushButton("Add Revolute Joint")
        self.btn_pri = QtWidgets.QPushButton("Add Prismatic Joint")
        self.btn_link = QtWidgets.QPushButton("Add Link (Fixed)")
        self.btn_rm = QtWidgets.QPushButton("Remove Selected")

        left.addWidget(QtWidgets.QLabel("Elements (Joint / Link)"))
        left.addWidget(self.list, 1)
        left.addWidget(self.btn_rev)
        left.addWidget(self.btn_pri)
        left.addWidget(self.btn_link)
        left.addWidget(self.btn_rm)

        # End effector controls
        left.addSpacing(12)
        left.addWidget(QtWidgets.QLabel("End Effector (visual)"))
        self.ee_combo = QtWidgets.QComboBox()
        self.ee_combo.addItems(
            [EEType.FLANGE.value, EEType.SUCTION.value, EEType.CLAW.value]
        )
        self.btn_set_ee = QtWidgets.QPushButton("Set End Effector")
        self.btn_clear_ee = QtWidgets.QPushButton("Clear End Effector")
        left.addWidget(self.ee_combo)
        left.addWidget(self.btn_set_ee)
        left.addWidget(self.btn_clear_ee)

        left.addStretch(1)

        # Center view
        self.view = GLRobotView()

        # Right panel
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Inspector"))

        # Units toggle
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Angle units:"))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["Radians", "Degrees"])
        unit_row.addWidget(self.unit_combo, 1)
        right.addLayout(unit_row)

        self.lbl_kind = QtWidgets.QLabel("")
        right.addWidget(self.lbl_kind)

        self.stack = QtWidgets.QStackedWidget()
        right.addWidget(self.stack, 1)

        # ---- Joint editor page ----
        joint_page = QtWidgets.QWidget()
        jl = QtWidgets.QVBoxLayout(joint_page)

        self.joint_info = QtWidgets.QLabel("Select a joint")
        jl.addWidget(self.joint_info)

        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(QtWidgets.QLabel("Axis preset:"))
        self.axis_mode = QtWidgets.QComboBox()
        self.axis_mode.addItems(["Z", "Y", "X", "Custom"])
        axis_row.addWidget(self.axis_mode, 1)
        jl.addLayout(axis_row)

        self.axis_x = QtWidgets.QDoubleSpinBox()
        self.axis_y = QtWidgets.QDoubleSpinBox()
        self.axis_z = QtWidgets.QDoubleSpinBox()
        for sb in (self.axis_x, self.axis_y, self.axis_z):
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(6)
            sb.setSingleStep(0.1)

        axis_vals = QtWidgets.QHBoxLayout()
        axis_vals.addWidget(QtWidgets.QLabel("ax"))
        axis_vals.addWidget(self.axis_x)
        axis_vals.addWidget(QtWidgets.QLabel("ay"))
        axis_vals.addWidget(self.axis_y)
        axis_vals.addWidget(QtWidgets.QLabel("az"))
        axis_vals.addWidget(self.axis_z)
        jl.addLayout(axis_vals)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.lbl_q = QtWidgets.QLabel("q = 0.0")
        jl.addWidget(self.slider)
        jl.addWidget(self.lbl_q)

        jl.addWidget(QtWidgets.QLabel("Joint mount transform (fixed alignment)"))
        self.joint_mount = MatrixEditor()
        jl.addWidget(self.joint_mount, 1)
        self.joint_valid = QtWidgets.QLabel("")
        jl.addWidget(self.joint_valid)

        # ---- Link editor page ----
        link_page = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(link_page)

        self.link_info = QtWidgets.QLabel("Select a link")
        ll.addWidget(self.link_info)

        ll.addWidget(QtWidgets.QLabel("Link transform (fixed) — use this for L-shapes"))
        self.link_mat = MatrixEditor()
        ll.addWidget(self.link_mat, 1)
        self.link_valid = QtWidgets.QLabel("")
        ll.addWidget(self.link_valid)

        self.stack.addWidget(joint_page)  # index 0
        self.stack.addWidget(link_page)  # index 1

        layout.addLayout(left, 1)
        layout.addWidget(self.view, 2)
        layout.addLayout(right, 1)

        # Matrices window
        self.mats_win = MatricesWindow(self)
        self.mats_win.show()

        # signals
        self.btn_rev.clicked.connect(self.add_rev)
        self.btn_pri.clicked.connect(self.add_pri)
        self.btn_link.clicked.connect(self.add_link)
        self.btn_rm.clicked.connect(self.remove_selected)
        self.list.currentRowChanged.connect(self.select_element)

        self.slider.valueChanged.connect(self.slider_changed)

        self.axis_mode.currentIndexChanged.connect(self.axis_mode_changed)
        self.axis_x.valueChanged.connect(self.axis_values_changed)
        self.axis_y.valueChanged.connect(self.axis_values_changed)
        self.axis_z.valueChanged.connect(self.axis_values_changed)

        self.joint_mount.matrixChanged.connect(self.joint_mount_edited)
        self.link_mat.matrixChanged.connect(self.link_matrix_edited)

        self.btn_set_ee.clicked.connect(self.set_end_effector)
        self.btn_clear_ee.clicked.connect(self.clear_end_effector)

        self.unit_combo.currentIndexChanged.connect(self.unit_changed)

        self._block = False
        self.refresh_list()
        self.list.setCurrentRow(0)
        self.update_view()

    # ---------- Units helpers ----------

    def q_to_display(self, q_rad: float) -> float:
        if self.angle_unit == "deg":
            return math.degrees(q_rad)
        return q_rad

    def format_q_label_joint(self, j: Joint) -> str:
        if j.joint_type == "prismatic":
            return f"q = {j.q:.5f} m"
        # revolute
        q_rad = float(j.q)
        if self.angle_unit == "deg":
            return f"q = {math.degrees(q_rad):.3f} deg  ({q_rad:.5f} rad)"
        return f"q = {q_rad:.5f} rad"

    def unit_changed(self, _idx: int):
        self.angle_unit = "deg" if self.unit_combo.currentText() == "Degrees" else "rad"
        e = self.current_element()
        if isinstance(e, Joint):
            self.lbl_q.setText(self.format_q_label_joint(e))
        self.update_view()

    # ---------- Helpers ----------

    def refresh_list(self):
        self.list.clear()
        for e in self.robot.elements:
            self.list.addItem(e.name)

    def current_index(self) -> int:
        return self.list.currentRow()

    def current_element(self) -> Optional[Element]:
        i = self.current_index()
        if 0 <= i < len(self.robot.elements):
            return self.robot.elements[i]
        return None

    def update_view(self):
        Ts = self.robot.fk_all()
        self.view.draw_chain(Ts, self.robot.elements, self.robot.end_effector)

        locals_list = [e.local_T() for e in self.robot.elements]
        self.mats_win.set_transforms(
            self.robot.elements, locals_list, Ts, self.angle_unit
        )

    def set_slider_from_q(self, j: Joint):
        t = (j.q - j.q_min) / (j.q_max - j.q_min + 1e-12)
        self.slider.setValue(int(np.clip(t, 0, 1) * 1000))

    def set_q_from_slider(self, j: Joint, v: int):
        t = v / 1000.0
        j.q = j.q_min + t * (j.q_max - j.q_min)

    def _set_joint_axis(self, j: Joint, axis_vec: np.ndarray):
        axis_vec = np.asarray(axis_vec, dtype=float)
        n = float(np.linalg.norm(axis_vec))
        if n < 1e-12:
            return
        j.axis = axis_vec / n

    # ---------- Actions ----------

    def add_rev(self):
        self.robot.add_revolute_joint()
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def add_pri(self):
        self.robot.add_prismatic_joint()
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def add_link(self):
        self.robot.add_link(dx=0.2, dy=0.0, dz=0.0)
        self.refresh_list()
        self.list.setCurrentRow(len(self.robot.elements) - 1)
        self.update_view()

    def remove_selected(self):
        idx = self.current_index()
        self.robot.remove(idx)
        self.refresh_list()
        self.list.setCurrentRow(min(idx, len(self.robot.elements) - 1))
        self.update_view()

    # ---------- End Effector ----------

    def set_end_effector(self):
        txt = self.ee_combo.currentText()
        if self.robot.end_effector is None:
            self.robot.end_effector = EndEffector(EEType.FLANGE, np.eye(4))

        if txt == EEType.FLANGE.value:
            self.robot.end_effector.ee_type = EEType.FLANGE
        elif txt == EEType.SUCTION.value:
            self.robot.end_effector.ee_type = EEType.SUCTION
        else:
            self.robot.end_effector.ee_type = EEType.CLAW

        self.update_view()

    def clear_end_effector(self):
        self.robot.end_effector = None
        self.update_view()

    # ---------- Selection ----------

    def select_element(self, _idx: int):
        e = self.current_element()
        if e is None:
            return

        self._block = True

        if isinstance(e, Joint):
            self.lbl_kind.setText("Selected: JOINT")
            self.stack.setCurrentIndex(0)

            self.joint_info.setText(
                f"{e.name} | {e.joint_type} | axis={e.axis.tolist()}"
            )
            self.set_slider_from_q(e)
            self.lbl_q.setText(self.format_q_label_joint(e))

            ax, ay, az = map(float, e.axis.tolist())
            self.axis_x.setValue(ax)
            self.axis_y.setValue(ay)
            self.axis_z.setValue(az)

            if np.allclose(e.axis, [1, 0, 0], atol=1e-6):
                self.axis_mode.setCurrentText("X")
            elif np.allclose(e.axis, [0, 1, 0], atol=1e-6):
                self.axis_mode.setCurrentText("Y")
            elif np.allclose(e.axis, [0, 0, 1], atol=1e-6):
                self.axis_mode.setCurrentText("Z")
            else:
                self.axis_mode.setCurrentText("Custom")

            self.joint_mount.set_matrix(e.T_mount)
            self.joint_valid.setText("")

        else:
            self.lbl_kind.setText("Selected: LINK")
            self.stack.setCurrentIndex(1)

            self.link_info.setText(f"{e.name} (fixed link transform)")
            self.link_mat.set_matrix(e.T)
            self.link_valid.setText("")

        self._block = False

    # ---------- Joint editor callbacks ----------

    def slider_changed(self, v: int):
        if self._block:
            return
        e = self.current_element()
        if not isinstance(e, Joint):
            return

        self.set_q_from_slider(e, v)
        self.lbl_q.setText(self.format_q_label_joint(e))
        self.update_view()

    def axis_mode_changed(self, _idx: int):
        if self._block:
            return
        e = self.current_element()
        if not isinstance(e, Joint):
            return

        mode = self.axis_mode.currentText()
        if mode == "X":
            self._set_joint_axis(e, np.array([1.0, 0.0, 0.0]))
        elif mode == "Y":
            self._set_joint_axis(e, np.array([0.0, 1.0, 0.0]))
        elif mode == "Z":
            self._set_joint_axis(e, np.array([0.0, 0.0, 1.0]))
        else:
            axis = np.array(
                [self.axis_x.value(), self.axis_y.value(), self.axis_z.value()],
                dtype=float,
            )
            self._set_joint_axis(e, axis)

        self._block = True
        self.joint_info.setText(f"{e.name} | {e.joint_type} | axis={e.axis.tolist()}")
        self._block = False
        self.update_view()

    def axis_values_changed(self, *_args):
        if self._block:
            return
        e = self.current_element()
        if not isinstance(e, Joint):
            return

        axis = np.array(
            [self.axis_x.value(), self.axis_y.value(), self.axis_z.value()], dtype=float
        )
        if float(np.linalg.norm(axis)) < 1e-12:
            return

        if self.axis_mode.currentText() != "Custom":
            self._block = True
            self.axis_mode.setCurrentText("Custom")
            self._block = False

        self._set_joint_axis(e, axis)

        self._block = True
        self.joint_info.setText(f"{e.name} | {e.joint_type} | axis={e.axis.tolist()}")
        self._block = False
        self.update_view()

    def joint_mount_edited(self, T_user: np.ndarray):
        if self._block:
            return
        e = self.current_element()
        if not isinstance(e, Joint):
            return

        T = T_user.copy()
        T[3, :] = np.array([0, 0, 0, 1], dtype=float)
        T[:3, :3] = project_to_so3(T[:3, :3])

        e.T_mount = T

        self._block = True
        self.joint_mount.set_matrix(e.T_mount)
        self.joint_valid.setText(
            "✅ SE(3) looks valid"
            if is_valid_transform(T)
            else "⚠️ Rotation projected; last row forced"
        )
        self._block = False

        self.update_view()

    # ---------- Link editor callbacks ----------

    def link_matrix_edited(self, T_user: np.ndarray):
        if self._block:
            return
        e = self.current_element()
        if not isinstance(e, Link):
            return

        T = T_user.copy()
        T[3, :] = np.array([0, 0, 0, 1], dtype=float)
        T[:3, :3] = project_to_so3(T[:3, :3])

        e.T = T

        self._block = True
        self.link_mat.set_matrix(e.T)
        self.link_valid.setText(
            "✅ SE(3) looks valid"
            if is_valid_transform(T)
            else "⚠️ Rotation projected; last row forced"
        )
        self._block = False

        self.update_view()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
