# model/kinematics.py

import math  # Provides trig + atan2 for rotation computations
import numpy as np  # Provides arrays + linear algebra tools


def hat(
    w: np.ndarray,
) -> np.ndarray:  # Convert a 3-vector into a skew-symmetric "cross product matrix"
    wx, wy, wz = w  # Unpack the vector components for readability
    return np.array(  # Build and return the skew-symmetric matrix
        [  # Start matrix rows
            [0.0, -wz, wy],  # Row 1:  0  -wz  wy
            [wz, 0.0, -wx],  # Row 2:  wz  0  -wx
            [-wy, wx, 0.0],  # Row 3: -wy wx   0
        ],  # End rows
        dtype=float,  # Force float dtype for stable numeric operations
    )  # Return the constructed matrix


def rot_axis_angle(
    axis: np.ndarray, theta: float
) -> np.ndarray:  # Rodrigues rotation formula for axis-angle rotation
    axis = np.asarray(axis, dtype=float)  # Ensure axis is a float numpy array
    n = np.linalg.norm(axis)  # Compute axis length (magnitude)
    if n < 1e-12:  # If axis is near-zero, rotation is ill-defined
        return np.eye(3, dtype=float)  # Return identity rotation (no rotation)
    a = axis / n  # Normalize axis to unit length
    K = hat(a)  # Compute skew matrix for Rodrigues formula
    return (  # Return the rotation matrix computed by Rodrigues' formula
        np.eye(3)  # Identity term
        + math.sin(theta) * K  # sin(theta)*K term
        + (1.0 - math.cos(theta)) * (K @ K)  # (1-cos(theta))*K^2 term
    )  # End Rodrigues expression


def T_from_Rp(
    R: np.ndarray, p: np.ndarray
) -> np.ndarray:  # Build a 4x4 homogeneous transform from rotation + translation
    T = np.eye(4, dtype=float)  # Start with 4x4 identity matrix
    T[:3, :3] = R  # Insert rotation into top-left 3x3 block
    T[:3, 3] = p  # Insert translation into top-right 3x1 column
    return T  # Return the resulting 4x4 transform


def project_to_so3(
    R: np.ndarray,
) -> np.ndarray:  # Project a matrix onto the nearest valid rotation matrix in SO(3)
    U, _, Vt = np.linalg.svd(R)  # SVD decomposition of R into U * S * V^T
    Rp = U @ Vt  # Closest orthonormal matrix (polar decomposition result)
    if (
        np.linalg.det(Rp) < 0
    ):  # If determinant is negative, we got a reflection (not a proper rotation)
        U[:, -1] *= -1  # Flip sign of last column of U to correct the reflection
        Rp = U @ Vt  # Recompute Rp with corrected U
    return Rp  # Return a proper rotation matrix with det(R)=+1


def is_valid_transform(
    T: np.ndarray, tol: float = 5e-3
) -> bool:  # Validate whether T is a proper 4x4 rigid transform
    if T.shape != (4, 4):  # Check matrix is 4x4
        return False  # Not a valid transform if wrong shape
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):  # Bottom row must be [0 0 0 1]
        return False  # Reject if bottom row does not match homogeneous transform rule
    R = T[:3, :3]  # Extract rotation part
    if not np.allclose(
        R.T @ R, np.eye(3), atol=tol
    ):  # Rotation must be orthonormal: R^T R = I
        return False  # Reject if not orthonormal
    if not np.isclose(
        np.linalg.det(R), 1.0, atol=tol
    ):  # Rotation must have determinant +1
        return False  # Reject if determinant not close to +1
    return True  # Passed all checks


def origin_of(
    T: np.ndarray,
) -> np.ndarray:  # Extract the translation component (origin) from a transform
    return T[:3, 3].copy()  # Return a copy of the xyz translation vector


def axes_of(
    T: np.ndarray,
) -> np.ndarray:  # Extract the rotation axes (the 3x3 rotation block) from a transform
    return T[:3, :3].copy()  # Return a copy of the rotation matrix


def rotation_matrix_from_z_to_vec(
    v: np.ndarray,
) -> np.ndarray:  # Compute R such that R*[0,0,1] aligns with vector v
    v = np.asarray(v, dtype=float)  # Ensure v is a float numpy array
    n = np.linalg.norm(v)  # Compute magnitude of v
    if n < 1e-12:  # If v is near-zero, direction is undefined
        return np.eye(3, dtype=float)  # Return identity rotation
    vhat = v / n  # Normalize v to unit direction

    z = np.array([0.0, 0.0, 1.0], dtype=float)  # Define the canonical +Z unit vector
    c = float(np.dot(z, vhat))  # Cosine of angle between z and vhat

    if abs(c - 1.0) < 1e-10:  # If vectors are nearly identical (+Z already aligned)
        return np.eye(3, dtype=float)  # No rotation needed

    if abs(c + 1.0) < 1e-10:  # If vectors are opposite (vhat ~ -Z)
        return rot_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=float), math.pi
        )  # Rotate 180° about X (any axis ⟂ Z works)

    axis = np.cross(z, vhat)  # Rotation axis is perpendicular to both z and vhat
    s = np.linalg.norm(axis)  # Sine magnitude (|cross|) gives sin(theta)
    axis = axis / (s + 1e-12)  # Normalize axis (avoid divide-by-zero with epsilon)
    theta = math.atan2(s, c)  # Robust angle from sin/cos using atan2
    return rot_axis_angle(axis, theta)  # Build rotation matrix from axis-angle
