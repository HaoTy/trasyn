from math import pi

import numpy as np
from numpy.typing import NDArray


def h() -> NDArray[np.float64]:
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def s() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, 1j]])


def t() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * pi / 4)]])


def x() -> NDArray[np.float64]:
    return np.array([[0, 1], [1, 0]])


def y() -> NDArray[np.complex128]:
    return np.array([[0, -1j], [1j, 0]])


def z() -> NDArray[np.float64]:
    return np.array([[1, 0], [0, -1]])


def w() -> NDArray[np.complex128]:
    return np.eye(2) * np.exp(1j * pi / 4)


def i() -> NDArray[np.float64]:
    return np.eye(2)


def rx(theta: float) -> NDArray[np.complex128]:
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def ry(theta: float) -> NDArray[np.float64]:
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def rz(theta: float) -> NDArray[np.complex128]:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])


def u(theta: float, phi: float, lam: float) -> NDArray[np.complex128]:
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
        ]
    )


def u1(lam: float) -> NDArray[np.complex128]:
    return u(0, 0, lam)


def u2(phi: float, lam: float) -> NDArray[np.complex128]:
    return u(pi / 2, phi, lam)


def u3(theta: float, phi: float, lam: float) -> NDArray[np.complex128]:
    return u(theta, phi, lam)


def cx() -> NDArray[np.float64]:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
    )


GATES = {
    "h": h,
    "s": s,
    "t": t,
    "x": x,
    "y": y,
    "z": z,
    "w": w,
    "i": i,
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "u": u,
    "u1": u1,
    "u2": u2,
    "u3": u3,
}
