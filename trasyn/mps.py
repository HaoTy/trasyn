from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray


def _concatenate_mps(
    mps1: Sequence[NDArray[np.complex128]], mps2: Sequence[NDArray[np.complex128]]
) -> list[NDArray[np.complex128]]:
    if mps1[-1].shape[2] != 1:
        return list(mps1) + list(mps2)
    tsr = np.einsum("ipj,jpk->ik", mps1[-1], mps2[0])
    tsr = np.einsum("ij,jpk->ipk", tsr, mps2[1])
    return list(mps1[:-1]) + [tsr] + list(mps2[2:])


def _svd(
    tsr: NDArray[np.complex128],
    rank: int | None = None,
    absorb_s: Literal["left", "even", "right"] = "left",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    u, s, vh = np.linalg.svd(tsr, full_matrices=False)
    if rank is None:
        if (nonzero_indices := np.argwhere(np.isclose(s, 0))).size > 0:
            rank = nonzero_indices[0][0]
        else:
            rank = s.size
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    if absorb_s == "even":
        s **= 0.5
    if absorb_s != "right":
        u = u @ np.diag(s)
    if absorb_s != "left":
        vh = np.diag(s) @ vh
    return u, s, vh


def _swap_local(
    tsr1: NDArray[np.complex128],
    tsr2: NDArray[np.complex128],
    rank: int | None = None,
    absorb_s: Literal["left", "even", "right"] = "left",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    tsr = np.einsum("ipj,jqk->iqpk", tsr1, tsr2)
    shape = tsr.shape
    if rank == 0:
        rank = tsr1.shape[2]
    u, _, vh = _svd(tsr.reshape(np.prod(shape[:2]), -1), rank, absorb_s)
    return u.reshape(*shape[:2], -1), vh.reshape(-1, *shape[2:])


def _swap(
    mps: Sequence[NDArray[np.complex128]],
    start: int = -1,
    dest: int = 1,
    rank: int | None = None,
    absorb_s: Literal["left", "even", "right"] = "left",
) -> list[NDArray[np.complex128]]:
    mps = list(mps)
    if dest == start:
        return mps
    l = len(mps)
    if dest < 0:
        dest = l + dest
    if start < 0:
        start = l + start
    if dest > start:
        for i in range(start, dest):
            mps[i], mps[i + 1] = _swap_local(mps[i], mps[i + 1], rank, absorb_s)
    else:
        for i in range(start, dest, -1):
            mps[i - 1], mps[i] = _swap_local(mps[i - 1], mps[i], rank, absorb_s)
    return mps


def _trace_target_unitary(
    mps: Sequence[NDArray[np.complex128]],
    target_unitary: NDArray[np.complex128],
    rank: int | None = None,
    absorb_s: Literal["left", "even", "right"] = "left",
) -> list[NDArray[np.complex128]]:
    if len(mps) == 1:
        # pylint: disable=too-many-function-args
        return [
            np.einsum("ipj,ji->p", mps[0], target_unitary.T.conj()).reshape(1, -1, 1)
        ]
    mps = _swap(
        list(mps) + [target_unitary.T.conj().reshape(*target_unitary.shape, 1)],
        rank=rank,
        absorb_s=absorb_s,
    )
    # pylint: disable=too-many-function-args
    mps[1] = np.einsum("ipj,jik->pk", mps[0], mps[1]).reshape(1, mps[0].shape[1], -1)
    return mps[1:]


def _sample(
    mps: Sequence[NDArray[np.complex128]],
    num_samples: int,
    min_fixed_fraction: float = 0,
    max_fixed_fraction: float = 0,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.int64], float]:
    if cp is np:
        xp = np
    else:
        xp = cp.get_array_module(mps[0])
    mps = list(mps)
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    bitstrings = None
    projected_tsr = mps[0].reshape(-1, mps[0].shape[2])
    for tsr, split in zip(
        mps[1:],
        np.linspace(
            num_samples * min_fixed_fraction,
            num_samples * max_fixed_fraction,
            len(mps) - 1,
            dtype=int,
        ),
    ):
        if num_samples >= projected_tsr.shape[0]:
            indices = xp.arange(projected_tsr.shape[0])
        else:
            probs = xp.linalg.norm(projected_tsr, 2, axis=1)
            if split > 0:
                indices = xp.argpartition(probs, -int(split))[::-1]
                probs = probs[indices[split:]]
            probs -= probs.min()
            probs /= probs.max()
            probs = 1 / (1 - xp.exp(probs * 0.99 - 1))
            if split > 0:
                indices = xp.concatenate(
                    (
                        indices[:split],
                        indices[
                            rng.choice( # pending cupy issue #8293 to enable cupy equivalent
                                np.arange(split, projected_tsr.shape[0]),
                                size=num_samples - split,
                                replace=False,
                                p=asnumpy(probs / probs.sum()),
                                shuffle=False,
                            )
                        ],
                    )
                )
            else:
                indices = xp.asarray(
                    rng.choice(
                        projected_tsr.shape[0],
                        size=num_samples - split,
                        replace=False,
                        p=asnumpy(probs / probs.sum()),
                        shuffle=False,
                    )
                )
            del probs
            projected_tsr = projected_tsr[indices]
        if bitstrings is None:
            bitstrings = indices.reshape(-1, 1)
        else:
            indices, new_bits = xp.unravel_index(indices, shape)
            bitstrings = xp.concatenate(
                (bitstrings[indices], new_bits.reshape(-1, 1)), axis=1
            )
            del new_bits
        del indices
        shape = projected_tsr.shape[0], tsr.shape[1]
        # if not xp is np:
        #     xp.get_default_memory_pool().free_all_blocks() # Manual garbage collection causes serious slowdown
        # projected_tsr = xp.einsum("pj,jqk->pqk", projected_tsr, tsr).reshape( # einsum requires more memory and is slower
        # projected_tsr = (projected_tsr @ tsr.reshape(tsr.shape[0], -1)).reshape( # matmul should be the same as tensordot
        projected_tsr = xp.tensordot(projected_tsr, tsr, axes=1).reshape(
            -1, tsr.shape[2]
        )
    projected_tsr = xp.abs(projected_tsr.reshape(-1))
    argmax = xp.argmax(projected_tsr)
    if len(mps) != 1:
        indices, new_bits = xp.unravel_index(argmax, shape)
        return asnumpy(
            xp.concatenate((bitstrings[indices], new_bits.reshape(-1)))
        ), float(projected_tsr[argmax])
    return np.array([int(argmax)]), float(projected_tsr[argmax])


@lru_cache(None)
def _basis_state_arr(dim: int, index: int) -> NDArray[np.complex128]:
    arr = np.zeros(dim)
    arr[index] = 1
    return arr


def _apply(
    mps: Sequence[NDArray[np.complex128]],
    bitstring: Sequence[int | None],
) -> list[NDArray[np.complex128]]:
    ret = []
    for tsr, bit in zip(mps, bitstring):
        if bit is None:
            ret.append(tsr)
            continue
        ret.append(
            # pylint: disable=too-many-function-args
            np.einsum(
                "ipj,p->ij",
                tsr,
                _basis_state_arr(tsr.shape[1], bit),
            ).reshape(tsr.shape[0], 1, tsr.shape[2])
        )
    return ret


def _to_dense(mps: Sequence[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    tsr = mps[0]
    for next_tsr in mps[1:]:
        # pylint: disable=too-many-function-args
        tsr = np.einsum("ipj,jqk->ipqk", tsr, next_tsr).reshape(
            tsr.shape[0], -1, next_tsr.shape[2]
        )
    return tsr
