"""Zarr-based replay buffer for storing and loading temporal robot demonstration data."""

import logging
import math
import numbers
import os
from functools import cached_property
from typing import Dict, Optional, Union

import numcodecs
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def check_chunks_compatible(chunks: tuple, shape: tuple):
    """Validate that chunk sizes are compatible with an array shape.

    :param chunks: Chunk sizes, one per dimension.
    :param shape: Array shape to validate against.
    :raises ValueError: If lengths differ or any chunk value is not a positive integer.
    """
    if len(shape) != len(chunks):
        raise ValueError(
            f"chunks length {len(chunks)} does not match shape length {len(shape)}."
        )
    for c in chunks:
        if not isinstance(c, numbers.Integral):
            raise ValueError(f"Chunk value {c!r} is not an integer.")
        if c <= 0:
            raise ValueError(f"Chunk value {c} must be positive.")


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    """Rechunk and/or recompress a Zarr array in place.

    :param group: Zarr group containing the array.
    :param name: Key of the array within group.
    :param chunks: New chunk shape. Defaults to the array's current chunks.
    :param chunk_length: If chunks is None, set the first-dimension chunk length
        to this value and keep remaining dimensions unchanged.
    :param compressor: New compressor. Defaults to the array's current compressor.
    :param tmp_key: Temporary key used during the copy; must not already exist in group.
    :return: The rechunked/recompressed Zarr array.
    """
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """Compute chunk sizes that keep each chunk close to target_chunk_bytes.

    Suitable for common array shapes: ``(T, D)``, ``(T, N, D)``,
    ``(T, H, W, C)``, ``(T, N, H, W, C)``.

    :param shape: Array shape.
    :param dtype: Array dtype.
    :param target_chunk_bytes: Target uncompressed bytes per chunk. Default 2 MB.
    :param max_chunk_length: Maximum first-dimension chunk length.
    :return: Chunk shape tuple.
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """Zarr-backed temporal data store for robot demonstrations.

    The first dimension of every stored array is time. Data is chunked only
    along the time dimension. Use the ``create_*`` or ``copy_from_*`` class
    methods to construct instances.
    """

    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """Construct a ReplayBuffer from an existing storage root.

        Prefer the ``create_*`` and ``copy_from_*`` class methods over calling
        this constructor directly.

        :param root: Storage backend — either a :class:`zarr.Group` (on-disk or
            in-memory) or a plain dict with ``'data'`` and ``'meta'`` keys.
        :raises ValueError: If root is missing required keys or data arrays are
            inconsistent with ``episode_ends``.
        """
        if "data" not in root:
            raise ValueError("root must contain a 'data' group.")
        if "meta" not in root:
            raise ValueError("root must contain a 'meta' group.")
        if "episode_ends" not in root["meta"]:
            raise ValueError("root['meta'] must contain 'episode_ends'.")
        for key, value in root["data"].items():
            if value.shape[0] != root["meta"]["episode_ends"][-1]:
                raise ValueError(
                    f"Data array '{key}' length {value.shape[0]} does not match "
                    f"episode_ends[-1] {root['meta']['episode_ends'][-1]}."
                )
        self.root = root

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        """Create an empty in-memory (or custom-store) Zarr-backed ReplayBuffer.

        :param storage: Zarr storage backend. Defaults to :class:`zarr.MemoryStore`.
        :param root: Existing Zarr group to use as root. Overrides storage if provided.
        :return: Empty :class:`ReplayBuffer` instance.
        """
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            episode_ends = meta.zeros(
                "episode_ends",
                shape=(0,),
                dtype=np.int64,
                compressor=None,
                overwrite=False,
            )
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        """Create an empty NumPy-backed (in-memory) ReplayBuffer.

        :return: Empty :class:`ReplayBuffer` instance backed by plain NumPy arrays.
        """
        root = {
            "data": dict(),
            "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)},
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        """Create a ReplayBuffer from an existing Zarr group.

        :param group: Zarr group. If it does not yet contain a ``'data'`` key,
            an empty buffer is initialised inside it.
        :return: :class:`ReplayBuffer` instance.
        """
        if "data" not in group:
            # initialise empty buffer in the group
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exists
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode="r", **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(
        cls,
        src_store,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = None,
        compressors: Union[dict, str, numcodecs.abc.Codec] = None,
        if_exists="replace",
        **kwargs,
    ):
        """Copy a Zarr store into a new ReplayBuffer.

        :param src_store: Source Zarr store to copy from.
        :param store: Destination Zarr store. If ``None``, loads into NumPy arrays
            (numpy backend).
        :param keys: Data keys to copy. Defaults to all keys.
        :param chunks: Per-key chunk overrides.
        :param compressors: Per-key compressor overrides.
        :param if_exists: Behaviour when a destination array already exists.
        :return: :class:`ReplayBuffer` backed by the destination store.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root["meta"].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root["data"].keys()
            data = dict()
            for key in keys:
                arr = src_root["data"][key]
                data[key] = arr[:]

            root = {"meta": meta, "data": data}
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=src_store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value
                )
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        backend=None,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = None,
        compressors: Union[dict, str, numcodecs.abc.Codec] = None,
        if_exists="replace",
        **kwargs,
    ):
        """Copy an on-disk Zarr dataset to an in-memory ReplayBuffer. (Recommended.)

        :param zarr_path: Path to the on-disk Zarr dataset.
        :param backend: Deprecated. Pass ``None`` (default).
        :param store: Destination Zarr store. If ``None``, uses an in-memory store.
        :param keys: Data keys to copy. Defaults to all keys.
        :param chunks: Per-key chunk overrides.
        :param compressors: Per-key compressor overrides.
        :param if_exists: Behaviour when a destination array already exists.
        :return: :class:`ReplayBuffer` loaded into memory.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        if backend == "numpy":
            logger.warning("backend argument is deprecated!")
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), "r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    # ============= save methods ===============
    def save_to_store(
        self,
        store,
        chunks: Optional[Dict[str, tuple]] = None,
        compressors: Union[str, numcodecs.abc.Codec, dict] = None,
        if_exists="replace",
        **kwargs,
    ):
        """Serialize the ReplayBuffer to a Zarr store.

        :param store: Destination Zarr store.
        :param chunks: Per-key chunk overrides for data arrays.
        :param compressors: Per-key compressor overrides for data arrays.
        :param if_exists: Behaviour when a destination array already exists.
        :return: The destination store.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        root = zarr.group(store)
        if self.backend == "zarr":
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            # save meta, no chunking
            for key, value in self.root["meta"].items():
                _ = meta_group.array(
                    name=key, data=value, shape=value.shape, chunks=value.shape
                )

        # save data, chunk
        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
            else:
                # numpy
                _ = data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: Optional[Dict[str, tuple]] = None,
        compressors: Union[str, numcodecs.abc.Codec, dict] = None,
        if_exists="replace",
        **kwargs,
    ):
        """Serialize the ReplayBuffer to an on-disk Zarr directory.

        :param zarr_path: Output path for the Zarr directory store.
        :param chunks: Per-key chunk overrides for data arrays.
        :param compressors: Per-key compressor overrides for data arrays.
        :param if_exists: Behaviour when a destination array already exists.
        :return: The destination store.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(compressor="default"):
        """Resolve a compressor shorthand string to a numcodecs compressor object.

        :param compressor: ``'default'`` (lz4), ``'disk'`` (zstd), ``None`` (no
            compression), or an existing compressor object (returned as-is).
        :return: Resolved compressor instance or ``None``.
        """
        if compressor == "default":
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
            )
        elif compressor == "disk":
            compressor = numcodecs.Blosc(
                "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
        return compressor

    @classmethod
    def _resolve_array_compressor(
        cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array
    ):
        """Resolve the compressor to use for a specific array key.

        :param compressors: Dict of per-key compressors, a single compressor string,
            or a compressor object applied to all keys.
        :param key: Data key being resolved.
        :param array: The array (used to fall back to its current compressor).
        :return: Resolved compressor instance or ``None``.
        """
        # allows compressor to be explicitly set to None
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array):
        """Resolve the chunk shape to use for a specific array key.

        :param chunks: Dict of per-key chunk shapes or a single tuple applied to
            all keys.
        :param key: Data key being resolved.
        :param array: The array (used to fall back to its current chunks or
            compute optimal chunks).
        :return: Chunk shape tuple.
        :raises TypeError: If chunks is not a dict or tuple.
        """
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        """Write or overwrite entries in the metadata group.

        :param data: Dict of metadata arrays or scalars to store.
        :raises TypeError: If a value cannot be converted to a NumPy array.
        :return: The updated metadata group.
        """
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    overwrite=True,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    def get_episode_idxs(self):
        """Return a per-step array of episode indices.

        :return: Integer array of shape ``(n_steps,)`` where each entry is the
            episode index that step belongs to.
        """
        if len(self.episode_ends) == 0:
            return np.zeros((0,), dtype=np.int64)

        import numba

        @numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    @property
    def backend(self):
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(
        self,
        data: Dict[str, np.ndarray],
        chunks: Optional[Dict[str, tuple]] = None,
        compressors: Union[str, numcodecs.abc.Codec, dict] = None,
    ):
        """Append one episode to the buffer.

        :param data: Dict mapping array names to NumPy arrays of shape
            ``(episode_length, ...)``. All arrays must share the same first dimension.
        :param chunks: Per-key chunk overrides for newly created arrays.
        :param compressors: Per-key compressor overrides for newly created arrays.
        :raises ValueError: If data is empty, any array has fewer than 1 dimension,
            or arrays have inconsistent first-dimension lengths.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        if len(data) == 0:
            raise ValueError("data must contain at least one array.")
        is_zarr = self.backend == "zarr"

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            if len(value.shape) < 1:
                raise ValueError(f"Array '{key}' must have at least 1 dimension.")
            if episode_length is None:
                episode_length = len(value)
            elif episode_length != len(value):
                raise ValueError(
                    f"Array '{key}' length {len(value)} does not match "
                    f"episode length {episode_length}."
                )
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value
                    )
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    arr = self.data.zeros(
                        name=key,
                        shape=new_shape,
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr,
                    )
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                if value.shape[1:] != arr.shape[1:]:
                    raise ValueError(
                        f"Array '{key}' trailing shape {value.shape[1:]} does not match "
                        f"existing shape {arr.shape[1:]}."
                    )
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0] :] = value

        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(
                    self.meta,
                    "episode_ends",
                    chunk_length=int(episode_ends.shape[0] * 1.5),
                )

    def drop_episode(self):
        """Remove the last episode from the buffer.

        :raises ValueError: If the buffer contains no episodes.
        """
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        if len(episode_ends) == 0:
            raise ValueError("Cannot drop an episode from an empty buffer.")
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self):
        """Remove and return the last episode from the buffer.

        :raises ValueError: If the buffer contains no episodes.
        :return: Dict of arrays for the removed episode.
        """
        if self.n_episodes == 0:
            raise ValueError("Cannot pop an episode from an empty buffer.")
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        """Alias for :meth:`add_episode`.

        :param data: Dict mapping array names to NumPy arrays.
        """
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        """Return all steps for one episode.

        :param idx: Episode index. Supports negative indexing.
        :param copy: If ``True``, return copies of NumPy arrays.
        :return: Dict mapping array names to slices of the stored data.
        """
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx):
        """Return the slice that selects all steps for one episode.

        :param idx: Episode index (non-negative).
        :return: :class:`slice` ``(start, end)`` into the time axis.
        """
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """Return a slice of steps across all data arrays.

        :param start: Start index (inclusive).
        :param stop: Stop index (exclusive).
        :param step: Step size. Defaults to ``None`` (contiguous).
        :param copy: If ``True``, return copies of NumPy arrays.
        :return: Dict mapping array names to the requested slice.
        """
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    # =========== chunking =============
    def get_chunks(self) -> dict:
        """Return the current chunk shape for every data array.

        :raises RuntimeError: If the backend is not ``'zarr'``.
        :return: Dict mapping array names to chunk shape tuples.
        """
        if self.backend != "zarr":
            raise RuntimeError("get_chunks is only supported for the zarr backend.")
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        """Rechunk data arrays to new chunk shapes.

        :param chunks: Dict mapping array names to desired chunk shape tuples.
        :raises RuntimeError: If the backend is not ``'zarr'``.
        """
        if self.backend != "zarr":
            raise RuntimeError("set_chunks is only supported for the zarr backend.")
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        """Return the current compressor for every data array.

        :raises RuntimeError: If the backend is not ``'zarr'``.
        :return: Dict mapping array names to compressor instances.
        """
        if self.backend != "zarr":
            raise RuntimeError(
                "get_compressors is only supported for the zarr backend."
            )
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        """Recompress data arrays with new compressors.

        :param compressors: Dict mapping array names to compressor strings or objects.
        :raises RuntimeError: If the backend is not ``'zarr'``.
        """
        if self.backend != "zarr":
            raise RuntimeError(
                "set_compressors is only supported for the zarr backend."
            )
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
