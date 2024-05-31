# From https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/packed_dataset.py


import json
import os
import random
import struct

import numpy as np
import torch
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes

class PackedDataset(IterableDataset):
    def __init__(
        self,
        filenames,
        n_chunks,
        block_size,
        seed=12345,
        shuffle=True,
        wrap=False,
        num_processes=1,
        process_rank=0,
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __str__(self) -> str:
        filename_str = (
            f"{','.join(x.name for x in self._filenames[:5])} ..... "
            + f"{','.join(x.name for x in self._filenames[-5:])}"
        )
        content = json.dumps(
            {
                "seed": self._seed,
                "shuffle": self._shuffle,
                "block_size": self._block_size,
                "wrap": self._wrap,
                "n_chunks": self._n_chunks,
                "num_processes": self._num_processes,
                "process_rank": self._process_rank,
                "file_names": filename_str,
            }
        )
        return f"PackedDataset({content})"

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]
        logger.info(f"There are {len(filenames)} files to be read by {worker_id=}")
        logger.debug(f"Files to be load: {[x.name for x in filenames]}")

        dataset_iterator = PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )
        logger.debug(f"PackedDataset: {self}")
        logger.debug(f"PackedDatasetIterator: {dataset_iterator}")
        return dataset_iterator


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

        total_token_num = len(self._filenames) * self._n_blocks * self._block_size
        logger.info(
            f"Total tokens to be load: {total_token_num/(1024**3)} Billion with "
            + f"file_num={len(self._filenames)}, block_per_file={self._n_blocks}, "
            + f"block_size={self._block_size}"
        )

    def __str__(self) -> str:
        filename_str = (
            f"{','.join(x.name for x in self._filenames[:5])} ..... "
            + f"{','.join(x.name for x in self._filenames[-5:])}"
        )
        block_idxs = self._block_idxs.tolist()
        block_idx_str = (
            f"{','.join([str(x) for x in block_idxs[:10]])} ..... "
            + f"{','.join([str(x) for x in block_idxs[-10:]])}"
        )

        content = json.dumps(
            {
                "seed": self._seed,
                "shuffle": self._shuffle,
                "dtype": str(self._dtype),
                "block_size": self._block_size,
                "wrap": self._wrap,
                "n_blocks": self._n_blocks,
                "cur_block_idx": self._curr_idx,
                "block_idxs": block_idx_str,
                "n_chunks": self._n_chunks,
                "file_idx": self._file_idx,
                "filenames": filename_str,
            }
        )
        return f"PackedDatasetIterator({content})"

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert (1,) == version
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            if not self._wrap:
                raise StopIteration
            else:
                self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = (
            self._rng.permutation(n_all_blocks)
            if self._shuffle
            else range(n_all_blocks)
        )

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(
            buffer, dtype=self._dtype, count=self._block_size, offset=offset
        )
        self._curr_idx += 1
        input_ids = torch.from_numpy(arr.astype(np.int64))[:-1]
        return {"input_ids": input_ids, "labels": input_ids}


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __str__(self) -> str:
        content = json.dumps(
            {
                "seed": self._seed,
                "weights": self._weights,
                "datasets": "\n".join([str(x) for x in self._datasets]),
            }
        )
        return f"CombinedDataset({content})"

    def __iter__(self):
        logger.debug(f"CombinedDataset: {self}")
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        item = next(dataset)
        # logger.debug(f"{self._datasets.index(dataset)=}, {item=}")
        return item


class PackedDatasetBuilder(object):
    def __init__(
        self,
        outdir,
        prefix,
        chunk_size,
        sep_token,
        dtype="auto",
        vocab_size=None,
    ):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()
