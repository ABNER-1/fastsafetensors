# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .common import init_logger
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType
from .tensor_factory import LazyTensorFactory

logger = init_logger(__name__)


class FilesBufferOnDevice:
    r"""Device buffer for .safetensors files.
        Users can call get_tensor(), get_sharded(), etc. to instantiate (sharded) tensors from the device buffer.
        Note that for multi-process loading, users must follow the single-program multiple-data (SPMD) paradigm, which is common for torch.distributed programs.
        In other words, users must ensure that every worker process calls the methods here in the same order.
        This is because methods here reuse torch.distributed operations: broadcast, scatter, recv, and send.
        They synchornously wait all the workers to execute copies among processes.

        Users should create this instance with SafeTensorsFileLoader.submit_io().

    Args:
        rank_loaders (Dict<rank, list(LazyTensorFacotry)>): Tensor factories per rank, which hold device pointers for buffers.
        pg (ProcessGroupBase): process group for calling distributed ops.
        auto_mem_delete (bool): automatically release device buffers when all the tensors are shuffled.

    Examples:
        See examples/run_single.py and examples/run_parallel.py.
    """

    def __init__(
        self,
        rank_loaders: Dict[int, List[LazyTensorFactory]],
        pg: ProcessGroupBase,
        framework: FrameworkOpBase,
        auto_mem_delete: bool = True,
    ):
        self.framework = framework
        self.rank_loaders: Dict[int, List[LazyTensorFactory]] = rank_loaders
        self.key_to_rank_lidx: Dict[str, Tuple[int, int]] = {}
        self.instantiated: Dict[int, Dict[int, Dict[str, bool]]] = {}  # rank, key name
        for rank, loaders in rank_loaders.items():
            self.instantiated[rank] = {}
            for lidx, loader in enumerate(loaders):
                for key in loader.metadata.tensors.keys():
                    if key in self.key_to_rank_lidx:
                        raise Exception(
                            f"FilesBufferOnDevice: key {key} must be unique among files"
                        )
                    self.key_to_rank_lidx[key] = (rank, lidx)
                self.instantiated[rank][lidx] = {}
        self.pg = pg
        self.auto_mem_delete = auto_mem_delete and self.pg.size() > 1

    def broadcast_all_files(self) -> None:
        """Broadcast all file buffers at once using file-level broadcast.

        Instead of broadcasting each tensor individually (which requires N separate
        memory allocations and N broadcast calls), this method broadcasts each file's
        entire data buffer in a single operation, then uses dlpack to zero-copy split
        individual tensors from the received buffer.

        This must be called by all ranks in the process group in the same order (SPMD).
        After calling this method, subsequent get_tensor() calls will find tensors
        already available locally and skip per-tensor broadcast.
        """
        if self.pg.size() <= 1:
            return

        for rank, loaders in sorted(self.rank_loaders.items()):
            for loader in loaders:
                loader.broadcast_file_buffer(self.pg)

    def ensure_file_broadcasted(self, rank: int, lidx: int) -> None:
        """Broadcast a single file buffer on demand.

        This is the lazy counterpart of broadcast_all_files(). It broadcasts
        only the file identified by (rank, lidx) if it has not been broadcast
        yet. The underlying broadcast_file_buffer() is idempotent, so calling
        this multiple times for the same file is safe.

        All ranks must call this for the same (rank, lidx) in the same order
        to satisfy SPMD constraints of collective communication.
        """
        if self.pg.size() <= 1:
            return
        self.rank_loaders[rank][lidx].broadcast_file_buffer(self.pg)

    def get_keys_grouped_by_file(self) -> List[str]:
        """Return tensor keys grouped by their source file.

        Keys belonging to the same file (rank, lidx) are placed consecutively.
        Files are ordered by (rank, lidx) ascending, matching the iteration
        order used by broadcast_all_files() to guarantee SPMD safety.

        This ordering ensures that when combined with ensure_file_broadcasted(),
        each file's buffer can be broadcast just before its tensors are consumed
        and freed immediately after, keeping peak GPU memory overhead to a
        single file buffer at a time.
        """
        groups: Dict[Tuple[int, int], List[str]] = {}
        for key, (rank, lidx) in self.key_to_rank_lidx.items():
            groups.setdefault((rank, lidx), []).append(key)
        ordered_keys: List[str] = []
        for file_id in sorted(groups.keys()):
            ordered_keys.extend(groups[file_id])
        return ordered_keys

    def close(self):
        for _, loaders in self.rank_loaders.items():
            for loader in loaders:
                loader.free_dev_ptrs()
        self.rank_loaders = {}

    def get_filename(self, tensor_name: str) -> str:
        if tensor_name not in self.key_to_rank_lidx:
            return ""
        rank, lidx = self.key_to_rank_lidx[tensor_name]
        return self.rank_loaders[rank][lidx].metadata.src

    def get_shape(self, tensor_name: str) -> List[int]:
        rank, lidx = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.tensors[tensor_name].shape

    def _get_rank_lidx(self, tensor_name: str) -> Tuple[int, int]:
        if tensor_name not in self.key_to_rank_lidx:
            raise ValueError(f"_get_rank: key {tensor_name} was not found in files")
        return self.key_to_rank_lidx[tensor_name]

    def _get_tensor(
        self,
        rank: int,
        lidx: int,
        tensor_name: str,
        ret: TensorBase,
        device: Optional[Device],
        dtype: DType,
    ) -> TensorBase:
        loader = self.rank_loaders[rank][lidx]
        if self.auto_mem_delete:
            self.instantiated[rank][lidx][tensor_name] = True
            if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                if self.pg.rank() == rank:
                    logger.debug(
                        "_get_tensor: free_dev_ptrs, lidx=%d, src=%s",
                        lidx,
                        loader.metadata.src,
                    )
                loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def get_sharded_wrapped(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        rank, lidix = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle(self.pg, tensor_name, dim)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_sharded(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        partition a tensor instance with the key tensor_name at the dimension dim and return it.
        In multi-process loading, this eventually calls torch.distributed.scatter.
        A special dim is -1, which broadcast a tensor to all the ranks (== get_tensor()).
        """
        return self.get_sharded_wrapped(tensor_name, dim, device, dtype).get_raw()

    def get_tensor_wrapped(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        return self.get_sharded_wrapped(tensor_name, -1, device, dtype)

    def get_tensor(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        get a tensor instance with the key tensor_name from a local or remote rank.
        In multi-process loading, this eventually calls torch.distributed.broadcast.
        So, every rank will allocate the same tensor at each device memroy.
        In single-process loading, this directly instantiates a tensor from the device buffer with zero copy.
        """
        return self.get_tensor_wrapped(tensor_name, device, dtype).get_raw()

    def push_tensor(
        self,
        tensor_name: str,
        dst_rank: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Optional[Any]:
        """
        push a tensor instance with the key tensor_name from a rank to a destination rank dst_rank.
        In multi-process loading, this eventually calls torch.distributed.send if the rank has the tensor instance.
        The destination rank will call torch.distributed.recv.
        Other ranks do nothing.
        """
        rank, lidix = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].push(self.pg, tensor_name, dst_rank, rank)
        if t:
            return self._get_tensor(
                rank, lidix, tensor_name, t, device, dtype
            ).get_raw()
        return None

    def get_multi_cols(
        self,
        tensor_names: List[str],
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        rank_lidixs: Dict[Tuple[int, int], List[str]] = {}
        for tensor_name in tensor_names:
            ranklidx = self._get_rank_lidx(tensor_name)
            if ranklidx in rank_lidixs:
                rank_lidixs[ranklidx].append(tensor_name)
            else:
                rank_lidixs[ranklidx] = [tensor_name]
        ts: List[TensorBase] = []
        for (rank, lidix), tns in sorted(rank_lidixs.items(), key=lambda x: x[0]):
            ts.append(
                self.rank_loaders[rank][lidix].shuffle_multi_cols(self.pg, tns, dim)
            )
        if len(ts) == 1:
            # fastpath: tensors at the same layer are often in the same file
            return self._get_tensor(
                rank, lidix, rank_lidixs[(rank, lidix)][0], ts[0], device, dtype
            )
        ret = self.framework.concat_tensors(ts, dim=dim)
        if self.auto_mem_delete:
            for tensor_name in tensor_names:
                rank, lidx = self._get_rank_lidx(tensor_name)
                loader = self.rank_loaders[rank][lidix]
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "get_multi_cols: free_dev_ptrs, rank=%d, lidx=%d, src=%s",
                            rank,
                            lidx,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def as_dict(self, tensor_shard_dim: OrderedDict[str, int]) -> Dict[str, TensorBase]:
        tensors: Dict[str, TensorBase] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            rank, lidx = self._get_rank_lidx(tensor_name)
            loader = self.rank_loaders[rank][lidx]
            tensors[tensor_name] = loader.shuffle(self.pg, tensor_name, dim)
            if self.auto_mem_delete:
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "as_dict: free_dev_ptrs, rank=%d, src=%s",
                            rank,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        if self.auto_mem_delete:
            self.rank_loaders = {}
        return tensors
