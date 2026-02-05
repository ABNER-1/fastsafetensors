# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any, List, Optional

from .common import SafeTensorsMetadata, init_logger
from .frameworks import get_framework_op
from .parallel_loader import PipelineParallel
from .loader import BaseSafeTensorsFileLoader, loaded_library
from fastsafetensor_3fs_reader import ThreeFSFileReader, extract_mount_point
from . import cpp as fstcpp

logger = init_logger(__name__)

class ThreeFSLoader(BaseSafeTensorsFileLoader):
    """Load .safetensors files using 3FS USRBIO for high-performance I/O.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed loading.
        device (str): Target device where tensors will be loaded (CPU, CUDA, etc.).
        mount_point (str): 3FS mount point path (e.g., "/mnt/3fs").
        debug_log (bool): Enable detailed debug logging.
        disable_cache (bool): Whether to disable caching of loaded tensors.
        framework (str): Deep learning framework to use ("pytorch" or "paddle").
        **kwargs: Additional arguments passed to BaseSafeTensorsFileLoader.

    Examples:
        >>> from fastsafetensors.threefs_loader import ThreeFSLoader
        >>> loader = ThreeFSLoader(None, device="cuda:0", mount_point="/mnt/3fs")
        >>> loader.add_filenames({0: ["/mnt/3fs/model.safetensors"]})
        >>> bufs = loader.copy_files_to_device()
        >>> tensor = bufs.get_tensor("weight")
        >>> loader.close()
    """

    def __init__(
        self,
        pg: Optional[Any],
        device: str = "cpu",
        mount_point: str = "/mnt/3fs",
        debug_log: bool = False,
        disable_cache: bool = True,
        framework: str = "pytorch",
        metadata_cache: Optional[dict] = None,
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = self.framework.get_device(device, self.pg)

        global loaded_library
        if not loaded_library:
            fstcpp.load_library_functions()
            loaded_library = True
        super().__init__(
            pg,
            self.device,
            copier_type="3fs",
            set_numa=True,
            debug_log=debug_log,
            disable_cache=disable_cache,
            framework=framework,
            metadata_cache=metadata_cache,
            mount_point=mount_point,
            **kwargs,
        )

class ParallelThreeFSLoader(PipelineParallel):
    """Parallel loader for .safetensors files using 3FS USRBIO.

    This class provides pipeline-parallel loading of multiple safetensors files
    using 3FS for high-performance I/O operations.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed operations.
        hf_weights_files (List[str]): List of safetensors files to load from 3FS.
        mount_point (str): 3FS mount point path (e.g., "/mnt/3fs").
        max_concurrent_producers (int): Maximum number of concurrent producer threads.
        queue_size (int): Size of the queue for buffering loaded file batches.
                         Default 0 for unbuffered behavior.
        use_tqdm_on_load (bool): Enable progress bar during loading.
        device (str): Target device for tensor loading.
        debug_log (bool): Enable debug logs.
        framework (str): Framework to use for tensor operations ("pytorch" or "paddle").
        **kwargs: Additional arguments passed to the loader.

    Examples:
        >>> from fastsafetensors.threefs_loader import ParallelThreeFSLoader
        >>> files = ["/mnt/3fs/model-00001.safetensors", "/mnt/3fs/model-00002.safetensors"]
        >>> loader = ParallelThreeFSLoader(
        ...     pg=None,
        ...     hf_weights_files=files,
        ...     mount_point="/mnt/3fs",
        ...     device="cuda:0"
        ... )
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        pg: Optional[Any],
        hf_weights_files: List[str],
        max_concurrent_producers: int = 1,
        queue_size: int = 0,
        use_tqdm_on_load: bool = True,
        device: str = "cpu",
        debug_log: bool = False,
        framework: str = "pytorch",
        pre_open_files: bool = True,
        lazy_broadcast: bool = True,
        **kwargs,
    ):
        t_total_start = time.time()

        # Timing accumulators (ms), default 0 for steps skipped when pre_open_files=False
        t_read_headers = 0.0
        t_parse_headers = 0.0
        t_inject_cache = 0.0

        metadata_cache: dict = {}
        self._reader: Optional[ThreeFSFileReader] = None
        mount_point: str = extract_mount_point(hf_weights_files[0])

        # Step 1: Create ThreeFSLoader
        t0 = time.time()
        loader = ThreeFSLoader(
            pg,
            device=device,
            mount_point=mount_point,
            disable_cache=True,
            debug_log=debug_log,
            framework=framework,
            metadata_cache=metadata_cache,
            **kwargs,
        )
        t_create_loader = (time.time() - t0) * 1000

        # Step 2: Get reader reference
        t0 = time.time()
        self._reader = getattr(loader.copier_constructor, 'reader', None)
        t_get_reader = (time.time() - t0) * 1000

        if pre_open_files and self._reader is not None:
            framework_op = get_framework_op(framework)

            # Step 3: Batch open + read headers (C++ thread pool)
            t0 = time.time()
            header_results = self._reader.read_headers_batch(hf_weights_files)
            t_read_headers = (time.time() - t0) * 1000

            # Step 4: Parse headers → SafeTensorsMetadata
            t0 = time.time()
            for filepath, (header_string, header_length, file_size) in header_results.items():
                try:
                    metadata_cache[filepath] = SafeTensorsMetadata.from_header_bytes(
                        header_string, header_length, file_size, filepath, framework_op
                    )
                except Exception as exc:
                    logger.warning(
                        "from_header_bytes failed for %s: %s, will load on demand",
                        filepath, exc,
                    )
            t_parse_headers = (time.time() - t0) * 1000

            # Step 5: Inject metadata_cache into loader
            t0 = time.time()
            loader._metadata_cache.update(metadata_cache)
            t_inject_cache = (time.time() - t0) * 1000

        # Step 6: PipelineParallel.__init__
        t0 = time.time()
        super().__init__(
            pg,
            loader,
            hf_weights_files,
            max_concurrent_producers,
            queue_size,
            use_tqdm_on_load,
            lazy_broadcast=lazy_broadcast,
            **kwargs,
        )
        t_pipeline_init = (time.time() - t0) * 1000

        t_total = (time.time() - t_total_start) * 1000
        logger.info(
            "ParallelThreeFSLoader.__init__: total=%.3fms | "
            "create_loader=%.3fms, get_reader=%.3fms, "
            "read_headers=%.3fms, parse_headers=%.3fms, "
            "inject_cache=%.3fms, pipeline_init=%.3fms, "
            "files=%d",
            t_total, t_create_loader, t_get_reader,
            t_read_headers, t_parse_headers,
            t_inject_cache, t_pipeline_init,
            len(metadata_cache),
        )

    def close(self):
        if self._reader is not None:
            self._reader.close()
        super().close()


__all__ = [
    "ThreeFSLoader",
    "ParallelThreeFSLoader",
]
