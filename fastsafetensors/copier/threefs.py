# SPDX-License-Identifier: Apache-2.0

"""
3FS USRBIO Copier Implementation

This module provides a copier implementation using DeepSeek AI's 3FS USRBIO
for high-performance file loading in fastsafetensors.

3FS USRBIO provides:
- Zero-copy I/O through shared memory regions (Iov)
- Asynchronous I/O via submit/wait pattern (Ior)
- High throughput for distributed filesystem access

Usage:
    from fastsafetensors.copier.threefs import new_threefs_file_copier

    # Create copier factory
    copier_fn = new_threefs_file_copier(
        device=Device("cuda:0"),
        mount_point="/mnt/3fs",
        entries=64,
        buffer_size=1 * 1024 * 1024 * 1024,
    )

    # Use with SafeTensorsFileLoader
    loader = SafeTensorsFileLoader(
        pg,
        device,
        copier_constructor=copier_fn,
        ...
    )
"""

from typing import Dict

# Import ThreeFSFileReader from the independent fastsafetensor_3fs_reader package
from fastsafetensor_3fs_reader import (
    ThreeFSFileReader,
)
from fastsafetensor_3fs_reader import is_available as _check_available

from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import SafeTensorsMetadata, init_logger
from fastsafetensors.copier.base import CopierInterface
from fastsafetensors.copier.registry import (
    CopierConstructFunc,
    register_copier_constructor,
)
from fastsafetensors.frameworks import FrameworkOpBase, TensorBase
from fastsafetensors.st_types import Device, DType

try:
    _USRBIO_AVAILABLE = _check_available()
except ImportError:
    _USRBIO_AVAILABLE = False

logger = init_logger(__name__)


class ThreeFSFileCopier(CopierInterface):
    """
    基于 3FS USRBIO 的文件 Copier 实现

    实现 CopierInterface 抽象，提供通过 3FS USRBIO 进行 safetensors 文件加载。

    工作流程:
    1. 使用 USRBIO Iov 进行零拷贝读取
    2. 使用 Ior 进行异步 I/O 提交
    3. 数据通过共享内存传输，无需 CPU 中转

    Args:
        metadata: Safetensors 元数据
        device: 目标设备
        reader: ThreeFSFileReader 实例
        framework: 框架操作接口
    """

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: ThreeFSFileReader,
        framework: FrameworkOpBase,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.reader = reader

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        """
        提交异步 I/O 请求

        Args:
            use_buf_register: 是否使用缓冲区注册 (3FS 不需要，忽略)
            max_copy_block_size: 最大复制块大小

        Returns:
            gds_device_buffer: 分配的目标设备缓冲区
        """
        # 计算读取范围
        offset = self.metadata.header_length
        length = self.metadata.size_bytes - self.metadata.header_length

        # 分配目标内存 (使用 framework)
        gbuf = self.framework.alloc_tensor_memory(length, self.device)

        # 使用 read_chunked 方法，fd 由 reader 内部管理
        logger.info(
            f"Reading {length} bytes from {self.metadata.src} using chunked read"
        )

        total_read = self.reader.read_chunked(
            path=self.metadata.src,
            dev_ptr=gbuf.get_base_address(),
            file_offset=offset,
            total_length=length,
            chunk_size=max_copy_block_size if max_copy_block_size > 0 else 0,
        )

        if total_read != length:
            raise Exception(
                f"ThreeFSFileCopier.submit_io: incomplete read, "
                f"expected={length}, actual={total_read}"
            )

        logger.info(f"Successfully read {total_read} bytes")

        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        """
        等待 I/O 完成并创建张量

        Args:
            gbuf: submit_io 返回的设备缓冲区
            dtype: 张量数据类型
            noalign: 是否跳过对齐检查

        Returns:
            Dict[str, TensorBase]: 张量字典
        """
        # read_chunked 是同步的，数据已在 submit_io 中完全读取完成
        # fd 由 reader 内部管理，无需手动关闭

        # 从缓冲区创建张量
        return self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )


@register_copier_constructor("3fs")
def new_threefs_file_copier(
    device: Device,
    mount_point: str,
    entries: int = 64,
    io_depth: int = 0,
    buffer_size: int = 64 * 1024 * 1024,
    **kwargs,
) -> CopierConstructFunc:
    """
    创建 3FS 文件 copier 工厂函数

    返回的工厂函数可作为 SafeTensorsFileLoader 的 copier_constructor 参数。

    Args:
        device: 目标设备
        mount_point: 3FS 挂载点路径
        entries: 最大并发 I/O 请求数
        io_depth: I/O 深度控制 (0=无限制, >0=批量阈值, <0=等待阈值)
        buffer_size: Iov 缓冲区大小 (bytes)

    Returns:
        工厂函数: (metadata, device, framework) -> CopierInterface

    Raises:
        ImportError: 如果 3FS USRBIO 库不可用

    Example:
        from fastsafetensors import SafeTensorsFileLoader
        from fastsafetensors.st_types import Device
        from fastsafetensors.copier.threefs import new_threefs_file_copier

        device = Device("cuda:0")
        copier_fn = new_threefs_file_copier(
            device=device,
            mount_point="/mnt/3fs",
            entries=128,
            buffer_size=4 * 1024 * 1024 * 1024,
        )

        loader = SafeTensorsFileLoader(
            pg=None,
            device=device.as_str(),
            copier_constructor=copier_fn,
        )
        loader.add_filenames({0: ["model.safetensors"]})
        bufs = loader.copy_files_to_device()
    """
    if not _USRBIO_AVAILABLE:
        raise ImportError(
            "3FS USRBIO C++ library is not available. "
            "Please install fastsafetensor-3fs-reader package: "
            "pip install fastsafetensor-3fs-reader"
        )

    # 创建 reader (跨文件复用)
    reader = ThreeFSFileReader(
        mount_point=mount_point,
        entries=entries,
        io_depth=io_depth,
        buffer_size=buffer_size,
    )

    def construct_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return ThreeFSFileCopier(metadata, device, reader, framework)

    # 暴露 reader 引用，供 ParallelThreeFSLoader 预开文件使用
    construct_copier.reader = reader  # type: ignore[attr-defined]

    return construct_copier
