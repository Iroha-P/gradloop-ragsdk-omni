from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class DirectoryIdentity:
    device: int
    file_id: int


class SecureWriteError(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def _validate_leaf_name(name: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise SecureWriteError("invalid_output_name")


def _preflight_leaf(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError:
        raise SecureWriteError("unsafe_existing_output") from None

    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or attributes & _REPARSE_POINT_FLAG
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise SecureWriteError("unsafe_existing_output")


def _temporary_name() -> str:
    return f".private-safe-{secrets.token_hex(16)}.tmp"


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _FILE_READ_ATTRIBUTES = 0x0080
    _DELETE = 0x00010000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_DISPOSITION_INFO_CLASS = 4
    _FILE_RENAME_INFORMATION_CLASS = 10
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", ctypes.c_ubyte),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        ]

    class _FileDispositionInformation(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("status_or_pointer", wintypes.LPVOID),
            ("information", ctypes.c_size_t),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CREATE_FILE = _KERNEL32.CreateFileW
    _CREATE_FILE.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CREATE_FILE.restype = wintypes.HANDLE
    _GET_FILE_INFORMATION = _KERNEL32.GetFileInformationByHandle
    _GET_FILE_INFORMATION.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _GET_FILE_INFORMATION.restype = wintypes.BOOL
    _WRITE_FILE = _KERNEL32.WriteFile
    _WRITE_FILE.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _WRITE_FILE.restype = wintypes.BOOL
    _FLUSH_FILE_BUFFERS = _KERNEL32.FlushFileBuffers
    _FLUSH_FILE_BUFFERS.argtypes = [wintypes.HANDLE]
    _FLUSH_FILE_BUFFERS.restype = wintypes.BOOL
    _SET_FILE_INFORMATION = _KERNEL32.SetFileInformationByHandle
    _SET_FILE_INFORMATION.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _SET_FILE_INFORMATION.restype = wintypes.BOOL
    _NTDLL = ctypes.WinDLL("ntdll")
    _NT_SET_INFORMATION_FILE = _NTDLL.NtSetInformationFile
    _NT_SET_INFORMATION_FILE.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    _NT_SET_INFORMATION_FILE.restype = wintypes.LONG
    _CLOSE_HANDLE = _KERNEL32.CloseHandle
    _CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _CLOSE_HANDLE.restype = wintypes.BOOL

    def _open_windows_directory(path: Path):
        handle = _CREATE_FILE(
            str(path),
            _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise SecureWriteError("write_directory_unavailable")
        return handle

    def _windows_information(handle) -> _ByHandleFileInformation:
        information = _ByHandleFileInformation()
        if not _GET_FILE_INFORMATION(handle, ctypes.byref(information)):
            raise SecureWriteError("write_directory_unavailable")
        return information

    def _windows_identity(handle) -> DirectoryIdentity:
        information = _windows_information(handle)
        return DirectoryIdentity(
            device=information.volume_serial_number,
            file_id=(information.file_index_high << 32) | information.file_index_low,
        )

    @contextmanager
    def _locked_windows_directory(path: Path) -> Iterator[tuple[object, DirectoryIdentity]]:
        handle = _open_windows_directory(path)
        try:
            information = _windows_information(handle)
            if (
                not information.file_attributes & _FILE_ATTRIBUTE_DIRECTORY
                or information.file_attributes & _REPARSE_POINT_FLAG
            ):
                raise SecureWriteError("write_directory_unavailable")
            yield handle, _windows_identity(handle)
        finally:
            _CLOSE_HANDLE(handle)

    def _revalidate_windows_directory(path: Path, identity: DirectoryIdentity) -> None:
        handle = _open_windows_directory(path)
        try:
            information = _windows_information(handle)
            if (
                not information.file_attributes & _FILE_ATTRIBUTE_DIRECTORY
                or information.file_attributes & _REPARSE_POINT_FLAG
                or _windows_identity(handle) != identity
            ):
                raise SecureWriteError("write_directory_changed")
        finally:
            _CLOSE_HANDLE(handle)

    def _create_new_windows_file(path: Path):
        handle = _CREATE_FILE(
            str(path),
            _GENERIC_WRITE | _DELETE,
            _FILE_SHARE_READ,
            None,
            _CREATE_NEW,
            _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise SecureWriteError("secure_write_failed")
        return handle

    def _write_windows_payload(handle, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 1024 * 1024]
            buffer = ctypes.create_string_buffer(chunk)
            written = wintypes.DWORD()
            if not _WRITE_FILE(
                handle,
                buffer,
                len(chunk),
                ctypes.byref(written),
                None,
            ):
                raise SecureWriteError("secure_write_failed")
            if written.value != len(chunk):
                raise SecureWriteError("secure_write_failed")
            offset += written.value
        if not _FLUSH_FILE_BUFFERS(handle):
            raise SecureWriteError("secure_write_failed")

    def _replace_windows_file(handle, directory_handle, name: str) -> None:
        encoded_name = name.encode("utf-16-le")
        file_name_offset = _FileRenameInformation.file_name.offset
        buffer_size = ctypes.sizeof(_FileRenameInformation) + len(encoded_name)
        buffer = ctypes.create_string_buffer(buffer_size)
        information = ctypes.cast(buffer, ctypes.POINTER(_FileRenameInformation)).contents
        information.replace_if_exists = 1
        information.root_directory = directory_handle
        information.file_name_length = len(encoded_name)
        ctypes.memmove(ctypes.addressof(buffer) + file_name_offset, encoded_name, len(encoded_name))
        io_status = _IoStatusBlock()
        status = _NT_SET_INFORMATION_FILE(
            handle,
            ctypes.byref(io_status),
            buffer,
            buffer_size,
            _FILE_RENAME_INFORMATION_CLASS,
        )
        if status != 0:
            raise SecureWriteError("secure_write_failed")

    def _discard_windows_file(handle) -> None:
        disposition = _FileDispositionInformation(delete_file=1)
        if not _SET_FILE_INFORMATION(
            handle,
            _FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise SecureWriteError("secure_cleanup_failed")

    def _atomic_write_windows(
        directory: Path,
        payloads: Mapping[str, bytes],
        expected_identity: DirectoryIdentity,
    ) -> None:
        temporary_handles: dict[str, object] = {}
        published_names: set[str] = set()
        with _locked_windows_directory(directory) as (directory_handle, identity):
            if identity != expected_identity:
                raise SecureWriteError("write_directory_changed")
            _revalidate_windows_directory(directory, identity)
            for name in payloads:
                _preflight_leaf(directory / name)
            try:
                for name in payloads:
                    temporary = directory / _temporary_name()
                    temporary_handles[name] = _create_new_windows_file(temporary)
                _revalidate_windows_directory(directory, identity)
                for name, payload in payloads.items():
                    _write_windows_payload(temporary_handles[name], payload)
                _revalidate_windows_directory(directory, identity)
                for name in payloads:
                    _preflight_leaf(directory / name)
                for name in payloads:
                    _replace_windows_file(
                        temporary_handles[name],
                        directory_handle,
                        name,
                    )
                    published_names.add(name)
                _revalidate_windows_directory(directory, identity)
            except SecureWriteError:
                raise
            except OSError:
                raise SecureWriteError("secure_write_failed") from None
            finally:
                cleanup_failed = False
                for name, handle in temporary_handles.items():
                    if name not in published_names:
                        try:
                            _discard_windows_file(handle)
                        except SecureWriteError:
                            cleanup_failed = True
                    try:
                        if not _CLOSE_HANDLE(handle):
                            cleanup_failed = True
                    except Exception:
                        cleanup_failed = True
                temporary_handles.clear()
                if cleanup_failed:
                    raise SecureWriteError("secure_cleanup_failed")


def _preflight_posix_leaf(directory_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise SecureWriteError("unsafe_existing_output") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SecureWriteError("unsafe_existing_output")


def _posix_directory_identity(directory_fd: int) -> DirectoryIdentity:
    try:
        metadata = os.fstat(directory_fd)
    except OSError:
        raise SecureWriteError("write_directory_unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise SecureWriteError("write_directory_unavailable")
    return DirectoryIdentity(device=metadata.st_dev, file_id=metadata.st_ino)


def _atomic_write_posix(
    directory: Path,
    payloads: Mapping[str, bytes],
    expected_identity: DirectoryIdentity,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        raise SecureWriteError("write_directory_unavailable") from None

    temporary_names: dict[str, str] = {}
    try:
        if _posix_directory_identity(directory_fd) != expected_identity:
            raise SecureWriteError("write_directory_changed")
        for name in payloads:
            _preflight_posix_leaf(directory_fd, name)
        for name, payload in payloads.items():
            temporary_name = _temporary_name()
            temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_descriptor = os.open(
                    temporary_name,
                    temporary_flags,
                    0o600,
                    dir_fd=directory_fd,
                )
            except OSError:
                raise SecureWriteError("secure_write_failed") from None
            temporary_names[name] = temporary_name
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(file_descriptor, view)
                    if written <= 0:
                        raise SecureWriteError("secure_write_failed")
                    view = view[written:]
                os.fsync(file_descriptor)
            finally:
                try:
                    os.close(file_descriptor)
                except OSError:
                    raise SecureWriteError("secure_cleanup_failed") from None
        for name in payloads:
            _preflight_posix_leaf(directory_fd, name)
        for name, temporary_name in temporary_names.items():
            os.replace(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        os.fsync(directory_fd)
    except SecureWriteError:
        raise
    except OSError:
        raise SecureWriteError("secure_write_failed") from None
    finally:
        cleanup_failed = False
        for temporary_name in temporary_names.values():
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_failed = True
        try:
            os.close(directory_fd)
        except OSError:
            cleanup_failed = True
        if cleanup_failed:
            raise SecureWriteError("secure_cleanup_failed")


def directory_identity(directory: Path) -> DirectoryIdentity:
    if os.name == "nt":
        with _locked_windows_directory(directory) as (_, identity):
            return identity

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        raise SecureWriteError("write_directory_unavailable") from None
    try:
        return _posix_directory_identity(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_batch(
    directory: Path,
    payloads: Mapping[str, bytes],
    *,
    expected_identity: DirectoryIdentity,
) -> None:
    if not payloads:
        return
    for name in payloads:
        _validate_leaf_name(name)
    if os.name == "nt":
        _atomic_write_windows(directory, payloads, expected_identity)
    else:
        _atomic_write_posix(directory, payloads, expected_identity)
