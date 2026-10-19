from __future__ import annotations

import hashlib
import os
from pathlib import Path


_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_tree(root: Path | str) -> dict[str, str]:
    root_path = Path(root)
    checksums: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root_path, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = [name for name in directory_names if not (directory_path / name).is_symlink()]
        for name in file_names:
            file_path = directory_path / name
            if file_path.is_symlink() or not file_path.is_file():
                continue
            relative_path = file_path.relative_to(root_path).as_posix()
            checksums[relative_path] = sha256_file(file_path)
    return dict(sorted(checksums.items()))
