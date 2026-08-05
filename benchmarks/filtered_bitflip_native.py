"""Optional ctypes bridge to the vendored libsais research backend."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "benchmarks" / "third_party" / "libsais"
DEFAULT_LIBRARY = ROOT / "build" / "filtered_bitflip" / "librosa_libsais.so"


def build_libsais(
    output: Path = DEFAULT_LIBRARY,
    *,
    compiler: str = "cc",
) -> Path:
    """Compile the pinned libsais source into one private shared library."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c99",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        "-I",
        str(VENDOR_ROOT / "include"),
        str(VENDOR_ROOT / "src" / "libsais.c"),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


class LibsaisBackend:
    """Exact SA/LCP builder using the libsais C99 API."""

    name = "libsais"

    def __init__(self, library_path: Path | str | None = None) -> None:
        if library_path is None:
            configured = os.environ.get("ROSA_LIBSAIS_PATH")
            library_path = Path(configured) if configured else DEFAULT_LIBRARY
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                f"libsais backend not built: {self.library_path}; run "
                "python benchmarks/filtered_bitflip_native.py --build"
            )
        library = ctypes.CDLL(str(self.library_path))
        int_pointer = ctypes.POINTER(ctypes.c_int32)
        library.libsais_int.argtypes = [
            int_pointer,
            int_pointer,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        library.libsais_int.restype = ctypes.c_int32
        library.libsais_plcp_int.argtypes = [
            int_pointer,
            int_pointer,
            int_pointer,
            ctypes.c_int32,
        ]
        library.libsais_plcp_int.restype = ctypes.c_int32
        library.libsais_lcp.argtypes = [
            int_pointer,
            int_pointer,
            int_pointer,
            ctypes.c_int32,
        ]
        library.libsais_lcp.restype = ctypes.c_int32
        self.library = library

    @classmethod
    def available(cls, library_path: Path | str | None = None) -> bool:
        try:
            cls(library_path)
        except (FileNotFoundError, OSError):
            return False
        return True

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"libsais {operation} failed with code {result}")

    def suffix_array_lcp(
        self,
        text: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        size = len(text)
        if size > (1 << 31) - 1:
            raise ValueError("32-bit libsais backend cannot index this text")
        if any(not 0 <= int(value) < (1 << 31) - 1 for value in text):
            raise ValueError(
                "libsais integer symbols must leave room for an int32 alphabet size"
            )
        if size == 0:
            return (), ()
        text_buffer = (ctypes.c_int32 * size)(*(int(value) for value in text))
        suffix_array = (ctypes.c_int32 * size)()
        plcp = (ctypes.c_int32 * size)()
        lcp = (ctypes.c_int32 * size)()
        alphabet_size = max(text) + 1
        self._check(
            self.library.libsais_int(
                text_buffer,
                suffix_array,
                size,
                alphabet_size,
                0,
            ),
            "suffix-array construction",
        )
        self._check(
            self.library.libsais_plcp_int(
                text_buffer,
                suffix_array,
                plcp,
                size,
            ),
            "PLCP construction",
        )
        self._check(
            self.library.libsais_lcp(plcp, suffix_array, lcp, size),
            "LCP construction",
        )
        return tuple(suffix_array), tuple(lcp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--compiler", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build")
    print(build_libsais(args.output, compiler=args.compiler))


if __name__ == "__main__":
    main()
