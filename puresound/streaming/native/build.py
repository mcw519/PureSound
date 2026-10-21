"""Build the optional ORT CPU library locally, without downloads or ORT linking.

python -m puresound.streaming.native.build --output /path/libpuresound_ssm.so
On x86-64 Linux the library uses the baseline ISA and selects its AVX2/AVX-512
kernel when it is loaded, so one build runs on any x86-64 Linux host with the
same C library. On arm64 (Linux, or macOS with Clang) NEON is the baseline.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def build_library(output: str | Path, compiler: str | None = None) -> Path:
    macos = sys.platform == "darwin"
    if not (sys.platform.startswith("linux") or macos):
        raise RuntimeError("native SSM build supports Linux and macOS; use the portable ONNX elsewhere")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("fused_ssm.cc")
    cxx = shlex.split(compiler or os.environ.get("CXX", "clang++" if macos else "g++"))
    # Clang reports -ffp-contract=off overriding -ffast-math; that is the intent.
    quiet = ["-Wno-overriding-option"] if macos else []
    # A running ORT session maps the library: rewriting that file in place can
    # crash it. Link beside the target and rename, so the path gets a new inode.
    staged = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="puresound-ssm-") as temp:
            obj = Path(temp) / "fused_ssm.o"
            subprocess.run([
                *cxx, "-std=c++17", "-O3", "-fPIC",
                "-fno-math-errno", "-fno-trapping-math", "-ffast-math",
                "-ffp-contract=off", "-Wno-psabi", *quiet, "-c", str(source), "-o", str(obj),
            ], check=True)
            # Do NOT link with -ffast-math: crtfastmath would change denormal
            # handling for the entire Python process, including other ORT sessions.
            subprocess.run([*cxx, "-shared", str(obj), "-o", str(staged), "-lm"], check=True)
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default=None,
                        help="defaults to CXX, else clang++ on macOS and g++ on Linux")
    args = parser.parse_args()
    print(build_library(args.output, args.compiler))


if __name__ == "__main__":
    main()
