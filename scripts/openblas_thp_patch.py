#!/usr/bin/env python3
"""Patch OpenBLAS 0.3.30's driver/others/memory.c: optional 2 MiB transparent
huge pages for its GEMM buffers (spalm.conf openblas_huge_pages).

OpenBLAS maps each thread's buffer (BUFFER_SIZE, 128 MiB on x86_64) with mmap
on 4 KiB pages, while MKL's buffers go through i_malloc to SPALM's huge-page
allocator (huge_pages.h). With SPALM_OPENBLAS_THP=1 in the environment when
the library is loaded, the patched allocator maps BUFFER_SIZE + 2 MiB, keeps
the 2 MiB aligned BUFFER_SIZE inside (unmapping the rest, so OpenBLAS's own
munmap(address, BUFFER_SIZE) still frees all of it) and marks it
MADV_HUGEPAGE. Otherwise it maps exactly as before. run_baseline.so sets the
variable from spalm.conf before dlopen.

Only the default build is patched: the legacy allocator (USE_TLS unset) with
NO_WARMUP (Makefile.rule's default). Anything else fails here.

    python3 openblas_thp_patch.py <OpenBLAS source dir>
"""
import re
import sys
from pathlib import Path

MARKER = "SPALM_OPENBLAS_THP"
LEGACY = "/* USE_TLS / COMPILE_TLS not set */"
# The legacy NO_WARMUP alloc_mmap, non-fixed branch (address is NULL there).
# The TLS allocator maps allocation_block_size and the warm-up variant
# mmap(NULL, ...), so neither matches.
CALL = re.compile(r"map_address\s*=\s*mmap\(\s*address\s*,\s*BUFFER_SIZE\s*,\s*"
                  r"MMAP_ACCESS\s*,\s*MMAP_POLICY\s*,\s*-1\s*,\s*0\s*\)\s*;")

HELPER = r"""
/* SPALM (inner-v39): SPALM_OPENBLAS_THP=1 at load puts the GEMM buffers on
   2 MiB transparent huge pages (scripts/openblas_thp_patch.py). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static int spalm_thp_state = -1;
static long spalm_thp_count = 0;
__attribute__((visibility("default"))) long openblas_spalm_thp_buffers(void) {
  return spalm_thp_state == 1 ? __atomic_load_n(&spalm_thp_count, __ATOMIC_RELAXED) : -1;
}
static void *spalm_thp_mmap(size_t size) {
  if (spalm_thp_state < 0) {
    const char *v = getenv("SPALM_OPENBLAS_THP");
    spalm_thp_state = (v && strcmp(v, "1") == 0) ? 1 : 0;
  }
  if (spalm_thp_state != 1) return mmap(NULL, size, MMAP_ACCESS, MMAP_POLICY, -1, 0);
  const size_t huge = (size_t)2 << 20;
  char *raw = (char *)mmap(NULL, size + huge, MMAP_ACCESS, MMAP_POLICY, -1, 0);
  if (raw == (char *)MAP_FAILED) return MAP_FAILED;
  char *aligned = (char *)(((uintptr_t)raw + huge - 1) & ~(uintptr_t)(huge - 1));
  if (aligned > raw) munmap(raw, (size_t)(aligned - raw));
  const size_t tail = (size_t)((raw + size + huge) - (aligned + size));
  if (tail) munmap(aligned + size, tail);
  if (madvise(aligned, size, MADV_HUGEPAGE) == 0)
    __atomic_add_fetch(&spalm_thp_count, 1, __ATOMIC_RELAXED);
  return aligned;
}
"""


def patch(text):
    if MARKER in text:
        raise SystemExit("memory.c is already patched")
    if text.count(LEGACY) != 1:
        raise SystemExit(f"expected one {LEGACY!r} line (legacy allocator section)")
    calls = CALL.findall(text)
    if len(calls) != 1:
        raise SystemExit(f"expected one legacy NO_WARMUP mmap(address, BUFFER_SIZE, ...) call, found {len(calls)}")
    split = text.index(LEGACY) + len(LEGACY)
    if CALL.search(text, split) is None:
        raise SystemExit("the mmap call is not in the legacy allocator section")
    head, tail = text[:split], text[split:]
    tail = CALL.sub("map_address = spalm_thp_mmap(BUFFER_SIZE);", tail, count=1)
    return head + "\n" + HELPER + tail


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    path = Path(argv[1]) / "driver/others/memory.c"
    path.write_text(patch(path.read_text()))
    print(f"patched {path}")


if __name__ == "__main__":
    main(sys.argv)
