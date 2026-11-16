"""scripts/openblas_thp_patch.py on a fixture shaped like OpenBLAS 0.3.30's
memory.c: a TLS allocator section, then the legacy one with the NO_WARMUP
alloc_mmap. Run from the repository root."""
import importlib.util
import re
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("thp", Path("scripts/openblas_thp_patch.py"))
thp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(thp)

FIXTURE = """#include "common.h"
#if defined(COMPILE_TLS)
static void *alloc_mmap(void *address){
  map_address = mmap(address,
                     allocation_block_size,
                     MMAP_ACCESS, MMAP_POLICY, -1, 0);
}
#else
/* USE_TLS / COMPILE_TLS not set */
#ifdef NO_WARMUP
static void *alloc_mmap(void *address){
  void *map_address;
  if (address){
    map_address = mmap(address,
                       BUFFER_SIZE,
                       MMAP_ACCESS, MMAP_POLICY | MAP_FIXED, -1, 0);
  } else {
    map_address = mmap(address,
                       BUFFER_SIZE,
                       MMAP_ACCESS, MMAP_POLICY, -1, 0);
  }
  return map_address;
}
#else
static void *alloc_mmap(void *address){
      map_address = mmap(NULL, BUFFER_SIZE, MMAP_ACCESS, MMAP_POLICY, -1, 0);
}
#endif
#endif
"""


class PatchTest(unittest.TestCase):
    def test_patches_only_the_legacy_no_warmup_call(self):
        out = thp.patch(FIXTURE)
        self.assertEqual(out.count("spalm_thp_mmap(BUFFER_SIZE)"), 1)
        legacy = out.index("/* USE_TLS / COMPILE_TLS not set */")
        self.assertGreater(out.index("static void *spalm_thp_mmap"), legacy)
        self.assertLess(out.index("static void *spalm_thp_mmap"), out.index("#ifdef NO_WARMUP"))
        # Untouched: the TLS call, the MAP_FIXED call and the warm-up variant.
        self.assertIn("allocation_block_size,\n                     MMAP_ACCESS, MMAP_POLICY, -1, 0);", out)
        self.assertIn("MMAP_POLICY | MAP_FIXED, -1, 0);", out)
        self.assertIn("mmap(NULL, BUFFER_SIZE, MMAP_ACCESS, MMAP_POLICY, -1, 0);", out)
        self.assertIn("openblas_spalm_thp_buffers", out)

    def test_refuses_twice_and_unknown_sources(self):
        with self.assertRaises(SystemExit):
            thp.patch(thp.patch(FIXTURE))
        with self.assertRaises(SystemExit):
            thp.patch(FIXTURE.replace("/* USE_TLS / COMPILE_TLS not set */", ""))
        with self.assertRaises(SystemExit):
            thp.patch(re.sub(r"mmap\(address,\s*BUFFER_SIZE,\s*MMAP_ACCESS, MMAP_POLICY, -1, 0\)",
                             "mmap(address, size)", FIXTURE))


if __name__ == "__main__":
    unittest.main()
