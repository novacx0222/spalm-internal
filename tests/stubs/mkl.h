#pragma once

// Test double for portable builds: no MKL is linked or executed.
inline void mkl_free_buffers() {}
