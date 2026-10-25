#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <dlfcn.h>
#include <filesystem>
#include <string>

// Resolves `rel_path` relative to the directory containing the shared library
// that holds `func_ptr`. Returns the absolute path if it exists on disk,
// otherwise returns `fallback`.
inline std::string resolve_path_near_lib(
    void *func_ptr, const std::filesystem::path &rel_path,
    const std::string &fallback
) {
    Dl_info info;
    if (dladdr(func_ptr, &info) && info.dli_fname) {
        auto candidate =
            std::filesystem::path(info.dli_fname).parent_path() / rel_path;
        if (std::filesystem::exists(candidate)) return candidate.string();
    }
    return fallback;
}

#endif
