// openblas_baseline.h against fake libraries (tests/fake_openblas.cpp):
// which builds are accepted, SPALM_OPENBLAS_THP set before dlopen, the
// setting fixed once loaded, the GEMM call and the log lines. Each case runs
// in its own process, since a process loads OpenBLAS once.
#include "openblas_baseline.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

static std::string dir;

static void in_child(const char *name, const std::function<void()> &body) {
    std::cout.flush();
    const pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        body();
        std::cout.flush();
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::cerr << "FAIL: " << name << std::endl;
        std::exit(1);
    }
    std::cout << "ok: " << name << std::endl;
}

static std::string lib(const char *which) {
    return dir + "/fake_openblas_" + which + ".so";
}

// C = R * S^T for R 2 x 3, S 2 x 3 through run(), with the log captured.
template <typename T>
static std::string run_case(const std::string &path, bool huge, int &status) {
    T R[] = {1, 2, 3, 4, 5, 6}, S[] = {1, 0, -1, 2, 1, 0};
    T C[4] = {-9, -9, -9, -9};
    std::ostringstream log;
    auto *old = std::cout.rdbuf(log.rdbuf());
    std::vector<std::string> stages;
    status = spalm_openblas::run(
        path, sizeof(T) == sizeof(float) ? 0 : 1, R, S, C, 2, 3, 2, 30, huge,
        [&](const char *label, auto) { stages.push_back(label); }
    );
    std::cout.rdbuf(old);
    if (status == 0) {
        const T expected[] = {-2, 4, -2, 13};
        for (int i = 0; i < 4; ++i) assert(C[i] == expected[i]);
        assert(stages.back() == "[openblas-baseline-total]");
        assert(stages[stages.size() - 2] == "[openblas-dense-mm-time]");
    }
    std::string text = log.str();
    for (auto &s : stages) text += s + "\n";
    return text;
}

static bool has(const std::string &text, const std::string &part) {
    return text.find(part) != std::string::npos;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    dir = argv[1];
    in_child("OpenMP build, 4 KiB buffers, double and float", [] {
        setenv(spalm_openblas::kThpEnv, "1", 1); // spalm.conf decides
        int status = 1;
        auto log = run_case<double>(lib("omp"), false, status);
        assert(status == 0);
        assert(has(log, "[openblas-load]"));
        assert(has(log, "[openblas-config] parallel=2 corename=fakecore threads=30 omp_max_threads=30 huge_pages=0 thp_buffers=-1"));
        assert(has(log, "config=fake OpenBLAS"));
        log = run_case<float>(lib("omp"), false, status);
        assert(status == 0 && !has(log, "[openblas-load]"));
        // The setting cannot change once loaded.
        log = run_case<double>(lib("omp"), true, status);
        assert(status == 1);
    });
    in_child("OpenMP build, huge pages seen by the constructor", [] {
        unsetenv(spalm_openblas::kThpEnv);
        int status = 1;
        auto log = run_case<double>(lib("omp"), true, status);
        assert(status == 0);
        assert(has(log, "huge_pages=1 thp_buffers=7"));
    });
    in_child("unpatched build: 4 KiB only", [] {
        int status = 0;
        run_case<double>(lib("plain"), true, status);
        assert(status == 1);
        auto log = run_case<double>(lib("plain"), false, status);
        assert(status == 0 && has(log, "thp_buffers=unpatched"));
    });
    in_child("pthreads build rejected", [] {
        std::string error;
        double ms = 0;
        assert(!spalm_openblas::load(lib("pthreads"), false, error, ms));
        assert(has(error, "not an OpenMP build (openblas_get_parallel() = 1)"));
    });
    in_child("missing library", [] {
        int status = 0;
        run_case<double>(dir + "/no-such-libopenblas.so", false, status);
        assert(status == 1);
    });
    in_child("bad type", [] {
        double x = 0;
        assert(spalm_openblas::run(lib("omp"), 2, &x, &x, &x, 1, 1, 1, 1, false,
                                   [](const char *, auto) {}) == 1);
    });
    std::cout << "PASS: OpenBLAS baseline loader\n";
}
