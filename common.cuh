// common.cuh - shared operator, CLI, input and output code for every scan.
//
//   #include "../common.cuh"
//   Args a = parse_args(argc, argv, 1 << 24);   // default N when none given
//   fill_input(src, a.n, a.input);              // repeat the --input pattern
//   ...  scan<<<...>>>(dst_d, src_d, a.n, a.op);
//   write_output(dst, a.n);                     // one line of values on stdout
//
// Command line (parsed with getopt_long, so N and the flags can come in any
// order):
//   ./scan.out [N] [-o|--op add|mul|max|min] [-i|--input v1,v2,...]
//   N is a plain count (16777216) or a power of two (2^24).
//   --op defaults to add, --input defaults to 1 (so add gives 1,2,3,...).
//
// Output (all on stdout, in this order):
//   [bench] GPU specs line
//   [bench] timing line
//   one line with all N scan results, comma separated, "%.9g" (enough digits
//   to round-trip a float). It is always the last line.

#pragma once

#include "bench.cuh"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <vector>

// The binary operator of the scan. All four are associative and commutative,
// so any combining order gives the same result (up to float rounding for add
// and mul).
enum BINOP { ADD, MUL, MAX, MIN };

__host__ __device__ __forceinline__ float binop(float a, float b,
                                                enum BINOP op) {
  switch (op) {
  case ADD:
    return a + b;
  case MUL:
    return a * b;
  case MAX:
    return fmaxf(a, b);
  case MIN:
    return fminf(a, b);
  }
  return a + b;
}

// The neutral element: binop(x, identity(op), op) == x. Used to pad slots past
// N and to seed exclusive values, so padding never changes a result.
__host__ __device__ __forceinline__ float identity(enum BINOP op) {
  switch (op) {
  case ADD:
    return 0.0f;
  case MUL:
    return 1.0f;
  case MAX:
    return -INFINITY;
  case MIN:
    return INFINITY;
  }
  return 0.0f;
}

inline const char *op_name(enum BINOP op) {
  switch (op) {
  case ADD:
    return "add";
  case MUL:
    return "mul";
  case MAX:
    return "max";
  case MIN:
    return "min";
  }
  return "?";
}

struct Args {
  size_t n = 0;
  enum BINOP op = ADD;
  std::vector<float> input{1.0f}; // pattern repeated to fill n elements
};

// Prints the usage to stderr and exits; never returns to the caller.
inline void usage_exit(const char *prog, const char *msg, const char *arg,
                       int status = EXIT_FAILURE) {
  if (msg)
    std::fprintf(stderr, "%s: %s '%s'\n", prog, msg, arg ? arg : "");
  std::fprintf(
      stderr,
      "usage: %s [N] [-o|--op add|mul|max|min] [-i|--input v1,v2,...]\n"
      "  N           element count, e.g. 16777216 or 2^24\n"
      "  -o, --op    scan operator (default add)\n"
      "  -i, --input values repeated to fill N elements (default 1)\n"
      "  -h, --help  show this help\n",
      prog);
  std::exit(status);
}

inline enum BINOP parse_op(const char *prog, const char *s) {
  if (!std::strcmp(s, "add"))
    return ADD;
  if (!std::strcmp(s, "mul"))
    return MUL;
  if (!std::strcmp(s, "max"))
    return MAX;
  if (!std::strcmp(s, "min"))
    return MIN;
  usage_exit(prog, "bad --op", s);
  return ADD; // not reached: usage_exit exits
}

inline std::vector<float> parse_input(const char *prog, const char *s) {
  std::vector<float> vals;
  const char *p = s;
  while (true) {
    char *end = nullptr;
    float v = std::strtof(p, &end);
    if (end == p || (*end != ',' && *end != '\0'))
      usage_exit(prog, "bad --input", s);
    vals.push_back(v);
    if (*end == '\0')
      break;
    p = end + 1;
  }
  return vals;
}

// getopt_long: --op/-o and --input/-i take a value ("--op mul", "--op=mul",
// "-o mul", "-omul"). GNU getopt moves non-option arguments to the end, so N
// can come before or after the flags. getopt prints its own message for an
// unknown option or a missing value; we then print the usage and exit.
inline Args parse_args(int argc, char **argv, size_t default_n) {
  const char *prog = argc > 0 ? argv[0] : "scan.out";
  static const struct option long_opts[] = {
      {"op", required_argument, nullptr, 'o'},
      {"input", required_argument, nullptr, 'i'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };
  Args a;
  a.n = default_n;
  optind = 1;
  int c;
  while ((c = getopt_long(argc, argv, "o:i:h", long_opts, nullptr)) != -1) {
    switch (c) {
    case 'o':
      a.op = parse_op(prog, optarg);
      break;
    case 'i':
      a.input = parse_input(prog, optarg);
      break;
    case 'h':
      usage_exit(prog, nullptr, nullptr, EXIT_SUCCESS);
      break;
    default: // '?': getopt already said what was wrong
      usage_exit(prog, nullptr, nullptr);
      break;
    }
  }
  if (optind < argc) {
    a.n = bench::arg_n(argc, argv, default_n, optind); // exits on a bad count
    if (optind + 1 < argc)
      usage_exit(prog, "unexpected argument", argv[optind + 1]);
  }
  // Every kernel takes N as unsigned int.
  if (a.n > UINT_MAX) {
    std::fprintf(stderr, "%s: N=%zu is too large (max %u)\n", prog, a.n,
                 UINT_MAX);
    std::exit(EXIT_FAILURE);
  }
  return a;
}

// dst[i] = pattern[i % pattern.size()]
inline void fill_input(float *dst, size_t n,
                       const std::vector<float> &pattern) {
  size_t k = pattern.size();
  for (size_t i = 0; i < n; ++i)
    dst[i] = pattern[i % k];
}

// All n values on one stdout line. Values are formatted into a 16 MB buffer
// that is written with one fwrite each time it fills, so 2^28 values don't
// turn into millions of small writes.
inline void write_output(const float *dst, size_t n) {
  static char buf[1 << 24];
  const size_t room = 32; // longest "%.9g" value plus a comma, rounded up
  size_t len = 0;
  std::fflush(stdout); // keep the [bench] lines ahead of the values
  for (size_t i = 0; i < n; ++i) {
    if (len + room > sizeof(buf)) {
      std::fwrite(buf, 1, len, stdout);
      len = 0;
    }
    len += (size_t)std::snprintf(buf + len, room, i ? ",%.9g" : "%.9g",
                                 (double)dst[i]);
  }
  buf[len++] = '\n';
  std::fwrite(buf, 1, len, stdout);
  std::fflush(stdout);
}
