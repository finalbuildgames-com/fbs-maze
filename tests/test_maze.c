/*
 * tests/test_maze.c — witnesses for include/fbs/maze.h.
 *
 * Self-contained: no test framework. Exit code = number of failures (clamped
 * to 100 so it survives the 8-bit exit status; the true count is printed).
 *
 * Covers the module's test plan, MT-1 .. MT-14, with the witnesses described
 * at each test below, plus allocator failure, every error status, every
 * E_TRUNCATED path, NULL/bad-enum validation, 0xA5 output-untouched checks on
 * every entry point, the status-name/version/algorithm-name functions, and
 * golden fixtures under tests/fixtures/maze/.
 *
 * Independence. Where the plan asks for a second opinion, this file computes
 * it with its own code rather than by calling the module back: it has its own
 * BFS, its own flood fill, its own cycle finder, its own render-grid analysis,
 * and — for MT-9 — its own transcription of Kruskal's draw sequence built on
 * the public PRNG, which pins the module's internal Fisher-Yates.
 *
 * MT-4's WASM half ("assert the same fixtures from the WASM build") is out of
 * scope for this C binary and is explicitly not part of this milestone; the
 * golden files below are the artifact a WASM build would be compared against,
 * byte for byte.
 *
 * MT-13's two-thread half is not run here: the test binary links no thread
 * library and adding one would change the build for every module. The module
 * has no globals and no static mutable state at all (checked with `nm` over
 * libfbs_maze.a: zero symbols in .data or .bss), and the interleaved-objects
 * witness below is the runnable half.
 *
 *   ./fbs_test_maze                       compare against tests/fixtures/maze/
 *   ./fbs_test_maze --write-fixtures      rewrite those files
 *   ./fbs_test_maze --fixture-dir DIR     look for fixtures under DIR
 *
 * MZ-1 (module review): test_mt9_rejection_branch
 * witnesses the rejection branch of fbs_maze_rng_below with bounds above 2^31,
 * where the threshold is half the range instead of a rounding error. Verified
 * to kill both plausible mutants: `next() % bound` with no rejection at all
 * (which leaves all eight golden mazes byte-identical, exactly as the review
 * says) and the other rejection formulation that discards the high residue.
 *
 * FBS_MAZE_TEST_FAST=1 in the environment scales the four statistical loops
 * down (MT-9's shuffle census 2,000,000 -> 200,000 draws, its rng_below census
 * 2^22 -> 2^18 draws per bound, MZ-1's draw-count census 20,000 -> 5,000 calls
 * per bound, MT-10's Eller census 2,000 -> 250 seeds) and shrinks MT-13's
 * maximum-size probe from 1024x1024 to 256x256. The chi-square bounds are
 * scale-invariant and stay in force; the +/-1% band MT-9 names is asserted in
 * full runs only, because at 200,000 samples it is tighter than one standard
 * deviation (see the note at test_mt9_shuffle_fairness: the chi-square is the
 * criterion, the band is a fixed-seed spot check at 2.95 sigma).
 */

#include "fbs/maze.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Harness                                                                   */
/* ------------------------------------------------------------------------- */

static int g_checks = 0;
static int g_fails = 0;
static int g_fast = 0;

static void check_impl(int cond, const char *expr, const char *file, int line) {
  ++g_checks;
  if (!cond) {
    ++g_fails;
    printf("FAIL %s:%d: %s\n", file, line, expr);
  }
}

#define CHECK(expr) check_impl((expr) ? 1 : 0, #expr, __FILE__, __LINE__)

static const char *g_fixture_dir = "fixtures/maze";
static int g_write_fixtures = 0;

static int untouched(const void *p, size_t n, unsigned char fill) {
  const unsigned char *b = (const unsigned char *)p;
  size_t i;
  for (i = 0; i < n; ++i)
    if (b[i] != fill) return 0;
  return 1;
}

/* Counting allocator (MT-12). */
typedef struct {
  int allocs;
  int frees;
  size_t bytes;
  int budget; /* < 0: unlimited, else the number of allocations still allowed */
} counting_alloc;

static void *ca_alloc(void *user, size_t bytes) {
  counting_alloc *c = (counting_alloc *)user;
  if (c->budget == 0) return NULL;
  if (c->budget > 0) --c->budget;
  ++c->allocs;
  c->bytes += bytes;
  return malloc(bytes);
}

static void ca_free(void *user, void *ptr) {
  counting_alloc *c = (counting_alloc *)user;
  if (!ptr) return;
  ++c->frees;
  free(ptr);
}

/* ------------------------------------------------------------------------- */
/* Independent maze analysis — none of this calls back into the module        */
/* ------------------------------------------------------------------------- */

#define ALGO_COUNT ((int)FBS_MAZE_ALGORITHM_COUNT)

static const uint8_t DIRS[4] = {FBS_MAZE_N, FBS_MAZE_E, FBS_MAZE_S, FBS_MAZE_W};

/* Neighbour of (x,y); returns 0 when it falls outside the grid. */
static int nb_of(uint32_t w, uint32_t h, uint32_t x, uint32_t y, uint8_t d, uint32_t *out) {
  switch (d) {
    case FBS_MAZE_N:
      if (y == 0u) return 0;
      *out = (y - 1u) * w + x;
      return 1;
    case FBS_MAZE_E:
      if (x + 1u >= w) return 0;
      *out = y * w + x + 1u;
      return 1;
    case FBS_MAZE_S:
      if (y + 1u >= h) return 0;
      *out = (y + 1u) * w + x;
      return 1;
    default:
      if (x == 0u) return 0;
      *out = y * w + x - 1u;
      return 1;
  }
}

typedef struct {
  uint8_t *seen;
  uint32_t *stack;
  uint32_t *dist;
  uint32_t cap;
} scratch;

static scratch *scratch_new(uint32_t cells) {
  scratch *s = (scratch *)malloc(sizeof *s);
  s->seen = (uint8_t *)malloc(cells);
  s->stack = (uint32_t *)malloc((size_t)cells * sizeof(uint32_t));
  s->dist = (uint32_t *)malloc((size_t)cells * sizeof(uint32_t));
  s->cap = cells;
  return s;
}

static void scratch_free(scratch *s) {
  free(s->seen);
  free(s->stack);
  free(s->dist);
  free(s);
}

/* Independent BFS: fills s->dist (UINT32_MAX unreachable), returns cells seen. */
static uint32_t ref_bfs(const uint8_t *cells, uint32_t w, uint32_t h, uint32_t start, scratch *s) {
  uint32_t n = w * h, head = 0u, tail = 0u, seen = 1u, i;
  for (i = 0u; i < n; ++i) s->dist[i] = UINT32_MAX;
  s->dist[start] = 0u;
  s->stack[tail++] = start;
  while (head < tail) {
    uint32_t c = s->stack[head++], x = c % w, y = c / w, d;
    for (d = 0u; d < 4u; ++d) {
      uint32_t nb;
      if (!(cells[c] & DIRS[d])) continue;
      if (!nb_of(w, h, x, y, DIRS[d], &nb)) continue;
      if (s->dist[nb] != UINT32_MAX) continue;
      s->dist[nb] = s->dist[c] + 1u;
      s->stack[tail++] = nb;
      ++seen;
    }
  }
  return seen;
}

/* Independent acyclicity: DFS that walks every open passage except the one it
 * arrived by; reaching an already-discovered cell is a cycle. */
static int ref_has_cycle(const uint8_t *cells, uint32_t w, uint32_t h, scratch *s) {
  uint32_t n = w * h, i, sp = 0u;
  uint8_t *from = (uint8_t *)malloc(n);
  int cycle = 0;
  memset(s->seen, 0, n);
  memset(from, 0, n);
  for (i = 0u; i < n && !cycle; ++i) {
    if (s->seen[i]) continue;
    s->seen[i] = 1u;
    s->stack[sp] = i;
    from[i] = 0u;
    ++sp;
    while (sp > 0u && !cycle) {
      uint32_t c, x, y, d;
      --sp;
      c = s->stack[sp];
      x = c % w;
      y = c / w;
      for (d = 0u; d < 4u; ++d) {
        uint32_t nb;
        if (!(cells[c] & DIRS[d])) continue;
        if (DIRS[d] == from[c]) continue;
        if (!nb_of(w, h, x, y, DIRS[d], &nb)) continue;
        if (s->seen[nb]) {
          cycle = 1;
          break;
        }
        s->seen[nb] = 1u;
        from[nb] = FBS_MAZE_OPPOSITE(DIRS[d]);
        s->stack[sp++] = nb;
      }
    }
  }
  free(from);
  return cycle;
}

static uint32_t ref_passages(const uint8_t *cells, uint32_t w, uint32_t h) {
  uint32_t x, y, c = 0u;
  for (y = 0u; y < h; ++y)
    for (x = 0u; x < w; ++x) {
      if (x + 1u < w && (cells[y * w + x] & FBS_MAZE_E)) ++c;
      if (y + 1u < h && (cells[y * w + x] & FBS_MAZE_S)) ++c;
    }
  return c;
}

static unsigned popcount4(uint8_t c) {
  return (unsigned)((c & 1u) + ((c >> 1) & 1u) + ((c >> 2) & 1u) + ((c >> 3) & 1u));
}

static uint32_t ref_dead_ends(const uint8_t *cells, uint32_t n) {
  uint32_t i, c = 0u;
  for (i = 0u; i < n; ++i)
    if (popcount4((uint8_t)(cells[i] & FBS_MAZE_DIRS)) == 1u) ++c;
  return c;
}

/* ------------------------------------------------------------------------- */
/* Fixtures                                                                  */
/* ------------------------------------------------------------------------- */

static unsigned char *read_file(const char *path, size_t *out_len) {
  FILE *fp = fopen(path, "rb");
  unsigned char *buf;
  long len;
  if (!fp) return NULL;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  len = ftell(fp);
  if (len < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);
  buf = (unsigned char *)malloc((size_t)len + 1u);
  if (!buf) {
    fclose(fp);
    return NULL;
  }
  *out_len = fread(buf, 1u, (size_t)len, fp);
  buf[*out_len] = 0;
  fclose(fp);
  return buf;
}

/* Writes the fixture under --write-fixtures, otherwise compares byte for byte. */
static void fixture(const char *name, const unsigned char *data, size_t len) {
  char path[512];
  sprintf(path, "%s/%s", g_fixture_dir, name);
  if (g_write_fixtures) {
    FILE *fp = fopen(path, "wb");
    ++g_checks;
    if (!fp) {
      ++g_fails;
      printf("FAIL cannot open %s for writing\n", path);
      return;
    }
    {
      size_t wrote = fwrite(data, 1u, len, fp);
      fclose(fp);
      CHECK(wrote == len);
      printf("wrote %s (%lu bytes)\n", path, (unsigned long)len);
    }
  } else {
    size_t got = 0u;
    unsigned char *golden = read_file(path, &got);
    ++g_checks;
    if (!golden) {
      ++g_fails;
      printf("FAIL cannot open fixture %s (run with --write-fixtures to create it)\n", path);
      return;
    }
    CHECK(got == len);
    if (got == len) CHECK(memcmp(golden, data, len) == 0);
    free(golden);
  }
}

/* ------------------------------------------------------------------------- */
/* Shared helpers                                                            */
/* ------------------------------------------------------------------------- */

static fbs_maze *make(uint32_t mw, uint32_t mh) {
  fbs_maze_config cfg;
  fbs_maze *m = NULL;
  cfg.max_width = mw;
  cfg.max_height = mh;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_OK);
  return m;
}

/* Field by field: fbs_maze_params has tail padding, so memcmp on it compares
 * indeterminate bytes. */
static int params_equal(const fbs_maze_params *a, const fbs_maze_params *b) {
  if (!a || !b) return 0;
  return a->width == b->width && a->height == b->height && a->seed == b->seed &&
         a->algorithm == b->algorithm && a->east_bias_percent == b->east_bias_percent &&
         a->newest_percent == b->newest_percent && a->braid_percent == b->braid_percent;
}

static fbs_maze_params params_of(uint32_t w, uint32_t h, uint64_t seed, int algorithm) {
  fbs_maze_params p = fbs_maze_params_default();
  p.width = w;
  p.height = h;
  p.seed = seed;
  p.algorithm = (uint8_t)algorithm;
  return p;
}

static const uint32_t SIZES[][2] = {{1u, 1u},  {1u, 7u},   {7u, 1u},  {2u, 2u},  {3u, 3u},
                                    {8u, 5u},  {5u, 8u},   {16u, 16u}, {31u, 17u}, {64u, 64u}};
#define SIZE_COUNT (sizeof SIZES / sizeof SIZES[0])
#define SWEEP_SEEDS 64u

/* ------------------------------------------------------------------------- */
/* MT-1 — connectivity over every algorithm x size x seed                    */
/* ------------------------------------------------------------------------- */

static void test_mt1_connectivity(void) {
  fbs_maze *m = make(64u, 64u);
  scratch *s = scratch_new(64u * 64u);
  uint32_t *dist = (uint32_t *)malloc(64u * 64u * sizeof(uint32_t));
  int a;
  size_t si;

  for (a = 0; a < ALGO_COUNT; ++a) {
    for (si = 0; si < SIZE_COUNT; ++si) {
      uint32_t w = SIZES[si][0], h = SIZES[si][1], n = w * h, seed;
      int ok = 1;
      for (seed = 0u; seed < SWEEP_SEEDS && ok; ++seed) {
        fbs_maze_params p = params_of(w, h, seed, a);
        uint32_t i;
        if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
        /* the module's own opinion */
        if (ok && fbs_maze_is_connected(m) != 1) ok = 0;
        /* an independent BFS reaches every cell */
        if (ok && ref_bfs(fbs_maze_cells(m), w, h, 0u, s) != n) ok = 0;
        /* and the published distance field has no unreachable entry */
        if (ok && fbs_maze_distances(m, 0u, 0u, dist, n) != FBS_MAZE_OK) ok = 0;
        if (ok)
          for (i = 0u; i < n; ++i)
            if (dist[i] == UINT32_MAX) ok = 0;
        if (!ok) printf("  MT-1 %s %ux%u seed %u\n", fbs_maze_algorithm_name(a), w, h, seed);
      }
      CHECK(ok);
    }
  }
  free(dist);
  scratch_free(s);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-2 — perfect mazes are perfect (the flagship witness; upstream Eller     */
/* fails this on 45-94% of seeds, defect M-1)                                */
/* ------------------------------------------------------------------------- */

static void test_mt2_perfect(void) {
  fbs_maze *m = make(64u, 64u);
  scratch *s = scratch_new(64u * 64u);
  int a;
  size_t si;

  for (a = 0; a < ALGO_COUNT; ++a) {
    for (si = 0; si < SIZE_COUNT; ++si) {
      uint32_t w = SIZES[si][0], h = SIZES[si][1], n = w * h, seed;
      int ok = 1;
      for (seed = 0u; seed < SWEEP_SEEDS && ok; ++seed) {
        fbs_maze_params p = params_of(w, h, seed, a);
        p.braid_percent = 0u;
        if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
        if (ok && fbs_maze_passage_count(m) != n - 1u) ok = 0;
        if (ok && ref_passages(fbs_maze_cells(m), w, h) != n - 1u) ok = 0;
        if (ok && ref_has_cycle(fbs_maze_cells(m), w, h, s)) ok = 0;
        if (ok && fbs_maze_is_perfect(m) != 1) ok = 0;
        if (!ok) printf("  MT-2 %s %ux%u seed %u\n", fbs_maze_algorithm_name(a), w, h, seed);
      }
      CHECK(ok);
    }
  }
  scratch_free(s);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-3 — edge symmetry, the reserved high nibble, border bookkeeping         */
/* ------------------------------------------------------------------------- */

static void test_mt3_symmetry(void) {
  fbs_maze *m = make(64u, 64u);
  int a;
  size_t si;

  for (a = 0; a < ALGO_COUNT; ++a) {
    for (si = 0; si < SIZE_COUNT; ++si) {
      uint32_t w = SIZES[si][0], h = SIZES[si][1], seed;
      int ok = 1;
      for (seed = 0u; seed < 8u && ok; ++seed) {
        fbs_maze_params p = params_of(w, h, seed, a);
        const uint8_t *c;
        uint32_t x, y, outward = 0u;
        p.braid_percent = (uint8_t)((seed % 2u) ? 40u : 0u);
        if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
        if (!ok) break;
        c = fbs_maze_cells(m);
        for (y = 0u; y < h && ok; ++y) {
          for (x = 0u; x < w && ok; ++x) {
            uint8_t v = c[y * w + x];
            unsigned d;
            if ((v & 0xF0u) != 0u) ok = 0; /* the reserved nibble is 0 */
            for (d = 0u; d < 4u; ++d) {
              uint32_t nb;
              if (!(v & DIRS[d])) continue;
              if (!nb_of(w, h, x, y, DIRS[d], &nb)) {
                ++outward; /* a border opening: the only unmirrored edge */
                continue;
              }
              if (!(c[nb] & FBS_MAZE_OPPOSITE(DIRS[d]))) ok = 0;
            }
          }
        }
        if (ok && outward != fbs_maze_border_opening_count(m)) ok = 0;
        if (ok && outward != 0u) ok = 0; /* generate() sealed the border */
      }
      CHECK(ok);
    }
  }

  /* now with border openings on all four sides */
  {
    fbs_maze_params p = params_of(9u, 6u, 3u, FBS_MAZE_BACKTRACKER);
    const uint8_t *c;
    uint32_t x, y, outward = 0u;
    unsigned d;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_border_opening_count(m) == 0u);
    CHECK(fbs_maze_open_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 8u, 5u, FBS_MAZE_S) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 0u, 3u, FBS_MAZE_W) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 8u, 1u, FBS_MAZE_E) == FBS_MAZE_OK);
    CHECK(fbs_maze_border_opening_count(m) == 4u);
    c = fbs_maze_cells(m);
    for (y = 0u; y < 6u; ++y)
      for (x = 0u; x < 9u; ++x) {
        uint8_t v = c[y * 9u + x];
        CHECK((v & 0xF0u) == 0u);
        for (d = 0u; d < 4u; ++d) {
          uint32_t nb;
          if (!(v & DIRS[d])) continue;
          if (!nb_of(9u, 6u, x, y, DIRS[d], &nb)) {
            ++outward;
            continue;
          }
          CHECK((c[nb] & FBS_MAZE_OPPOSITE(DIRS[d])) != 0u);
        }
      }
    CHECK(outward == 4u);
    CHECK(fbs_maze_close_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
    CHECK(fbs_maze_border_opening_count(m) == 3u);
    /* closing twice is idempotent, and the corner carries two outward bits */
    CHECK(fbs_maze_close_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
    CHECK(fbs_maze_border_opening_count(m) == 3u);
    CHECK(fbs_maze_open_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 0u, 0u, FBS_MAZE_W) == FBS_MAZE_OK);
    CHECK(fbs_maze_border_opening_count(m) == 5u);
    /* a direction that points INTO the grid is not a border */
    CHECK(fbs_maze_open_border(m, 0u, 0u, FBS_MAZE_E) == FBS_MAZE_E_INVALID);
    CHECK(fbs_maze_open_border(m, 4u, 3u, FBS_MAZE_N) == FBS_MAZE_E_INVALID);
    CHECK(fbs_maze_close_border(m, 4u, 3u, FBS_MAZE_S) == FBS_MAZE_E_INVALID);
    /* border openings are not interior passages */
    CHECK(fbs_maze_passage_count(m) == 9u * 6u - 1u);
    CHECK(fbs_maze_is_perfect(m) == 1);
  }
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-4 — reproducibility, golden mazes, golden PRNG vectors                 */
/* ------------------------------------------------------------------------- */

static void test_mt4_reproducibility(void) {
  fbs_maze *m = make(32u, 32u);
  fbs_maze *big = make(64u, 64u);
  uint8_t *first = (uint8_t *)malloc(32u * 32u);
  int a;

  /* (a) the same params 100 times in one process */
  for (a = 0; a < ALGO_COUNT; ++a) {
    fbs_maze_params p = params_of(21u, 13u, 0xABCDEF01u, a);
    int i, ok = 1;
    p.braid_percent = 25u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    memcpy(first, fbs_maze_cells(m), 21u * 13u);
    for (i = 0; i < 100; ++i) {
      if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
      if (memcmp(first, fbs_maze_cells(m), 21u * 13u) != 0) ok = 0;
    }
    CHECK(ok);
    /* (b) capacity must not influence output */
    CHECK(fbs_maze_generate(big, &p) == FBS_MAZE_OK);
    CHECK(memcmp(first, fbs_maze_cells(big), 21u * 13u) == 0);
  }
  free(first);
  fbs_maze_destroy(big);
  fbs_maze_destroy(m);

  /* (c) golden serialized maze per algorithm at 17x11 seed 0x5EED */
  {
    fbs_maze *g = make(17u, 11u);
    for (a = 0; a < ALGO_COUNT; ++a) {
      fbs_maze_params p = params_of(17u, 11u, 0x5EEDu, a);
      unsigned char blob[40u + 17u * 11u];
      size_t len = 0u;
      char name[128];
      CHECK(fbs_maze_generate(g, &p) == FBS_MAZE_OK);
      CHECK(fbs_maze_serialized_size(g) == sizeof blob);
      CHECK(fbs_maze_serialize(g, blob, sizeof blob, &len) == FBS_MAZE_OK);
      CHECK(len == sizeof blob);
      sprintf(name, "%s-17x11-5eed.bin", fbs_maze_algorithm_name(a));
      fixture(name, blob, len);
      /* the committed bytes must still load and still mean the same maze */
      if (!g_write_fixtures) {
        char path[512];
        size_t got = 0u;
        unsigned char *golden;
        sprintf(path, "%s/%s", g_fixture_dir, name);
        golden = read_file(path, &got);
        if (golden && got == len) {
          fbs_maze *back = NULL;
          CHECK(fbs_maze_deserialize(golden, got, NULL, NULL, &back) == FBS_MAZE_OK);
          if (back) {
            CHECK(memcmp(fbs_maze_cells(back), fbs_maze_cells(g), 17u * 11u) == 0);
            CHECK(fbs_maze_params_of(back)->algorithm == (uint8_t)a);
            CHECK(fbs_maze_params_of(back)->seed == 0x5EEDu);
            fbs_maze_destroy(back);
          }
        }
        free(golden);
      }
    }
    fbs_maze_destroy(g);
  }

  /* (d) golden PRNG vectors */
  {
    char text[4096];
    size_t at = 0u;
    fbs_maze_rng r;
    int i;
    static const uint64_t seeds[2] = {0u, 0xDEADBEEFCAFEF00DULL};
    at += (size_t)sprintf(
        text + at,
        "# fbs_maze PRNG golden vectors (MT-4d)\n"
        "# xoshiro128** 1.1, state seeded by splitmix64 over the 64-bit seed:\n"
        "#   w0 = splitmix64(); s[0] = low32(w0); s[1] = high32(w0);\n"
        "#   w1 = splitmix64(); s[2] = low32(w1); s[3] = high32(w1);\n"
        "# Bounded draws: threshold = (0 - bound) %% bound; reject x < threshold; return x %% bound.\n");
    for (i = 0; i < 2; ++i) {
      int k;
      fbs_maze_rng_seed(&r, seeds[i]);
      at += (size_t)sprintf(text + at, "\nseed 0x%016llX\nstate %08X %08X %08X %08X\nnext",
                            (unsigned long long)seeds[i], r.s[0], r.s[1], r.s[2], r.s[3]);
      for (k = 0; k < 16; ++k)
        at += (size_t)sprintf(text + at, " %08X", fbs_maze_rng_next(&r));
      at += (size_t)sprintf(text + at, "\n");
    }
    fbs_maze_rng_seed(&r, 0u);
    at += (size_t)sprintf(text + at, "\nseed 0x%016llX\nbelow(1000)", 0ULL);
    for (i = 0; i < 16; ++i)
      at += (size_t)sprintf(text + at, " %u", fbs_maze_rng_below(&r, 1000u));
    at += (size_t)sprintf(text + at, "\n");
    fixture("rng-vectors.txt", (const unsigned char *)text, at);

    /* and the two documented invariants of the bounded draw */
    fbs_maze_rng_seed(&r, 999u);
    {
      fbs_maze_rng before = r;
      CHECK(fbs_maze_rng_below(&r, 0u) == 0u);
      CHECK(before.s[0] == r.s[0] && before.s[1] == r.s[1] && before.s[2] == r.s[2] &&
            before.s[3] == r.s[3]);
      CHECK(fbs_maze_rng_below(&r, 1u) == 0u);
      CHECK(!(before.s[0] == r.s[0] && before.s[1] == r.s[1] && before.s[2] == r.s[2] &&
              before.s[3] == r.s[3]));
    }
    /* the seed-0 state is exactly the published splitmix64 output, split low
       then high; this is the one place the seeding order is pinned in code */
    fbs_maze_rng_seed(&r, 0u);
    CHECK(r.s[0] == 0x7B1DCDAFu && r.s[1] == 0xE220A839u);
    CHECK(r.s[2] == 0xA1B965F4u && r.s[3] == 0x6E789E6Au);
    /* NULL is not a crash */
    fbs_maze_rng_seed(NULL, 5u);
    CHECK(fbs_maze_rng_next(NULL) == 0u);
    CHECK(fbs_maze_rng_below(NULL, 10u) == 0u);
  }

  /* the recorded upstream Eller failure (MT-2, defect M-1) */
  {
    static const char note[] =
        "Upstream Eller cycle — recorded so the reason for the union-find is never lost.\n"
        "\n"
        "Defect M-1 of the upstream plugin. Not reproducible by this test binary on purpose:\n"
        "fbs_maze never had the defect, so there is nothing here to run.\n"
        "\n"
        "The instance, measured against a faithful C re-implementation of Eller.cpp\n"
        "plus FRandomStream:\n"
        "\n"
        "  algorithm : Eller (LowkeyMe UE5-MazeGenerator-Plugin, Eller.cpp)\n"
        "  seed      : 42\n"
        "  grid      : 6 x 6 direction cells (the plugin's MazeSize 11 x 11)\n"
        "  cells     : 36\n"
        "  edges     : 36        <- a perfect maze of 36 cells has exactly 35\n"
        "  components: 1\n"
        "  verdict   : one surplus edge, i.e. exactly one cycle\n"
        "\n"
        "Cause, in one line: Eller.cpp:38-43 and :95-100 relabel only the contiguous\n"
        "run starting at X+1 when two sets merge, where Buck's reference relabels every\n"
        "cell of the dissolved set. A set that sent two south passages down with a gap\n"
        "between them occupies non-contiguous columns in the next row, so the last-row\n"
        "join loop later reconnects two regions that were already connected through the\n"
        "rows above. Census over sizes 5-41 x 500 seeds: 4,474 of 9,500 mazes contain at\n"
        "least one cycle, 8,169 surplus edges in total, worst case 9 in one maze; the\n"
        "failure rate is 1% at MazeSize 7, 46% at 21 and 94% at 41, while the plugin's\n"
        "README.md:94 claims \"Generated mazes are perfect\".\n"
        "\n"
        "What fbs_maze does instead (src/maze.c, maze_eller): a real union-find over the\n"
        "row, so a merge joins whole sets and a passage is never carved between two cells\n"
        "that already share one. MT-2 asserts the result — passage_count == w*h - 1 and an\n"
        "independent DFS finding no back edge — for all eight algorithms over the whole\n"
        "size x seed matrix, and MT-10a asserts the second Eller defect (M-2) is gone too.\n";
    fixture("eller-cycle.txt", (const unsigned char *)note, sizeof note - 1u);
  }
}

/* ------------------------------------------------------------------------- */
/* MT-5 — per-algorithm characteristic properties                            */
/* ------------------------------------------------------------------------- */

static void test_mt5_sidewinder(void) {
  fbs_maze *m = make(33u, 33u);
  uint32_t seed;
  int ok_top = 1, ok_runs = 1, ok_count = 1;

  for (seed = 0u; seed < 64u; ++seed) {
    fbs_maze_params p = params_of(17u, 11u, seed, FBS_MAZE_SIDEWINDER);
    const uint8_t *c;
    uint32_t x, y;
    if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok_top = 0;
    c = fbs_maze_cells(m);
    /* the entire top row is one corridor */
    for (x = 0u; x + 1u < 17u; ++x)
      if (!(c[x] & FBS_MAZE_E)) ok_top = 0;
    if (c[16] & FBS_MAZE_E) ok_top = 0;
    if (c[0] & FBS_MAZE_N) ok_top = 0;
    /* every closed run in rows > 0 carries exactly one north passage */
    for (y = 1u; y < 11u; ++y) {
      uint32_t run_north = 0u;
      for (x = 0u; x < 17u; ++x) {
        uint8_t v = c[y * 17u + x];
        if (v & FBS_MAZE_N) ++run_north;
        if (!(v & FBS_MAZE_E) || x + 1u == 17u) {
          if (run_north != 1u) ok_runs = 0;
          run_north = 0u;
        }
      }
    }
    if (fbs_maze_passage_count(m) != 16u + 10u * 17u) ok_count = 0;
  }
  CHECK(ok_top);
  CHECK(ok_runs);
  CHECK(ok_count);

  /* the east bias is honoured: 0 gives no east passage below the top row,
     100 gives a full corridor in every row and one north per row */
  {
    fbs_maze_params p = params_of(9u, 7u, 11u, FBS_MAZE_SIDEWINDER);
    const uint8_t *c;
    uint32_t x, y, east = 0u;
    p.east_bias_percent = 0u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    c = fbs_maze_cells(m);
    for (y = 1u; y < 7u; ++y)
      for (x = 0u; x + 1u < 9u; ++x)
        if (c[y * 9u + x] & FBS_MAZE_E) ++east;
    CHECK(east == 0u);
    CHECK(fbs_maze_is_perfect(m) == 1);
    p.east_bias_percent = 100u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    c = fbs_maze_cells(m);
    east = 0u;
    for (y = 0u; y < 7u; ++y)
      for (x = 0u; x + 1u < 9u; ++x)
        if (c[y * 9u + x] & FBS_MAZE_E) ++east;
    CHECK(east == 7u * 8u);
    CHECK(fbs_maze_is_perfect(m) == 1);
  }
  fbs_maze_destroy(m);
}

static void test_mt5_eller_rows(void) {
  fbs_maze *a = make(24u, 24u);
  fbs_maze *b = make(24u, 24u);
  uint32_t seed;
  int ok = 1;
  /* Eller finishes each row before starting the next: growing the height
     cannot change the rows already emitted. */
  for (seed = 0u; seed < 64u && ok; ++seed) {
    uint32_t w = 13u, k = 9u;
    fbs_maze_params p = params_of(w, k, seed, FBS_MAZE_ELLER);
    if (fbs_maze_generate(a, &p) != FBS_MAZE_OK) ok = 0;
    p.height = k + 1u;
    if (fbs_maze_generate(b, &p) != FBS_MAZE_OK) ok = 0;
    if (ok && memcmp(fbs_maze_cells(a), fbs_maze_cells(b), (size_t)w * (k - 1u)) != 0) ok = 0;
    if (!ok) printf("  MT-5 eller row prefix differs at seed %u\n", seed);
  }
  CHECK(ok);
  fbs_maze_destroy(a);
  fbs_maze_destroy(b);
}

static void test_mt5_growing_tree_identity(void) {
  fbs_maze *x = make(40u, 40u);
  fbs_maze *y = make(40u, 40u);
  uint32_t seed;
  int ok100 = 1, ok0 = 1, okmid = 1;

  for (seed = 0u; seed < 32u; ++seed) {
    fbs_maze_params p = params_of(31u, 17u, seed, FBS_MAZE_BACKTRACKER);
    fbs_maze_params q = params_of(31u, 17u, seed, FBS_MAZE_GROWING_TREE);
    size_t n = 31u * 17u;
    q.newest_percent = 100u;
    if (fbs_maze_generate(x, &p) != FBS_MAZE_OK) ok100 = 0;
    if (fbs_maze_generate(y, &q) != FBS_MAZE_OK) ok100 = 0;
    if (memcmp(fbs_maze_cells(x), fbs_maze_cells(y), n) != 0) ok100 = 0;
    p.algorithm = (uint8_t)FBS_MAZE_PRIM;
    q.newest_percent = 0u;
    if (fbs_maze_generate(x, &p) != FBS_MAZE_OK) ok0 = 0;
    if (fbs_maze_generate(y, &q) != FBS_MAZE_OK) ok0 = 0;
    if (memcmp(fbs_maze_cells(x), fbs_maze_cells(y), n) != 0) ok0 = 0;
    /* and 50 is genuinely a blend, not an alias of either extreme */
    q.newest_percent = 50u;
    if (fbs_maze_generate(y, &q) != FBS_MAZE_OK) okmid = 0;
    if (memcmp(fbs_maze_cells(x), fbs_maze_cells(y), n) == 0) okmid = 0;
  }
  CHECK(ok100);
  CHECK(ok0);
  CHECK(okmid);
  fbs_maze_destroy(x);
  fbs_maze_destroy(y);
}

static void test_mt5_dead_end_texture(void) {
  fbs_maze *m = make(32u, 32u);
  unsigned long bt = 0u, pr = 0u, kr = 0u, gt = 0u;
  uint32_t seed;
  for (seed = 0u; seed < 32u; ++seed) {
    fbs_maze_params p = params_of(32u, 32u, seed, FBS_MAZE_BACKTRACKER);
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    bt += fbs_maze_dead_end_count(m);
    p.algorithm = (uint8_t)FBS_MAZE_PRIM;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    pr += fbs_maze_dead_end_count(m);
    p.algorithm = (uint8_t)FBS_MAZE_KRUSKAL;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    kr += fbs_maze_dead_end_count(m);
    p.algorithm = (uint8_t)FBS_MAZE_GROWING_TREE;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    gt += fbs_maze_dead_end_count(m);
  }
  /* Wikipedia: "depth-first search is biased toward long corridors, while
     Kruskal's/Prim's algorithms are biased toward many short dead ends". The
     backtracker is the outlier; no ordering is claimed among the other three,
     and measurement says growing_tree(50) is not between its two extremes —
     mixing newest with random interrupts corridors more often than either
     rule alone, so it produces the most dead ends of the four. */
  CHECK(bt < pr);
  CHECK(bt < kr);
  CHECK(bt < gt);
  CHECK(bt * 2u < pr); /* the gap is large, not marginal */
  printf("  MT-5 dead ends over 32 seeds at 32x32: backtracker %lu, growing_tree(50) %lu, "
         "kruskal %lu, prim %lu\n",
         bt, gt, kr, pr);
  fbs_maze_destroy(m);
}

static void test_mt5_kruskal_hak_identity(void) {
  fbs_maze *m = make(40u, 40u);
  uint32_t seed;
  int ok = 1;
  /* No structural claim beyond MT-2 is defensible for these two; the passage
     count identity is the one thing they must satisfy at every size. */
  for (seed = 0u; seed < 32u && ok; ++seed) {
    fbs_maze_params p = params_of(23u, 19u, seed, FBS_MAZE_KRUSKAL);
    if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
    if (fbs_maze_passage_count(m) != 23u * 19u - 1u) ok = 0;
    if (fbs_maze_is_perfect(m) != 1) ok = 0;
    p.algorithm = (uint8_t)FBS_MAZE_HUNT_AND_KILL;
    if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
    if (fbs_maze_passage_count(m) != 23u * 19u - 1u) ok = 0;
    if (fbs_maze_is_perfect(m) != 1) ok = 0;
  }
  CHECK(ok);
  fbs_maze_destroy(m);
}

/* Entrance/exit rendering: the ring is punched exactly where a border opening
 * is, and nowhere else (M-7, M-21). */
static void test_mt5_border_render(void) {
  fbs_maze *m = make(16u, 16u);
  fbs_maze_params p = params_of(7u, 5u, 2u, FBS_MAZE_KRUSKAL);
  size_t rw = 15u, rh = 11u, need = rw * rh;
  uint8_t *buf = (uint8_t *)malloc(need + 8u);
  uint32_t ow = 0u, oh = 0u;
  size_t i;

  CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
  CHECK(fbs_maze_render_size(m) == need);
  memset(buf, 0xA5, need + 8u);
  CHECK(fbs_maze_render(m, buf, need, &ow, &oh) == FBS_MAZE_OK);
  CHECK(ow == (uint32_t)rw && oh == (uint32_t)rh);
  CHECK(untouched(buf + need, 8u, 0xA5)); /* never writes past cap */
  /* sealed border */
  for (i = 0u; i < rw; ++i) {
    CHECK(buf[i] == 0u);
    CHECK(buf[(rh - 1u) * rw + i] == 0u);
  }
  for (i = 0u; i < rh; ++i) {
    CHECK(buf[i * rw] == 0u);
    CHECK(buf[i * rw + rw - 1u] == 0u);
  }
  /* one opening per side punches exactly one ring cell */
  CHECK(fbs_maze_open_border(m, 3u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
  CHECK(fbs_maze_open_border(m, 6u, 2u, FBS_MAZE_E) == FBS_MAZE_OK);
  CHECK(fbs_maze_open_border(m, 1u, 4u, FBS_MAZE_S) == FBS_MAZE_OK);
  CHECK(fbs_maze_open_border(m, 0u, 1u, FBS_MAZE_W) == FBS_MAZE_OK);
  memset(buf, 0xA5, need + 8u);
  CHECK(fbs_maze_render(m, buf, need, &ow, &oh) == FBS_MAZE_OK);
  CHECK(buf[0u * rw + 7u] == 1u);              /* N at x=3 */
  CHECK(buf[5u * rw + 14u] == 1u);             /* E at y=2 */
  CHECK(buf[10u * rw + 3u] == 1u);             /* S at x=1 */
  CHECK(buf[3u * rw + 0u] == 1u);              /* W at y=1 */
  CHECK(untouched(buf + need, 8u, 0xA5));
  {
    uint32_t ring = 0u;
    for (i = 0u; i < rw; ++i) {
      ring = (uint32_t)(ring + buf[i] + buf[(rh - 1u) * rw + i]);
    }
    for (i = 1u; i + 1u < rh; ++i) {
      ring = (uint32_t)(ring + buf[i * rw] + buf[i * rw + rw - 1u]);
    }
    CHECK(ring == 4u);
  }
  CHECK(fbs_maze_close_border(m, 3u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
  memset(buf, 0xA5, need);
  CHECK(fbs_maze_render(m, buf, need, NULL, NULL) == FBS_MAZE_OK);
  CHECK(buf[0u * rw + 7u] == 0u);
  free(buf);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-6 — braiding                                                           */
/* ------------------------------------------------------------------------- */

static void test_mt6_braiding(void) {
  fbs_maze *m = make(32u, 32u);
  fbs_maze *n2 = make(32u, 32u);
  int a;

  for (a = 0; a < ALGO_COUNT; ++a) {
    uint32_t seed;
    for (seed = 0u; seed < 8u; ++seed) {
      fbs_maze_params p = params_of(17u, 13u, seed, a);
      uint32_t d0, p0, d1, p1, opened;
      CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
      d0 = fbs_maze_dead_end_count(m);
      p0 = fbs_maze_passage_count(m);
      CHECK(d0 > 0u);
      /* the module's dead-end count agrees with an independent one */
      CHECK(ref_dead_ends(fbs_maze_cells(m), 17u * 13u) == d0);
      CHECK(ref_passages(fbs_maze_cells(m), 17u, 13u) == p0);

      /* braid_percent 0 changes nothing */
      p.braid_percent = 0u;
      CHECK(fbs_maze_generate(n2, &p) == FBS_MAZE_OK);
      CHECK(fbs_maze_dead_end_count(n2) == d0);
      CHECK(memcmp(fbs_maze_cells(m), fbs_maze_cells(n2), 17u * 13u) == 0);

      /* braid_percent 100 removes every dead end in a 2-D grid */
      p.braid_percent = 100u;
      CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
      d1 = fbs_maze_dead_end_count(m);
      p1 = fbs_maze_passage_count(m);
      opened = p1 - p0;
      CHECK(d1 == 0u);
      CHECK(opened > 0u);
      /* every opened wall is carved at a cell that is a dead end at that
         moment, so it removes at least that one and at most also its
         neighbour: opened <= removed <= 2 * opened */
      CHECK(opened <= d0 - d1);
      CHECK(d0 - d1 <= 2u * opened);
      CHECK(fbs_maze_is_perfect(m) == 0);
      CHECK(fbs_maze_is_connected(m) == 1);
      /* deterministic */
      CHECK(fbs_maze_generate(n2, &p) == FBS_MAZE_OK);
      CHECK(memcmp(fbs_maze_cells(m), fbs_maze_cells(n2), 17u * 13u) == 0);
    }
  }

  /* a partial braid still lands between the two extremes and stays connected */
  {
    fbs_maze_params p = params_of(24u, 24u, 77u, FBS_MAZE_KRUSKAL);
    uint32_t d0, d50, d100;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    d0 = fbs_maze_dead_end_count(m);
    p.braid_percent = 50u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    d50 = fbs_maze_dead_end_count(m);
    CHECK(fbs_maze_is_connected(m) == 1);
    CHECK(fbs_maze_is_perfect(m) == 0);
    p.braid_percent = 100u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    d100 = fbs_maze_dead_end_count(m);
    CHECK(d100 == 0u);
    CHECK(d50 < d0 && d50 > d100);
  }

  /* the documented exception: the two ends of a 1xN corridor are dead ends
     with no closed interior wall, so braiding cannot relieve them */
  {
    fbs_maze_params p = params_of(1u, 9u, 3u, FBS_MAZE_BACKTRACKER);
    p.braid_percent = 100u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_dead_end_count(m) == 2u);
    CHECK(fbs_maze_passage_count(m) == 8u);
    CHECK(fbs_maze_is_perfect(m) == 1);
  }
  fbs_maze_destroy(m);
  fbs_maze_destroy(n2);
}

/* ------------------------------------------------------------------------- */
/* MT-7 — bounds, degenerate sizes, every error status                       */
/* ------------------------------------------------------------------------- */

static void test_mt7_degenerate(void) {
  fbs_maze *m = make(64u, 64u);
  scratch *s = scratch_new(64u * 64u);
  int a;

  for (a = 0; a < ALGO_COUNT; ++a) {
    uint32_t k;
    /* 1x1: no passages, connected, perfect, renders 3x3 with a lone floor */
    fbs_maze_params p = params_of(1u, 1u, 4u, a);
    uint8_t buf[9];
    uint32_t ow = 0u, oh = 0u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_passage_count(m) == 0u);
    CHECK(fbs_maze_dead_end_count(m) == 0u);
    CHECK(fbs_maze_is_connected(m) == 1);
    CHECK(fbs_maze_is_perfect(m) == 1);
    CHECK(fbs_maze_render_size(m) == 9u);
    CHECK(fbs_maze_render(m, buf, sizeof buf, &ow, &oh) == FBS_MAZE_OK);
    CHECK(ow == 3u && oh == 3u);
    CHECK(buf[4] == 1u);
    CHECK(buf[0] + buf[1] + buf[2] + buf[3] + buf[5] + buf[6] + buf[7] + buf[8] == 0);

    /* 1xN and Nx1 are corridors for every algorithm — the case upstream
       Division leaves in five disconnected pieces (M-6) */
    for (k = 2u; k <= 9u; ++k) {
      fbs_maze_params q = params_of(1u, k, k, a);
      CHECK(fbs_maze_generate(m, &q) == FBS_MAZE_OK);
      CHECK(fbs_maze_passage_count(m) == k - 1u);
      CHECK(fbs_maze_is_perfect(m) == 1);
      CHECK(ref_bfs(fbs_maze_cells(m), 1u, k, 0u, s) == k);
      q.width = k;
      q.height = 1u;
      CHECK(fbs_maze_generate(m, &q) == FBS_MAZE_OK);
      CHECK(fbs_maze_passage_count(m) == k - 1u);
      CHECK(fbs_maze_is_perfect(m) == 1);
      CHECK(ref_bfs(fbs_maze_cells(m), k, 1u, 0u, s) == k);
    }
  }
  scratch_free(s);
  fbs_maze_destroy(m);
}

static void test_mt7_errors(void) {
  fbs_maze *m = NULL;
  fbs_maze *fresh = NULL;
  fbs_maze_config cfg;
  fbs_maze_params p;
  uint8_t cell = 0xA5u;
  uint32_t dist[16];
  uint32_t path[16];
  size_t len = 0u;
  unsigned char blob[64];

  /* create */
  CHECK(fbs_maze_create(NULL, NULL, NULL) == FBS_MAZE_E_INVALID);
  cfg.max_width = 0u;
  cfg.max_height = 8u;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_E_RANGE);
  cfg.max_width = 4097u;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_E_RANGE);
  cfg.max_width = 4096u;
  cfg.max_height = 4097u;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_E_RANGE);
  cfg.max_height = 0u;
  cfg.max_width = 8u;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_E_RANGE);
  CHECK(m == NULL);
  /* the documented ceiling: 4096 x 4096 is exactly 1<<24 cells and is legal,
     which is also why max_width * max_height can never exceed 1<<24 once both
     dimensions are inside [1, 4096] */
  cfg.max_width = 4096u;
  cfg.max_height = 4096u;
  CHECK(fbs_maze_memory_for(&cfg) > 0u);
  cfg.max_width = 3000u;
  cfg.max_height = 3000u;
  CHECK(fbs_maze_memory_for(&cfg) > 0u);
  CHECK(fbs_maze_memory_for(NULL) > 0u);
  {
    fbs_maze_config def = fbs_maze_config_default();
    CHECK(fbs_maze_memory_for(NULL) == fbs_maze_memory_for(&def));
  }
  m = make(16u, 16u);

  /* E_STATE: everything needs a generated maze */
  fresh = make(8u, 8u);
  CHECK(fbs_maze_width(fresh) == 0u);
  CHECK(fbs_maze_height(fresh) == 0u);
  CHECK(fbs_maze_cells(fresh) == NULL);
  CHECK(fbs_maze_params_of(fresh) == NULL);
  CHECK(fbs_maze_passage_count(fresh) == 0u);
  CHECK(fbs_maze_dead_end_count(fresh) == 0u);
  CHECK(fbs_maze_border_opening_count(fresh) == 0u);
  CHECK(fbs_maze_is_connected(fresh) == 0);
  CHECK(fbs_maze_is_perfect(fresh) == 0);
  CHECK(fbs_maze_render_size(fresh) == 0u);
  CHECK(fbs_maze_serialized_size(fresh) == 0u);
  CHECK(fbs_maze_cell(fresh, 0u, 0u, &cell) == FBS_MAZE_E_STATE);
  CHECK(cell == 0xA5u);
  CHECK(fbs_maze_is_open(fresh, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_set_passage(fresh, 0u, 0u, FBS_MAZE_E, 1) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_open_border(fresh, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_close_border(fresh, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_longest_path(fresh, NULL, NULL, NULL, NULL, NULL) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_solve(fresh, 0u, 0u, 0u, 0u, path, 16u, &len) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_distances(fresh, 0u, 0u, dist, 16u) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_render(fresh, blob, sizeof blob, NULL, NULL) == FBS_MAZE_E_STATE);
  CHECK(fbs_maze_serialize(fresh, blob, sizeof blob, &len) == FBS_MAZE_E_STATE);
  fbs_maze_destroy(fresh);

  /* generate: E_INVALID / E_RANGE / E_FULL, and the maze is left alone */
  p = params_of(8u, 8u, 1u, FBS_MAZE_BACKTRACKER);
  CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
  CHECK(fbs_maze_generate(NULL, &p) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_generate(m, NULL) == FBS_MAZE_E_INVALID);
  {
    uint8_t before[64];
    fbs_maze_params bad;
    memcpy(before, fbs_maze_cells(m), 64u);
    bad = p;
    bad.algorithm = (uint8_t)FBS_MAZE_ALGORITHM_COUNT;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_INVALID);
    bad.algorithm = 255u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_INVALID);
    bad = p;
    bad.width = 0u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_RANGE);
    bad = p;
    bad.height = 0u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_RANGE);
    bad = p;
    bad.east_bias_percent = 101u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_RANGE);
    bad = p;
    bad.newest_percent = 101u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_RANGE);
    bad = p;
    bad.braid_percent = 101u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_RANGE);
    bad = p;
    bad.width = 17u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_FULL);
    bad = p;
    bad.height = 17u;
    CHECK(fbs_maze_generate(m, &bad) == FBS_MAZE_E_FULL);
    CHECK(memcmp(before, fbs_maze_cells(m), 64u) == 0);
    CHECK(fbs_maze_width(m) == 8u && fbs_maze_height(m) == 8u);
  }

  /* coordinate and direction validation */
  CHECK(fbs_maze_cell(m, 8u, 0u, &cell) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_cell(m, 0u, 8u, &cell) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_cell(m, 0u, 0u, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_cell(NULL, 0u, 0u, &cell) == FBS_MAZE_E_INVALID);
  CHECK(cell == 0xA5u);
  CHECK(fbs_maze_cell(m, 0u, 0u, &cell) == FBS_MAZE_OK);
  CHECK(cell != 0xA5u);
  CHECK(fbs_maze_is_open(m, 0u, 0u, 3) == FBS_MAZE_E_INVALID);  /* two bits */
  CHECK(fbs_maze_is_open(m, 0u, 0u, 0) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_is_open(m, 0u, 0u, 16) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_is_open(NULL, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_is_open(m, 9u, 0u, FBS_MAZE_N) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_set_passage(m, 0u, 0u, FBS_MAZE_N, 1) == FBS_MAZE_E_INVALID); /* off grid */
  CHECK(fbs_maze_set_passage(m, 7u, 7u, FBS_MAZE_E, 1) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_set_passage(m, 0u, 0u, 5, 1) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_set_passage(NULL, 0u, 0u, FBS_MAZE_E, 1) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_open_border(NULL, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_open_border(m, 0u, 0u, 6) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_open_border(m, 8u, 0u, FBS_MAZE_N) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_solve(NULL, 0u, 0u, 0u, 0u, path, 16u, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_solve(m, 0u, 0u, 0u, 0u, path, 16u, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_solve(m, 8u, 0u, 0u, 0u, path, 16u, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_solve(m, 0u, 0u, 0u, 8u, path, 16u, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_solve(m, 0u, 0u, 0u, 0u, NULL, 16u, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_distances(m, 0u, 0u, NULL, 64u) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_distances(NULL, 0u, 0u, dist, 64u) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_distances(m, 8u, 0u, dist, 64u) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_render(NULL, blob, sizeof blob, NULL, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_render(m, NULL, 4u, NULL, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_serialize(NULL, blob, sizeof blob, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_serialize(m, blob, sizeof blob, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_serialize(m, NULL, 4u, &len) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_longest_path(NULL, NULL, NULL, NULL, NULL, NULL) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_deserialize(NULL, 40u, NULL, NULL, &fresh) == FBS_MAZE_E_INVALID);
  CHECK(fbs_maze_deserialize(blob, 40u, NULL, NULL, NULL) == FBS_MAZE_E_INVALID);

  /* NULL is not a crash anywhere */
  fbs_maze_destroy(NULL);
  CHECK(fbs_maze_width(NULL) == 0u);
  CHECK(fbs_maze_height(NULL) == 0u);
  CHECK(fbs_maze_cells(NULL) == NULL);
  CHECK(fbs_maze_params_of(NULL) == NULL);
  CHECK(fbs_maze_memory(NULL) == 0u);
  CHECK(fbs_maze_passage_count(NULL) == 0u);
  CHECK(fbs_maze_dead_end_count(NULL) == 0u);
  CHECK(fbs_maze_border_opening_count(NULL) == 0u);
  CHECK(fbs_maze_is_connected(NULL) == 0);
  CHECK(fbs_maze_is_perfect(NULL) == 0);
  CHECK(fbs_maze_render_size(NULL) == 0u);
  CHECK(fbs_maze_serialized_size(NULL) == 0u);

  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-8 — the BFS solver                                                     */
/* ------------------------------------------------------------------------- */

/* A hand-drawn 5x5 comb: row 0 is one corridor and every column hangs from it.
 * 24 passages over 25 cells, so it is a tree and every path below is unique. */
static void build_comb(fbs_maze *m) {
  uint32_t x, y;
  fbs_maze_params p = params_of(5u, 5u, 0u, FBS_MAZE_BACKTRACKER);
  CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
  for (y = 0u; y < 5u; ++y)
    for (x = 0u; x < 5u; ++x) {
      if (x + 1u < 5u) CHECK(fbs_maze_set_passage(m, x, y, FBS_MAZE_E, 0) == FBS_MAZE_OK);
      if (y + 1u < 5u) CHECK(fbs_maze_set_passage(m, x, y, FBS_MAZE_S, 0) == FBS_MAZE_OK);
    }
  CHECK(fbs_maze_passage_count(m) == 0u);
  for (x = 0u; x + 1u < 5u; ++x) CHECK(fbs_maze_set_passage(m, x, 0u, FBS_MAZE_E, 1) == FBS_MAZE_OK);
  for (x = 0u; x < 5u; ++x)
    for (y = 0u; y + 1u < 5u; ++y)
      CHECK(fbs_maze_set_passage(m, x, y, FBS_MAZE_S, 1) == FBS_MAZE_OK);
  CHECK(fbs_maze_passage_count(m) == 24u);
  CHECK(fbs_maze_is_perfect(m) == 1);
}

static void test_mt8_solver(void) {
  fbs_maze *m = make(16u, 16u);
  uint32_t path[64];
  size_t len = 0u;
  uint32_t i;

  build_comb(m);

  /* (a) start == end is a path of one cell */
  memset(path, 0xA5, sizeof path);
  CHECK(fbs_maze_solve(m, 2u, 2u, 2u, 2u, path, 64u, &len) == FBS_MAZE_OK);
  CHECK(len == 1u);
  CHECK(path[0] == 2u * 5u + 2u);

  /* (b) the L-shaped detour (0,0) -> (4,4): along row 0, then down column 4 */
  CHECK(fbs_maze_solve(m, 0u, 0u, 4u, 4u, path, 64u, &len) == FBS_MAZE_OK);
  CHECK(len == 9u);
  {
    static const uint32_t want[9] = {0u, 1u, 2u, 3u, 4u, 9u, 14u, 19u, 24u};
    int ok = 1;
    for (i = 0u; i < 9u; ++i)
      if (path[i] != want[i]) ok = 0;
    CHECK(ok);
  }

  /* (c) the long way round: (0,4) -> (4,4) climbs to row 0 and back down */
  CHECK(fbs_maze_solve(m, 0u, 4u, 4u, 4u, path, 64u, &len) == FBS_MAZE_OK);
  CHECK(len == 13u);
  {
    static const uint32_t want[13] = {20u, 15u, 10u, 5u,  0u,  1u, 2u,
                                      3u,  4u,  9u,  14u, 19u, 24u};
    int ok = 1;
    for (i = 0u; i < 13u; ++i)
      if (path[i] != want[i]) ok = 0;
    CHECK(ok);
  }

  /* (d) E_TRUNCATED one short, with the required length written and the
     caller's buffer untouched */
  memset(path, 0xA5, sizeof path);
  len = 0u;
  CHECK(fbs_maze_solve(m, 0u, 4u, 4u, 4u, path, 12u, &len) == FBS_MAZE_E_TRUNCATED);
  CHECK(len == 13u);
  CHECK(untouched(path, sizeof path, 0xA5));
  len = 0u;
  CHECK(fbs_maze_solve(m, 0u, 4u, 4u, 4u, NULL, 0u, &len) == FBS_MAZE_E_TRUNCATED);
  CHECK(len == 13u);

  /* (e) an isolated cell is E_UNREACHABLE, and so is longest_path */
  CHECK(fbs_maze_set_passage(m, 2u, 1u, FBS_MAZE_N, 0) == FBS_MAZE_OK);
  CHECK(fbs_maze_set_passage(m, 2u, 1u, FBS_MAZE_S, 0) == FBS_MAZE_OK);
  CHECK(fbs_maze_is_connected(m) == 0);
  memset(path, 0xA5, sizeof path);
  len = 0xdeadu;
  CHECK(fbs_maze_solve(m, 0u, 0u, 2u, 1u, path, 64u, &len) == FBS_MAZE_E_UNREACHABLE);
  CHECK(untouched(path, sizeof path, 0xA5));
  CHECK(len == 0xdeadu); /* untouched on a non-truncation error */
  {
    uint32_t ax = 0xA5u, ay = 0xA5u, bx = 0xA5u, by = 0xA5u, d = 0xA5u;
    CHECK(fbs_maze_longest_path(m, &ax, &ay, &bx, &by, &d) == FBS_MAZE_E_UNREACHABLE);
    CHECK(ax == 0xA5u && ay == 0xA5u && bx == 0xA5u && by == 0xA5u && d == 0xA5u);
  }

  /* (f) the fixed N,E,S,W neighbour order is observable when several shortest
     paths exist: a fully open 3x3 has two, and BFS must pick the eastward one */
  {
    fbs_maze_params p = params_of(3u, 3u, 0u, FBS_MAZE_BACKTRACKER);
    uint32_t x, y;
    static const uint32_t want[5] = {0u, 1u, 2u, 5u, 8u};
    int ok = 1;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    for (y = 0u; y < 3u; ++y)
      for (x = 0u; x < 3u; ++x) {
        if (x + 1u < 3u) CHECK(fbs_maze_set_passage(m, x, y, FBS_MAZE_E, 1) == FBS_MAZE_OK);
        if (y + 1u < 3u) CHECK(fbs_maze_set_passage(m, x, y, FBS_MAZE_S, 1) == FBS_MAZE_OK);
      }
    CHECK(fbs_maze_passage_count(m) == 12u);
    CHECK(fbs_maze_solve(m, 0u, 0u, 2u, 2u, path, 64u, &len) == FBS_MAZE_OK);
    CHECK(len == 5u);
    for (i = 0u; i < 5u; ++i)
      if (path[i] != want[i]) ok = 0;
    CHECK(ok);
  }
  fbs_maze_destroy(m);
}

/* Counts the simple paths between two cells with an explicit stack. */
static uint32_t count_simple_paths(const uint8_t *cells, uint32_t w, uint32_t h, uint32_t a,
                                   uint32_t b) {
  uint32_t n = w * h, count = 0u, sp = 0u;
  uint8_t *on = (uint8_t *)calloc(n, 1u);
  uint32_t *stack = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
  uint8_t *next_dir = (uint8_t *)calloc(n, 1u);

  stack[sp] = a;
  on[a] = 1u;
  next_dir[a] = 0u;
  ++sp;
  while (sp > 0u) {
    uint32_t c = stack[sp - 1u];
    if (c == b) {
      ++count;
      on[c] = 0u;
      --sp;
      continue;
    }
    if (next_dir[c] >= 4u) {
      on[c] = 0u;
      --sp;
      continue;
    }
    {
      uint8_t d = next_dir[c];
      uint32_t nb;
      next_dir[c] = (uint8_t)(d + 1u);
      if (!(cells[c] & DIRS[d])) continue;
      if (!nb_of(w, h, c % w, c / w, DIRS[d], &nb)) continue;
      if (on[nb]) continue;
      on[nb] = 1u;
      next_dir[nb] = 0u;
      stack[sp++] = nb;
    }
  }
  free(on);
  free(stack);
  free(next_dir);
  return count;
}

static void test_mt8_generated(void) {
  fbs_maze *m = make(32u, 32u);
  scratch *s = scratch_new(32u * 32u);
  uint32_t *dist = (uint32_t *)malloc(32u * 32u * sizeof(uint32_t));
  uint32_t *path = (uint32_t *)malloc(32u * 32u * sizeof(uint32_t));
  int a;

  for (a = 0; a < ALGO_COUNT; ++a) {
    uint32_t seed;
    int ok = 1;
    for (seed = 0u; seed < 8u && ok; ++seed) {
      fbs_maze_params p = params_of(19u, 15u, seed, a);
      uint32_t w = 19u, h = 15u, n = w * h, ax = 0u, ay = 0u, bx = 0u, by = 0u, d = 0u, i;
      size_t len = 0u;
      if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
      if (fbs_maze_distances(m, 0u, 0u, dist, n) != FBS_MAZE_OK) ok = 0;
      if (fbs_maze_solve(m, 0u, 0u, w - 1u, h - 1u, path, n, &len) != FBS_MAZE_OK) ok = 0;
      /* the path length is the distance plus one */
      if (ok && len != (size_t)dist[n - 1u] + 1u) ok = 0;
      /* the path is a valid walk with every step through an open passage */
      if (ok) {
        if (path[0] != 0u || path[len - 1u] != n - 1u) ok = 0;
        for (i = 0u; i + 1u < (uint32_t)len; ++i) {
          uint32_t c = path[i], e = path[i + 1u], nb;
          unsigned dd;
          int step = 0;
          for (dd = 0u; dd < 4u; ++dd) {
            if (!(fbs_maze_cells(m)[c] & DIRS[dd])) continue;
            if (!nb_of(w, h, c % w, c / w, DIRS[dd], &nb)) continue;
            if (nb == e) step = 1;
          }
          if (!step) ok = 0;
        }
      }
      /* the published distance field agrees with an independent BFS */
      if (ok) {
        (void)ref_bfs(fbs_maze_cells(m), w, h, 0u, s);
        for (i = 0u; i < n; ++i)
          if (s->dist[i] != dist[i]) ok = 0;
      }
      /* longest_path agrees with the maximum of a full distance field */
      if (ok && fbs_maze_longest_path(m, &ax, &ay, &bx, &by, &d) != FBS_MAZE_OK) ok = 0;
      if (ok) {
        uint32_t best = 0u, j;
        if (fbs_maze_distances(m, ax, ay, dist, n) != FBS_MAZE_OK) ok = 0;
        for (j = 0u; j < n; ++j)
          if (dist[j] != UINT32_MAX && dist[j] > best) best = dist[j];
        if (best != d) ok = 0;
        if (dist[by * w + bx] != d) ok = 0;
        /* and it really is the diameter of this tree */
        for (j = 0u; j < n; ++j) {
          uint32_t k, far = 0u;
          if (j % 37u != 0u) continue; /* a spot check, not all-pairs */
          if (fbs_maze_distances(m, j % w, j / w, dist, n) != FBS_MAZE_OK) ok = 0;
          for (k = 0u; k < n; ++k)
            if (dist[k] != UINT32_MAX && dist[k] > far) far = dist[k];
          if (far > d) ok = 0;
        }
      }
      if (!ok) printf("  MT-8 %s seed %u\n", fbs_maze_algorithm_name(a), seed);
    }
    CHECK(ok);
  }

  /* uniqueness: in a perfect maze there is exactly one simple path */
  {
    uint32_t seed;
    int ok = 1;
    for (seed = 0u; seed < 8u && ok; ++seed) {
      fbs_maze_params p = params_of(5u, 4u, seed, FBS_MAZE_KRUSKAL);
      if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
      if (count_simple_paths(fbs_maze_cells(m), 5u, 4u, 0u, 19u) != 1u) ok = 0;
      if (count_simple_paths(fbs_maze_cells(m), 5u, 4u, 7u, 12u) != 1u) ok = 0;
    }
    CHECK(ok);
    /* and a braided maze has more than one, which is the point of braiding */
    {
      fbs_maze_params p = params_of(5u, 4u, 3u, FBS_MAZE_KRUSKAL);
      p.braid_percent = 100u;
      CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
      CHECK(count_simple_paths(fbs_maze_cells(m), 5u, 4u, 0u, 19u) > 1u);
    }
  }

  free(path);
  free(dist);
  scratch_free(s);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-9 — the shuffle is fair, and the bounded draw is unbiased               */
/* ------------------------------------------------------------------------- */

/* The module's Fisher-Yates, transcribed: for i = 0 .. n-2, swap a[i] with
 * a[i + rng_below(n - i)]. src/maze.c maze_shuffle is this and nothing else;
 * test_mt9_kruskal_cross_check below pins that claim against the real
 * generator rather than trusting this comment. */
static void ref_shuffle(fbs_maze_rng *r, uint32_t *a, uint32_t n) {
  uint32_t i;
  for (i = 0u; i + 1u < n; ++i) {
    uint32_t j = i + fbs_maze_rng_below(r, n - i);
    uint32_t t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

static void test_mt9_shuffle_fairness(void) {
  uint32_t counts[24];
  fbs_maze_rng r;
  unsigned long trials = g_fast ? 200000ul : 2000000ul;
  unsigned long i;
  uint32_t k;
  double chi = 0.0, expect;
  int all_in_band = 1;
  static const uint32_t perm_index[24][4] = {
      {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
      {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
      {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
      {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

  memset(counts, 0, sizeof counts);
  fbs_maze_rng_seed(&r, 0x5EEDu);
  for (i = 0ul; i < trials; ++i) {
    uint32_t a[4];
    a[0] = 0u;
    a[1] = 1u;
    a[2] = 2u;
    a[3] = 3u;
    ref_shuffle(&r, a, 4u);
    for (k = 0u; k < 24u; ++k) {
      if (a[0] == perm_index[k][0] && a[1] == perm_index[k][1] && a[2] == perm_index[k][2] &&
          a[3] == perm_index[k][3]) {
        ++counts[k];
        break;
      }
    }
  }
  expect = (double)trials / 24.0;
  for (k = 0u; k < 24u; ++k) {
    double diff = (double)counts[k] - expect;
    chi += diff * diff / expect;
    if (diff > expect * 0.01 || -diff > expect * 0.01) all_in_band = 0;
  }
  printf("  MT-9 Fisher-Yates: %lu shuffles, chi-square %.2f on 23 df (p=0.001 critical 49.73)\n",
         trials, chi);
  /* The chi-square is the real criterion: it is the whole 24-bucket statement
     at a stated significance, and the source's naive shuffle scores 59,432 on
     it (M-3). The +/-1% band the test plan also names is a per-bucket spot
     check at ONE fixed seed, not a statistical test: at 2,000,000 samples
     sigma = sqrt(N * (1/24) * (23/24)) = 282.6 and the band is N/24 * 0.01 =
     833.3, i.e. 2.95 sigma, which a fair shuffle would trip about 8% of the
     time across 24 buckets. It is kept because the plan asks for it and
     because the seed is fixed, so it is reproducible either way; it is not
     evidence of unfairness on its own. */
  CHECK(chi < 49.73);
  if (!g_fast) {
    CHECK(all_in_band); /* every permutation within +/-1% (2.95 sigma) of N/24 */
  } else {
    printf("  MT-9 FAST: the +/-1%% band is skipped (tighter than 1 sigma at %lu samples)\n",
           trials);
  }
  for (k = 0u; k < 24u; ++k) CHECK(counts[k] > 0u);
}

static void test_mt9_bounded_draw(void) {
  static const uint32_t bounds[5] = {2u, 3u, 5u, 7u, 1000u};
  /* chi-square critical values at p = 0.001 for 1, 2, 4, 6 and 999 df */
  static const double critical[5] = {10.828, 13.816, 18.467, 22.458, 1142.0};
  unsigned long draws = g_fast ? (1ul << 18) : (1ul << 22);
  int b;

  for (b = 0; b < 5; ++b) {
    uint32_t bound = bounds[b];
    uint32_t *counts = (uint32_t *)calloc(bound, sizeof(uint32_t));
    fbs_maze_rng r;
    unsigned long i;
    double chi = 0.0, expect = (double)draws / (double)bound;
    int nonzero = 1;
    fbs_maze_rng_seed(&r, 0x1234567890ABCDEFULL + (uint64_t)(unsigned)b);
    for (i = 0ul; i < draws; ++i) {
      uint32_t v = fbs_maze_rng_below(&r, bound);
      if (v >= bound) {
        nonzero = 0;
        break;
      }
      ++counts[v];
    }
    for (i = 0ul; i < (unsigned long)bound; ++i) {
      double diff = (double)counts[i] - expect;
      chi += diff * diff / expect;
      if (counts[i] == 0u) nonzero = 0;
    }
    printf("  MT-9 rng_below(%u): %lu draws, chi-square %.2f on %u df (critical %.2f)\n", bound,
           draws, chi, bound - 1u, critical[b]);
    CHECK(nonzero);
    CHECK(chi < critical[b]);
    free(counts);
  }
}

/* How many rng_next outputs one rng_below call consumed: step a copy of the
 * pre-call state forward until it matches the post-call state. xoshiro128**
 * has period 2^128-1, so within a handful of steps the match is exact. */
static uint32_t below_draw_count(fbs_maze_rng *r, uint32_t bound, uint32_t *out_value) {
  fbs_maze_rng probe = *r;
  uint32_t n = 0u;
  uint32_t v = fbs_maze_rng_below(r, bound);
  while (n < 64u && !(probe.s[0] == r->s[0] && probe.s[1] == r->s[1] && probe.s[2] == r->s[2] &&
                      probe.s[3] == r->s[3])) {
    (void)fbs_maze_rng_next(&probe);
    ++n;
  }
  if (out_value) *out_value = v;
  return n;
}

/* MZ-1 (module review) — the rejection branch itself. Every bound used above is
 * small enough that 2^32 mod bound is a rounding error, so a mutant that just
 * returned `next() % bound` would pass all of them. These bounds are not: at
 * 0x80000001 the threshold is 0x7FFFFFFF, half of all outputs are thrown away,
 * and both the discarded draws and the exact results become observable.
 *
 * Derivation of the pinned constants, by hand, from the golden stream for
 * seed 0 in tests/fixtures/maze/rng-vectors.txt:
 *
 *   raw next():  DEC9045D 9A089D75 AB77D362 C3E16405 5C95A8DA 60DEA056 C25A5140 ...
 *   bound      = 0x80000001
 *   threshold  = (0 - bound) % bound = 0x100000000 - 0x80000001 = 0x7FFFFFFF
 *                -> reject x < 0x7FFFFFFF, i.e. the LOW half of the range
 *
 *   call 0:  DEC9045D >= threshold  accept   DEC9045D - 80000001 = 5EC9045C   1 draw
 *   call 4:  5C95A8DA <  threshold  reject
 *            60DEA056 <  threshold  reject
 *            C25A5140 >= threshold  accept   C25A5140 - 80000001 = 425A513F   3 draws
 *
 * Two mutants die on those two lines:
 *   - `x % bound` with no rejection at all returns 5C95A8DA at call 4, in one
 *     draw instead of three;
 *   - the OTHER rejection formulation (accept when x - x%bound <= UINT32_MAX -
 *     bound + 1, i.e. discard the HIGH residue, accepting only x <= 0x80000000)
 *     rejects DEC9045D and would answer call 0 with 5C95A8DA after five draws.
 *   Both are excluded, so this pins the documented formula and not merely
 *   "some rejection happens". */
static void test_mt9_rejection_branch(void) {
  static const uint32_t big[6] = {0x80000001u, 0xC0000000u, 0xFFFFFFFFu,
                                  0x80000000u, 0x7FFFFFFFu, 0xAAAAAAABu};
  /* (c) hand-computed results and draw counts, seed 0, bound 0x80000001 */
  static const uint32_t want_val[8] = {0x5EC9045Cu, 0x1A089D74u, 0x2B77D361u, 0x43E16404u,
                                       0x425A513Fu, 0x24290613u, 0x1E0525AEu, 0x153D37B8u};
  static const uint32_t want_draws[8] = {1u, 1u, 1u, 1u, 3u, 1u, 1u, 1u};
  fbs_maze_rng r;
  int i;
  unsigned long j, draws = g_fast ? 5000ul : 20000ul;

  fbs_maze_rng_seed(&r, 0u);
  for (i = 0; i < 8; ++i) {
    uint32_t v = 0u;
    uint32_t n = below_draw_count(&r, 0x80000001u, &v);
    CHECK(v == want_val[i]);
    CHECK(n == want_draws[i]);
  }
  /* the two mutant answers, spelled out so the intent survives a refactor */
  fbs_maze_rng_seed(&r, 0u);
  CHECK(fbs_maze_rng_below(&r, 0x80000001u) != 0x5C95A8DAu); /* not the high-residue variant */
  fbs_maze_rng_seed(&r, 0u);
  for (i = 0; i < 4; ++i) (void)fbs_maze_rng_below(&r, 0x80000001u);
  CHECK(fbs_maze_rng_below(&r, 0x80000001u) != 0x5C95A8DAu); /* not modulo alone */

  /* a bound of 0xFFFFFFFF has threshold 1, so only the single output 0 is ever
     rejected and the results are the raw stream verbatim */
  fbs_maze_rng_seed(&r, 0u);
  {
    fbs_maze_rng raw;
    fbs_maze_rng_seed(&raw, 0u);
    for (i = 0; i < 16; ++i) {
      uint32_t x = fbs_maze_rng_next(&raw);
      uint32_t v = 0u;
      uint32_t n = below_draw_count(&r, 0xFFFFFFFFu, &v);
      CHECK(n == 1u);
      CHECK(v == (x == 0xFFFFFFFFu ? 0u : x));
    }
  }

  /* (a) every result is inside its bound, over many draws, for every bound */
  for (i = 0; i < 6; ++i) {
    uint32_t bound = big[i];
    int in_range = 1, saw_low = 0, saw_high = 0;
    fbs_maze_rng_seed(&r, 0xFEEDFACEu + (uint64_t)(unsigned)i);
    for (j = 0ul; j < draws; ++j) {
      uint32_t v = fbs_maze_rng_below(&r, bound);
      if (v >= bound) in_range = 0;
      if (v < bound / 2u) saw_low = 1;
      else saw_high = 1;
    }
    CHECK(in_range);
    CHECK(saw_low && saw_high);
  }

  /* (b) the branch is really taken: count the draws each call consumed */
  {
    static const uint32_t census[3] = {0x80000001u, 0xC0000000u, 0xFFFFFFFFu};
    /* expected share of calls needing a second draw: threshold / 2^32 */
    static const double lo[3] = {0.40, 0.18, 0.0};
    static const double hi[3] = {0.60, 0.32, 0.0001};
    for (i = 0; i < 3; ++i) {
      unsigned long multi = 0ul, total = 0ul;
      uint32_t worst = 0u;
      fbs_maze_rng_seed(&r, 0x0BADC0DEu + (uint64_t)(unsigned)i);
      for (j = 0ul; j < draws; ++j) {
        uint32_t n = below_draw_count(&r, census[i], NULL);
        if (n == 0u || n >= 64u) worst = 0xFFFFFFFFu; /* never: a draw always happens */
        if (n > 1u) ++multi;
        if (n > worst && worst != 0xFFFFFFFFu) worst = n;
        total += n;
      }
      printf("  MZ-1 rng_below(0x%08X): %lu of %lu calls needed more than one rng_next, "
             "max %u, mean %.4f\n",
             census[i], multi, draws, worst, (double)total / (double)draws);
      CHECK(worst != 0xFFFFFFFFu);
      CHECK((double)multi >= lo[i] * (double)draws);
      CHECK((double)multi <= hi[i] * (double)draws);
    }
    /* the literal review requirement, stated on its own: at least one call in
       the suite consumed more than one output */
    fbs_maze_rng_seed(&r, 1u);
    {
      int seen_multi = 0;
      for (j = 0ul; j < 64ul && !seen_multi; ++j)
        if (below_draw_count(&r, 0x80000001u, NULL) > 1u) seen_multi = 1;
      CHECK(seen_multi);
    }
  }
}

/* Pins the module's internal shuffle: an independent transcription of the
 * whole Kruskal draw sequence (edge order, Fisher-Yates, then a naive
 * union-find, whose internals cannot change the outcome) must reproduce the
 * generator byte for byte. */
static uint32_t uf_root(uint32_t *p, uint32_t x) {
  while (p[x] != x) x = p[x];
  return x;
}

static void test_mt9_kruskal_cross_check(void) {
  fbs_maze *m = make(16u, 16u);
  uint32_t seed;
  int ok = 1;
  for (seed = 0u; seed < 16u && ok; ++seed) {
    uint32_t w = 9u, h = 7u, n = w * h, x, y, count = 0u, i;
    uint32_t *edges = (uint32_t *)malloc(2u * (size_t)n * sizeof(uint32_t));
    uint32_t *parent = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    uint8_t *cells = (uint8_t *)calloc(n, 1u);
    fbs_maze_rng r;
    fbs_maze_params p = params_of(w, h, seed, FBS_MAZE_KRUSKAL);

    for (i = 0u; i < n; ++i) parent[i] = i;
    for (y = 0u; y < h; ++y)
      for (x = 0u; x < w; ++x) {
        uint32_t c = y * w + x;
        if (x + 1u < w) edges[count++] = c << 1;
        if (y + 1u < h) edges[count++] = (c << 1) | 1u;
      }
    CHECK(count == (w - 1u) * h + w * (h - 1u));
    fbs_maze_rng_seed(&r, seed);
    ref_shuffle(&r, edges, count);
    for (i = 0u; i < count; ++i) {
      uint32_t e = edges[i], c = e >> 1u;
      uint32_t nb = (e & 1u) ? c + w : c + 1u;
      uint32_t ra = uf_root(parent, c), rb = uf_root(parent, nb);
      if (ra == rb) continue;
      parent[rb] = ra;
      cells[c] = (uint8_t)(cells[c] | ((e & 1u) ? FBS_MAZE_S : FBS_MAZE_E));
      cells[nb] = (uint8_t)(cells[nb] | ((e & 1u) ? FBS_MAZE_N : FBS_MAZE_W));
    }
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    if (memcmp(cells, fbs_maze_cells(m), n) != 0) {
      ok = 0;
      printf("  MT-9 kruskal cross-check differs at seed %u\n", seed);
    }
    free(edges);
    free(parent);
    free(cells);
  }
  CHECK(ok);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-10 — the structural-bias regressions                                   */
/* ------------------------------------------------------------------------- */

/* (a) Eller: upstream can never carve two adjacent east passages in a
 * non-final row (M-2: measured max run 1 against 20 in the final row). Here the
 * middle rows follow the same rule as every other row, so runs longer than one
 * occur, and the run-length histograms of two different middle rows are
 * statistically indistinguishable. The final row is still denser by
 * construction — it joins EVERY remaining pair of distinct sets, which is what
 * makes the maze connected — so the plan's "indistinguishable from the other
 * rows" is asserted between middle rows, and the final row is asserted to be
 * denser rather than 20-against-1 different. */
static void test_mt10_eller_runs(void) {
  fbs_maze *m = make(24u, 24u);
  uint32_t w = 21u, h = 21u;
  unsigned long seeds = g_fast ? 250ul : 2000ul;
  unsigned long seed;
  uint32_t hist_a[8], hist_b[8], maxmid = 0u, maxlast = 0u;
  unsigned long east_mid = 0u, east_last = 0u;
  double chi = 0.0;
  int k;

  memset(hist_a, 0, sizeof hist_a);
  memset(hist_b, 0, sizeof hist_b);
  for (seed = 0ul; seed < seeds; ++seed) {
    fbs_maze_params p = params_of(w, h, seed, FBS_MAZE_ELLER);
    const uint8_t *c;
    uint32_t x, y;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    c = fbs_maze_cells(m);
    for (y = 0u; y < h; ++y) {
      uint32_t run = 0u;
      for (x = 0u; x < w; ++x) {
        int open = (x + 1u < w) && (c[y * w + x] & FBS_MAZE_E);
        if (open) {
          ++run;
          if (y + 1u == h) {
            ++east_last;
            if (run > maxlast) maxlast = run;
          } else {
            ++east_mid;
            if (run > maxmid) maxmid = run;
          }
        } else {
          if (run > 0u) {
            uint32_t bucket = run > 7u ? 7u : run;
            if (y == 3u) ++hist_a[bucket];
            if (y == 11u) ++hist_b[bucket];
          }
          run = 0u;
        }
      }
    }
  }
  printf("  MT-10a Eller over %lu seeds at %ux%u: max east run middle %u, final %u; "
         "east passages middle %lu, final %lu\n",
         seeds, w, h, maxmid, maxlast, east_mid, east_last);
  /* the M-2 regression itself: adjacent east passages in a non-final row */
  CHECK(maxmid >= 2u);
  /* two middle rows are drawn from the same rule */
  for (k = 1; k < 8; ++k) {
    double a = (double)hist_a[k], b = (double)hist_b[k];
    if (a + b > 0.0) {
      double e = (a + b) / 2.0;
      chi += (a - e) * (a - e) / e + (b - e) * (b - e) / e;
    }
  }
  printf("  MT-10a middle-row run-length chi-square %.2f on <=7 df (p=0.001 critical 24.32)\n",
         chi);
  CHECK(chi < 24.32);
  /* and the final row is denser, because it joins every distinct pair */
  CHECK(east_last * (h - 1u) > east_mid);
  fbs_maze_destroy(m);
}

/* Independent structural verifier for Recursive Division: is this region a
 * valid guillotine decomposition? A region with an extent of 1 must be an
 * uninterrupted corridor; a region with both extents >= 2 must carry a
 * full-span wall line with exactly one gap, oriented per the aspect-ratio rule
 * (taller than wide cuts horizontally, wider than tall vertically, square
 * either way), splitting it into two regions that are themselves valid. This
 * knows nothing about the PRNG or the draw order — it reads only the finished
 * maze — so it is a genuine second opinion on "the maze decomposes into
 * rectangles and every maximal straight wall run spans a full sub-region".
 * Several lines can qualify at one level (a horizontal cut inside the top half
 * also spans the parent's full width), so it backtracks, under a node budget.
 * Recursion is fine here; the module under test is the thing that must not
 * recurse. */
static long g_div_budget;

static int div_ok(const uint8_t *cells, uint32_t w, uint32_t rx, uint32_t ry, uint32_t rw,
                  uint32_t rh) {
  uint32_t i, j;
  if (--g_div_budget < 0) return -1;
  if (rw == 1u && rh == 1u) return 1;
  if (rw == 1u) {
    for (j = 0u; j + 1u < rh; ++j)
      if (!(cells[(ry + j) * w + rx] & FBS_MAZE_S)) return 0;
    return 1;
  }
  if (rh == 1u) {
    for (i = 0u; i + 1u < rw; ++i)
      if (!(cells[ry * w + rx + i] & FBS_MAZE_E)) return 0;
    return 1;
  }
  if (rh >= rw) { /* horizontal cuts are allowed */
    for (j = 0u; j + 1u < rh; ++j) {
      uint32_t open = 0u;
      int a, b;
      for (i = 0u; i < rw; ++i)
        if (cells[(ry + j) * w + rx + i] & FBS_MAZE_S) ++open;
      if (open != 1u) continue;
      a = div_ok(cells, w, rx, ry, rw, j + 1u);
      if (a < 0) return -1;
      if (!a) continue;
      b = div_ok(cells, w, rx, ry + j + 1u, rw, rh - j - 1u);
      if (b < 0) return -1;
      if (b) return 1;
    }
  }
  if (rw >= rh) { /* vertical cuts are allowed */
    for (i = 0u; i + 1u < rw; ++i) {
      uint32_t open = 0u;
      int a, b;
      for (j = 0u; j < rh; ++j)
        if (cells[(ry + j) * w + rx + i] & FBS_MAZE_E) ++open;
      if (open != 1u) continue;
      a = div_ok(cells, w, rx, ry, i + 1u, rh);
      if (a < 0) return -1;
      if (!a) continue;
      b = div_ok(cells, w, rx + i + 1u, ry, rw - i - 1u, rh);
      if (b < 0) return -1;
      if (b) return 1;
    }
  }
  return 0;
}

static void test_mt10_division_structure(void) {
  fbs_maze *m = make(24u, 24u);
  uint32_t seed;
  int ok = 1, budget_hit = 0, negative_control = 0;
  static const uint32_t shapes[4][2] = {{16u, 16u}, {13u, 21u}, {21u, 13u}, {7u, 4u}};
  int s;

  for (s = 0; s < 4 && ok; ++s) {
    for (seed = 0u; seed < 16u && ok; ++seed) {
      fbs_maze_params p = params_of(shapes[s][0], shapes[s][1], seed, FBS_MAZE_DIVISION);
      int r;
      CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
      g_div_budget = 2000000L;
      r = div_ok(fbs_maze_cells(m), shapes[s][0], 0u, 0u, shapes[s][0], shapes[s][1]);
      if (r < 0) {
        budget_hit = 1;
      } else if (!r) {
        ok = 0;
        printf("  MT-10b division %ux%u seed %u is not a guillotine decomposition\n", shapes[s][0],
               shapes[s][1], seed);
      }
    }
  }
  CHECK(ok);
  CHECK(!budget_hit);
  /* negative control: a backtracker maze of the same shape is not one */
  for (seed = 0u; seed < 16u; ++seed) {
    fbs_maze_params p = params_of(16u, 16u, seed, FBS_MAZE_BACKTRACKER);
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    g_div_budget = 2000000L;
    if (div_ok(fbs_maze_cells(m), 16u, 0u, 0u, 16u, 16u) == 0) ++negative_control;
  }
  printf("  MT-10b guillotine verifier rejects %u of 16 backtracker mazes at 16x16\n",
         negative_control);
  CHECK(negative_control == 16u);
  fbs_maze_destroy(m);
}

/* (b) Division: the first cut is predicted exactly from the public PRNG, which
 * pins both the split range (M-4: a one-cell sub-region must be reachable) and
 * the aspect-ratio orientation at the first level (M-5: never vertical
 * upstream). */
static void test_mt10_division_cuts(void) {
  fbs_maze *m = make(32u, 32u);
  uint32_t seed, single_cell = 0u, vertical = 0u;
  int ok_square = 1, ok_wide = 1, ok_tall = 1;

  for (seed = 0u; seed < 64u; ++seed) {
    fbs_maze_params p = params_of(32u, 32u, seed, FBS_MAZE_DIVISION);
    fbs_maze_rng r;
    uint32_t orient, off, gap, i;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    /* square: the orientation coin comes first, then the offset, then the gap */
    fbs_maze_rng_seed(&r, seed);
    orient = fbs_maze_rng_below(&r, 2u); /* 1 = horizontal */
    off = fbs_maze_rng_below(&r, 31u);
    gap = fbs_maze_rng_below(&r, 32u);
    if (off == 0u) ++single_cell;
    if (!orient) ++vertical;
    for (i = 0u; i < 32u; ++i) {
      int open = orient ? fbs_maze_is_open(m, i, off, FBS_MAZE_S)
                        : fbs_maze_is_open(m, off, i, FBS_MAZE_E);
      if ((i == gap) != (open == 1)) ok_square = 0;
    }
  }
  printf("  MT-10b Division at 32x32 over 64 seeds: %u vertical first cuts, %u one-cell "
         "sub-regions\n",
         vertical, single_cell);
  CHECK(ok_square);
  CHECK(single_cell > 0u); /* M-4: impossible upstream */
  CHECK(vertical > 0u);    /* M-5: impossible upstream, the first cut was always horizontal */

  /* a wide maze always cuts vertically, with no orientation draw at all */
  for (seed = 0u; seed < 64u; ++seed) {
    fbs_maze_params p = params_of(32u, 8u, seed, FBS_MAZE_DIVISION);
    fbs_maze_rng r;
    uint32_t off, gap, i;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    fbs_maze_rng_seed(&r, seed);
    off = fbs_maze_rng_below(&r, 31u);
    gap = fbs_maze_rng_below(&r, 8u);
    for (i = 0u; i < 8u; ++i)
      if ((i == gap) != (fbs_maze_is_open(m, off, i, FBS_MAZE_E) == 1)) ok_wide = 0;
  }
  CHECK(ok_wide);
  /* and a tall maze always cuts horizontally */
  for (seed = 0u; seed < 64u; ++seed) {
    fbs_maze_params p = params_of(8u, 32u, seed, FBS_MAZE_DIVISION);
    fbs_maze_rng r;
    uint32_t off, gap, i;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    fbs_maze_rng_seed(&r, seed);
    off = fbs_maze_rng_below(&r, 31u);
    gap = fbs_maze_rng_below(&r, 8u);
    for (i = 0u; i < 8u; ++i)
      if ((i == gap) != (fbs_maze_is_open(m, i, off, FBS_MAZE_S) == 1)) ok_tall = 0;
  }
  CHECK(ok_tall);
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-11 — serialization round-trip and corruption                           */
/* ------------------------------------------------------------------------- */

static void test_mt11_round_trip(void) {
  fbs_maze *m = make(40u, 40u);
  int a;

  for (a = 0; a < ALGO_COUNT; ++a) {
    fbs_maze_params p = params_of(23u, 17u, 0xC0FFEEu + (uint64_t)a, a);
    size_t need, len = 0u, len2 = 0u;
    unsigned char *blob, *blob2;
    fbs_maze *back = NULL;
    p.braid_percent = 30u;
    p.east_bias_percent = 61u;
    p.newest_percent = 42u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
    CHECK(fbs_maze_open_border(m, 22u, 16u, FBS_MAZE_S) == FBS_MAZE_OK);
    need = fbs_maze_serialized_size(m);
    CHECK(need == 40u + 23u * 17u);
    blob = (unsigned char *)malloc(need);
    blob2 = (unsigned char *)malloc(need);
    CHECK(fbs_maze_serialize(m, blob, need, &len) == FBS_MAZE_OK);
    CHECK(len == need);
    /* the header matches the FBSM v1 schema */
    CHECK(blob[0] == 'F' && blob[1] == 'B' && blob[2] == 'S' && blob[3] == 'M');
    CHECK(blob[4] == 1u && blob[5] == 0u);
    CHECK(blob[6] == 0u && blob[7] == 0u);
    CHECK(blob[24] == (uint8_t)a);
    CHECK(blob[25] == 61u && blob[26] == 42u && blob[27] == 30u);
    CHECK(blob[28] == 1u); /* rng_id */
    CHECK(blob[29] == 0u && blob[30] == 0u && blob[31] == 0u);
    CHECK(fbs_maze_deserialize(blob, len, NULL, NULL, &back) == FBS_MAZE_OK);
    if (back) {
      uint32_t pa = 0u, pb = 0u;
      uint32_t *da, *db;
      uint8_t *ra, *rb;
      size_t rsz = fbs_maze_render_size(m);
      CHECK(fbs_maze_width(back) == 23u && fbs_maze_height(back) == 17u);
      CHECK(memcmp(fbs_maze_cells(back), fbs_maze_cells(m), 23u * 17u) == 0);
      CHECK(params_equal(fbs_maze_params_of(back), &p));
      CHECK(fbs_maze_border_opening_count(back) == 2u);
      /* byte-level fixed point */
      CHECK(fbs_maze_serialize(back, blob2, need, &len2) == FBS_MAZE_OK);
      CHECK(len2 == len);
      CHECK(memcmp(blob, blob2, len) == 0);
      /* it solves and renders identically */
      da = (uint32_t *)malloc(23u * 17u * sizeof(uint32_t));
      db = (uint32_t *)malloc(23u * 17u * sizeof(uint32_t));
      CHECK(fbs_maze_distances(m, 1u, 1u, da, 23u * 17u) == FBS_MAZE_OK);
      CHECK(fbs_maze_distances(back, 1u, 1u, db, 23u * 17u) == FBS_MAZE_OK);
      CHECK(memcmp(da, db, 23u * 17u * sizeof(uint32_t)) == 0);
      CHECK(fbs_maze_longest_path(m, &pa, NULL, NULL, NULL, NULL) == FBS_MAZE_OK);
      CHECK(fbs_maze_longest_path(back, &pb, NULL, NULL, NULL, NULL) == FBS_MAZE_OK);
      CHECK(pa == pb);
      ra = (uint8_t *)malloc(rsz);
      rb = (uint8_t *)malloc(rsz);
      CHECK(fbs_maze_render(m, ra, rsz, NULL, NULL) == FBS_MAZE_OK);
      CHECK(fbs_maze_render_size(back) == rsz);
      CHECK(fbs_maze_render(back, rb, rsz, NULL, NULL) == FBS_MAZE_OK);
      CHECK(memcmp(ra, rb, rsz) == 0);
      free(ra);
      free(rb);
      free(da);
      free(db);
      fbs_maze_destroy(back);
    }
    /* and re-generating from the restored params reproduces the same cells */
    {
      fbs_maze *again = make(23u, 17u);
      CHECK(fbs_maze_generate(again, &p) == FBS_MAZE_OK);
      /* the border openings are not part of generate()'s output */
      CHECK(fbs_maze_border_opening_count(again) == 0u);
      CHECK(fbs_maze_close_border(m, 0u, 0u, FBS_MAZE_N) == FBS_MAZE_OK);
      CHECK(fbs_maze_close_border(m, 22u, 16u, FBS_MAZE_S) == FBS_MAZE_OK);
      CHECK(memcmp(fbs_maze_cells(again), fbs_maze_cells(m), 23u * 17u) == 0);
      fbs_maze_destroy(again);
    }
    free(blob);
    free(blob2);
  }
  fbs_maze_destroy(m);
}

static void test_mt11_corruption(void) {
  fbs_maze *m = make(16u, 16u);
  fbs_maze_params p = params_of(9u, 6u, 5u, FBS_MAZE_ELLER);
  size_t need;
  unsigned char good[40u + 9u * 6u], bad[40u + 9u * 6u];
  size_t len = 0u;
  fbs_maze *out = NULL;

  CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
  need = fbs_maze_serialized_size(m);
  CHECK(need == sizeof good);
  CHECK(fbs_maze_serialize(m, good, need, &len) == FBS_MAZE_OK);
  CHECK(fbs_maze_deserialize(good, need, NULL, NULL, &out) == FBS_MAZE_OK);
  fbs_maze_destroy(out);
  out = NULL;

#define CORRUPT(setup, want)                                    \
  do {                                                          \
    memcpy(bad, good, need);                                    \
    setup;                                                      \
    out = (fbs_maze *)0xA5;                                     \
    CHECK(fbs_maze_deserialize(bad, need, NULL, NULL, &out) == (want)); \
    CHECK(out == (fbs_maze *)0xA5); /* the out pointer is untouched */  \
  } while (0)

  CORRUPT(bad[0] = 'X', FBS_MAZE_E_SCHEMA);
  CORRUPT(bad[3] = 'X', FBS_MAZE_E_SCHEMA);
  CORRUPT(bad[4] = 2u, FBS_MAZE_E_SCHEMA);   /* version */
  CORRUPT(bad[5] = 1u, FBS_MAZE_E_SCHEMA);   /* version high byte */
  CORRUPT(bad[6] = 1u, FBS_MAZE_E_SCHEMA);   /* flags */
  CORRUPT(bad[8] = 10u, FBS_MAZE_E_SCHEMA);  /* width, so len no longer matches */
  CORRUPT(bad[12] = 0u, FBS_MAZE_E_SCHEMA);  /* height zero */
  CORRUPT(bad[24] = 200u, FBS_MAZE_E_SCHEMA); /* algorithm */
  CORRUPT(bad[25] = 101u, FBS_MAZE_E_SCHEMA); /* east bias */
  CORRUPT(bad[26] = 101u, FBS_MAZE_E_SCHEMA); /* newest */
  CORRUPT(bad[27] = 101u, FBS_MAZE_E_SCHEMA); /* braid */
  CORRUPT(bad[28] = 2u, FBS_MAZE_E_SCHEMA);   /* rng_id */
  CORRUPT(bad[29] = 1u, FBS_MAZE_E_SCHEMA);   /* pad */
  CORRUPT(bad[31] = 1u, FBS_MAZE_E_SCHEMA);   /* pad */
  CORRUPT(bad[32] = (unsigned char)(bad[32] + 1u), FBS_MAZE_E_SCHEMA); /* passage_count */
  CORRUPT(bad[36] = 1u, FBS_MAZE_E_SCHEMA);                            /* border_openings */
  CORRUPT(bad[40] = (unsigned char)(bad[40] | 0x10u), FBS_MAZE_E_SCHEMA); /* high nibble */
  CORRUPT(bad[40] = (unsigned char)(bad[40] ^ FBS_MAZE_E), FBS_MAZE_E_SCHEMA); /* unmirrored */
#undef CORRUPT

  /* short and long blobs */
  out = (fbs_maze *)0xA5;
  CHECK(fbs_maze_deserialize(good, 39u, NULL, NULL, &out) == FBS_MAZE_E_SCHEMA);
  CHECK(fbs_maze_deserialize(good, 0u, NULL, NULL, &out) == FBS_MAZE_E_SCHEMA);
  CHECK(fbs_maze_deserialize(good, need - 1u, NULL, NULL, &out) == FBS_MAZE_E_SCHEMA);
  CHECK(out == (fbs_maze *)0xA5);

  /* a blob that does not fit the caller's config is E_FULL, not E_SCHEMA */
  {
    fbs_maze_config small;
    small.max_width = 4u;
    small.max_height = 4u;
    CHECK(fbs_maze_deserialize(good, need, &small, NULL, &out) == FBS_MAZE_E_FULL);
    small.max_width = 9u;
    small.max_height = 5u;
    CHECK(fbs_maze_deserialize(good, need, &small, NULL, &out) == FBS_MAZE_E_FULL);
    CHECK(out == (fbs_maze *)0xA5);
    small.max_width = 0u;
    small.max_height = 8u;
    CHECK(fbs_maze_deserialize(good, need, &small, NULL, &out) == FBS_MAZE_E_RANGE);
    small.max_width = 32u;
    small.max_height = 32u;
    out = NULL;
    CHECK(fbs_maze_deserialize(good, need, &small, NULL, &out) == FBS_MAZE_OK);
    CHECK(out != NULL);
    if (out) {
      CHECK(fbs_maze_memory(out) == fbs_maze_memory_for(&small));
      fbs_maze_destroy(out);
    }
    out = NULL;
  }

  /* serialize one byte short writes the required length and nothing else */
  {
    unsigned char probe[40u + 9u * 6u];
    memset(probe, 0xA5, sizeof probe);
    len = 0u;
    CHECK(fbs_maze_serialize(m, probe, need - 1u, &len) == FBS_MAZE_E_TRUNCATED);
    CHECK(len == need);
    CHECK(untouched(probe, sizeof probe, 0xA5));
    len = 0u;
    CHECK(fbs_maze_serialize(m, NULL, 0u, &len) == FBS_MAZE_E_TRUNCATED);
    CHECK(len == need);
  }
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-12 — capacity, allocation discipline and allocator failure             */
/* ------------------------------------------------------------------------- */

static void test_mt12_allocator(void) {
  counting_alloc ca;
  fbs_maze_allocator alloc;
  fbs_maze_config cfg;
  fbs_maze *m = NULL;
  fbs_maze_params p;
  int a;

  alloc.alloc = ca_alloc;
  alloc.free = ca_free;
  alloc.user = &ca;

  /* the first allocation failing is E_MEMORY and leaks nothing */
  ca.allocs = 0;
  ca.frees = 0;
  ca.bytes = 0u;
  ca.budget = 0;
  cfg.max_width = 33u;
  cfg.max_height = 21u;
  CHECK(fbs_maze_create(&cfg, &alloc, &m) == FBS_MAZE_E_MEMORY);
  CHECK(m == NULL);
  CHECK(ca.allocs == 0 && ca.frees == 0);

  /* exactly one allocation for create, and zero for everything after */
  ca.budget = -1;
  CHECK(fbs_maze_create(&cfg, &alloc, &m) == FBS_MAZE_OK);
  CHECK(ca.allocs == 1);
  CHECK(ca.bytes == fbs_maze_memory_for(&cfg));
  CHECK(fbs_maze_memory(m) == fbs_maze_memory_for(&cfg));
  ca.budget = 0; /* any further allocation now returns NULL and is counted */
  for (a = 0; a < ALGO_COUNT; ++a) {
    uint8_t render[(2u * 33u + 1u) * (2u * 21u + 1u)];
    uint32_t dist[33u * 21u];
    uint32_t path[33u * 21u];
    unsigned char blob[40u + 33u * 21u];
    size_t len = 0u;
    uint32_t ax, ay, bx, by, d;
    p = params_of(33u, 21u, 9u, a);
    p.braid_percent = 20u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_solve(m, 0u, 0u, 32u, 20u, path, sizeof path / sizeof path[0], &len) ==
          FBS_MAZE_OK);
    CHECK(fbs_maze_distances(m, 0u, 0u, dist, sizeof dist / sizeof dist[0]) == FBS_MAZE_OK);
    CHECK(fbs_maze_longest_path(m, &ax, &ay, &bx, &by, &d) == FBS_MAZE_OK);
    CHECK(fbs_maze_render(m, render, sizeof render, NULL, NULL) == FBS_MAZE_OK);
    CHECK(fbs_maze_serialize(m, blob, sizeof blob, &len) == FBS_MAZE_OK);
    CHECK(fbs_maze_is_connected(m) == 1);
    CHECK(fbs_maze_passage_count(m) > 0u);
  }
  CHECK(ca.allocs == 1); /* still one: nothing allocated after create */
  fbs_maze_destroy(m);
  CHECK(ca.frees == 1);

  /* deserialize honours the allocator too, and fails cleanly */
  {
    fbs_maze *src = make(9u, 6u);
    unsigned char blob[40u + 9u * 6u];
    size_t len = 0u;
    fbs_maze *back = NULL;
    p = params_of(9u, 6u, 1u, FBS_MAZE_PRIM);
    CHECK(fbs_maze_generate(src, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_serialize(src, blob, sizeof blob, &len) == FBS_MAZE_OK);
    ca.allocs = 0;
    ca.frees = 0;
    ca.bytes = 0u;
    ca.budget = 0;
    CHECK(fbs_maze_deserialize(blob, len, NULL, &alloc, &back) == FBS_MAZE_E_MEMORY);
    CHECK(back == NULL);
    ca.budget = -1;
    CHECK(fbs_maze_deserialize(blob, len, NULL, &alloc, &back) == FBS_MAZE_OK);
    CHECK(ca.allocs == 1);
    fbs_maze_destroy(back);
    CHECK(ca.frees == 1);
    fbs_maze_destroy(src);
  }

  /* a half-built allocator is rejected before anything is called */
  {
    fbs_maze_allocator broken;
    fbs_maze *x = NULL;
    broken.alloc = NULL;
    broken.free = ca_free;
    broken.user = NULL;
    CHECK(fbs_maze_create(NULL, &broken, &x) == FBS_MAZE_E_INVALID);
    broken.alloc = ca_alloc;
    broken.free = NULL;
    CHECK(fbs_maze_create(NULL, &broken, &x) == FBS_MAZE_E_INVALID);
    CHECK(x == NULL);
  }

  /* memory_for is a pure function of the config and matches what was taken */
  {
    fbs_maze_config c2;
    fbs_maze *y = NULL;
    c2.max_width = 100u;
    c2.max_height = 7u;
    ca.allocs = 0;
    ca.frees = 0;
    ca.bytes = 0u;
    ca.budget = -1;
    CHECK(fbs_maze_create(&c2, &alloc, &y) == FBS_MAZE_OK);
    CHECK(ca.bytes == fbs_maze_memory_for(&c2));
    CHECK(fbs_maze_memory(y) == ca.bytes);
    fbs_maze_destroy(y);
    /* an invalid config has no size */
    c2.max_width = 0u;
    CHECK(fbs_maze_memory_for(&c2) == 0u);
    c2.max_width = 4097u;
    c2.max_height = 1u;
    CHECK(fbs_maze_memory_for(&c2) == 0u);
    c2.max_width = 4096u;
    c2.max_height = 4096u;
    CHECK(fbs_maze_memory_for(&c2) > 0u); /* exactly 1<<24 cells is legal */
    c2.max_width = 4096u;
    c2.max_height = 4095u;
    CHECK(fbs_maze_memory_for(&c2) > 0u);
  }
}

/* ------------------------------------------------------------------------- */
/* MT-13 — no recursion, no globals, reentrancy                              */
/* ------------------------------------------------------------------------- */

static void test_mt13_interleaved(void) {
  fbs_maze *a = make(32u, 32u);
  fbs_maze *b = make(48u, 24u);
  fbs_maze *c = make(32u, 32u);
  uint8_t *ref_a = (uint8_t *)malloc(29u * 23u);
  uint8_t *ref_b = (uint8_t *)malloc(41u * 19u);
  fbs_maze_params pa = params_of(29u, 23u, 0x11112222u, FBS_MAZE_HUNT_AND_KILL);
  fbs_maze_params pb = params_of(41u, 19u, 0x33334444u, FBS_MAZE_DIVISION);
  int i;

  CHECK(fbs_maze_generate(a, &pa) == FBS_MAZE_OK);
  memcpy(ref_a, fbs_maze_cells(a), 29u * 23u);
  CHECK(fbs_maze_generate(b, &pb) == FBS_MAZE_OK);
  memcpy(ref_b, fbs_maze_cells(b), 41u * 19u);

  /* interleave generation, solving and serialization on three live objects */
  for (i = 0; i < 8; ++i) {
    uint32_t path[64];
    size_t len = 0u;
    fbs_maze_params pc = params_of(30u, 30u, (uint64_t)i, FBS_MAZE_GROWING_TREE);
    CHECK(fbs_maze_generate(a, &pa) == FBS_MAZE_OK);
    CHECK(fbs_maze_generate(c, &pc) == FBS_MAZE_OK);
    CHECK(fbs_maze_solve(c, 0u, 0u, 1u, 0u, path, 64u, &len) != FBS_MAZE_E_INVALID);
    CHECK(fbs_maze_generate(b, &pb) == FBS_MAZE_OK);
    CHECK(fbs_maze_is_connected(a) == 1);
    CHECK(memcmp(ref_a, fbs_maze_cells(a), 29u * 23u) == 0);
    CHECK(memcmp(ref_b, fbs_maze_cells(b), 41u * 19u) == 0);
  }
  free(ref_a);
  free(ref_b);
  fbs_maze_destroy(a);
  fbs_maze_destroy(b);
  fbs_maze_destroy(c);
}

/* Stack high-water probe. paint_below() and scan_below() are called from the
 * same frame, so their big local arrays occupy the same stack region; anything
 * the module pushes below our frame lands in it. If the two arrays do not land
 * at the same address (a compiler is free to lay them out differently) the
 * probe reports and skips instead of failing. */
#define PROBE_BYTES (256u * 1024u)

/* The address travels as an integer through a file-scope variable: handing back
 * a pointer to a dead frame is exactly what this probe is about, and no
 * compiler should be asked to like it. Nothing is ever dereferenced through
 * it; it is only ever compared for equality with the next frame's address. */
static volatile int g_probe_mode; /* 0 = paint, 1 = scan */
static volatile uintptr_t g_probe_lo;
static volatile size_t g_probe_used;
static volatile int g_probe_matched;

/* One function for both phases, with ONE code path and ONE loop, so the two
 * calls get the identical frame layout: two functions get different frames
 * under a sanitizer, and an early `return` lets GCC give the paint path and
 * the scan path a stack slot each (measured: the two buffers landed exactly
 * PROBE_BYTES apart at -O2). The scan reads the array back through a pointer
 * laundered via a volatile integer — the bytes were painted by the previous
 * call in that same region, but no compiler can know that, and reading them
 * directly makes every optimizer shout "may be used uninitialized", which is
 * precisely the observation this probe is built on. */
static void probe_frame(void) {
  volatile unsigned char buf[PROBE_BYTES];
  const volatile unsigned char *p;
  size_t i, used = 0u;
  int mode = g_probe_mode;

  g_probe_matched = ((uintptr_t)buf == g_probe_lo);
  p = (const volatile unsigned char *)(uintptr_t)buf;
  for (i = 0u; i < PROBE_BYTES; ++i) {
    if (mode == 0)
      buf[i] = 0xA5u;
    else if (p[PROBE_BYTES - 1u - i] != 0xA5u)
      used = i + 1u;
  }
  if (mode == 0) g_probe_lo = (uintptr_t)buf;
  g_probe_used = used;
}

/* Called through a volatile function pointer so no optimizer can inline the
 * two calls into two separate stack slots in this file's frame (which is what
 * -O3 did before, and which made the two frames incomparable). */
static void (*volatile g_probe_call)(void) = probe_frame;

static void test_mt13_stack_depth(void) {
  uint32_t dim = g_fast ? 256u : 1024u;
  fbs_maze_config cfg;
  fbs_maze *m = NULL;
  int algs[4];
  int i;

  cfg.max_width = dim;
  cfg.max_height = dim;
  CHECK(fbs_maze_create(&cfg, NULL, &m) == FBS_MAZE_OK);
  /* MT-7: the configured maximum must fit the budget memory_for reported */
  CHECK(fbs_maze_memory(m) == fbs_maze_memory_for(&cfg));

  algs[0] = FBS_MAZE_BACKTRACKER; /* recursed once per cell upstream (M-8) */
  algs[1] = FBS_MAZE_KRUSKAL;     /* recursive union-find upstream (M-9)   */
  algs[2] = FBS_MAZE_DIVISION;    /* recursive by name                     */
  algs[3] = FBS_MAZE_GROWING_TREE;

  printf("  MT-13 maximum-size probe at %ux%u, %lu bytes of maze memory\n", dim, dim,
         (unsigned long)fbs_maze_memory(m));
  for (i = 0; i < 4; ++i) {
    fbs_maze_params p = params_of(dim, dim, 1234u, algs[i]);
    size_t used;
    int matched;
    fbs_maze_status st;
    p.braid_percent = 10u;
    g_probe_mode = 0;
    g_probe_used = 0u;
    g_probe_matched = 0;
    g_probe_call();
    st = fbs_maze_generate(m, &p);
    g_probe_mode = 1;
    g_probe_call();
    matched = g_probe_matched;
    used = g_probe_used;
    CHECK(st == FBS_MAZE_OK);
    CHECK(fbs_maze_passage_count(m) >= dim * dim - 1u);
    if (matched) {
      printf("  MT-13 %-13s stack high-water %lu bytes\n", fbs_maze_algorithm_name(algs[i]),
             (unsigned long)used);
      CHECK(used < 64u * 1024u);
      CHECK(used < PROBE_BYTES); /* the probe region was not exhausted */
    } else {
      printf("  MT-13 %-13s stack probe skipped (frames not comparable)\n",
             fbs_maze_algorithm_name(algs[i]));
    }
  }
  /* the maximum size also solves, renders and round-trips inside the budget */
  {
    fbs_maze_params p = params_of(dim, dim, 7u, FBS_MAZE_BACKTRACKER);
    size_t n = (size_t)dim * dim;
    uint32_t *dist = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t *path = (uint32_t *)malloc(n * sizeof(uint32_t));
    unsigned char *blob = (unsigned char *)malloc(40u + n);
    uint8_t *render = (uint8_t *)malloc((2u * (size_t)dim + 1u) * (2u * (size_t)dim + 1u));
    size_t len = 0u;
    fbs_maze *back = NULL;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_distances(m, 0u, 0u, dist, n) == FBS_MAZE_OK);
    CHECK(fbs_maze_solve(m, 0u, 0u, dim - 1u, dim - 1u, path, n, &len) == FBS_MAZE_OK);
    CHECK(len == (size_t)dist[n - 1u] + 1u);
    CHECK(fbs_maze_render(m, render, (2u * (size_t)dim + 1u) * (2u * (size_t)dim + 1u), NULL,
                          NULL) == FBS_MAZE_OK);
    CHECK(fbs_maze_serialize(m, blob, 40u + n, &len) == FBS_MAZE_OK);
    CHECK(fbs_maze_deserialize(blob, len, NULL, NULL, &back) == FBS_MAZE_OK);
    if (back) {
      CHECK(memcmp(fbs_maze_cells(back), fbs_maze_cells(m), n) == 0);
      fbs_maze_destroy(back);
    }
    free(dist);
    free(path);
    free(blob);
    free(render);
  }
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* MT-14 — the render grid                                                   */
/* ------------------------------------------------------------------------- */

static void test_mt14_render(void) {
  fbs_maze *m = make(24u, 24u);
  int a;

  for (a = 0; a < ALGO_COUNT; ++a) {
    uint32_t seed;
    int ok = 1;
    for (seed = 0u; seed < 8u && ok; ++seed) {
      fbs_maze_params p = params_of(13u, 9u, seed, a);
      size_t rw = 27u, rh = 19u, need = rw * rh, i;
      uint8_t *buf = (uint8_t *)malloc(need);
      uint32_t ow = 0u, oh = 0u, floors = 0u, adj = 0u, seen, sp = 0u;
      uint32_t *stack = (uint32_t *)malloc(need * sizeof(uint32_t));
      uint8_t *vis = (uint8_t *)calloc(need, 1u);

      if (fbs_maze_generate(m, &p) != FBS_MAZE_OK) ok = 0;
      if (fbs_maze_render_size(m) != need) ok = 0;
      if (fbs_maze_render(m, buf, need, &ow, &oh) != FBS_MAZE_OK) ok = 0;
      if (ow != (uint32_t)rw || oh != (uint32_t)rh) ok = 0;
      /* the ring is solid: generate() leaves no border opening */
      for (i = 0u; i < rw; ++i)
        if (buf[i] || buf[(rh - 1u) * rw + i]) ok = 0;
      for (i = 0u; i < rh; ++i)
        if (buf[i * rw] || buf[i * rw + rw - 1u]) ok = 0;
      /* floors and 4-adjacencies */
      for (i = 0u; i < need; ++i) {
        size_t x = i % rw, y = i / rw;
        if (!buf[i]) continue;
        ++floors;
        if (x + 1u < rw && buf[i + 1u]) ++adj;
        if (y + 1u < rh && buf[i + rw]) ++adj;
      }
      /* a perfect, unbraided maze renders w*h cell floors plus w*h-1 passage
         floors, in one component, with exactly floors-1 adjacencies */
      if (floors != 2u * 13u * 9u - 1u) ok = 0;
      if (adj != floors - 1u) ok = 0;
      /* one component, by flood fill */
      seen = 0u;
      for (i = 0u; i < need; ++i)
        if (buf[i]) {
          stack[sp++] = (uint32_t)i;
          vis[i] = 1u;
          seen = 1u;
          break;
        }
      while (sp > 0u) {
        uint32_t c = stack[--sp];
        size_t x = c % rw, y = c / rw;
        if (x > 0u && buf[c - 1u] && !vis[c - 1u]) {
          vis[c - 1u] = 1u;
          stack[sp++] = c - 1u;
          ++seen;
        }
        if (x + 1u < rw && buf[c + 1u] && !vis[c + 1u]) {
          vis[c + 1u] = 1u;
          stack[sp++] = c + 1u;
          ++seen;
        }
        if (y > 0u && buf[c - rw] && !vis[c - rw]) {
          vis[c - rw] = 1u;
          stack[sp++] = (uint32_t)(c - rw);
          ++seen;
        }
        if (y + 1u < rh && buf[c + rw] && !vis[c + rw]) {
          vis[c + rw] = 1u;
          stack[sp++] = (uint32_t)(c + rw);
          ++seen;
        }
      }
      if (seen != floors) ok = 0;
      if (!ok) printf("  MT-14 %s seed %u\n", fbs_maze_algorithm_name(a), seed);
      free(buf);
      free(stack);
      free(vis);
    }
    CHECK(ok);
  }

  /* every size renders exactly, odd and even alike (M-18) */
  {
    uint32_t w, h;
    for (w = 1u; w <= 12u; ++w)
      for (h = 1u; h <= 12u; ++h) {
        fbs_maze_params p = params_of(w, h, w * 31u + h, FBS_MAZE_SIDEWINDER);
        size_t need = (2u * (size_t)w + 1u) * (2u * (size_t)h + 1u);
        uint8_t *buf = (uint8_t *)malloc(need);
        uint32_t ow = 0u, oh = 0u;
        CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
        CHECK(fbs_maze_render_size(m) == need);
        CHECK(fbs_maze_render(m, buf, need, &ow, &oh) == FBS_MAZE_OK);
        CHECK(ow == 2u * w + 1u && oh == 2u * h + 1u);
        free(buf);
      }
  }

  /* one byte short is E_TRUNCATED, with the size reported and nothing written */
  {
    fbs_maze_params p = params_of(6u, 4u, 1u, FBS_MAZE_ELLER);
    size_t need = 13u * 9u;
    uint8_t buf[13u * 9u];
    uint32_t ow = 0u, oh = 0u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    memset(buf, 0xA5, sizeof buf);
    CHECK(fbs_maze_render(m, buf, need - 1u, &ow, &oh) == FBS_MAZE_E_TRUNCATED);
    CHECK(ow == 13u && oh == 9u);
    CHECK(untouched(buf, sizeof buf, 0xA5));
    ow = 0u;
    oh = 0u;
    CHECK(fbs_maze_render(m, NULL, 0u, &ow, &oh) == FBS_MAZE_E_TRUNCATED);
    CHECK((size_t)ow * oh == need);
    CHECK(fbs_maze_render(m, buf, need, NULL, NULL) == FBS_MAZE_OK);
  }
  fbs_maze_destroy(m);
}

/* ------------------------------------------------------------------------- */
/* Names, version, defaults                                                  */
/* ------------------------------------------------------------------------- */

static void test_names_and_defaults(void) {
  fbs_maze_config cfg = fbs_maze_config_default();
  fbs_maze_params p = fbs_maze_params_default();
  int i;

  CHECK(fbs_maze_version() == FBS_MAZE_VERSION);
  CHECK(fbs_maze_version() == 100u);
  CHECK(cfg.max_width == 257u && cfg.max_height == 257u);
  CHECK(p.width == 16u && p.height == 16u && p.seed == 0u);
  CHECK(p.algorithm == (uint8_t)FBS_MAZE_BACKTRACKER);
  CHECK(p.east_bias_percent == 50u && p.newest_percent == 50u && p.braid_percent == 0u);

  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_OK), "ok") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_INVALID), "invalid") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_RANGE), "range") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_FULL), "full") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_UNREACHABLE), "unreachable") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_SCHEMA), "schema") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_TRUNCATED), "truncated") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_STATE), "state") == 0);
  CHECK(strcmp(fbs_maze_status_name(FBS_MAZE_E_MEMORY), "memory") == 0);
  CHECK(strcmp(fbs_maze_status_name(1), "unknown") == 0);
  CHECK(strcmp(fbs_maze_status_name(-99), "unknown") == 0);

  for (i = 0; i < ALGO_COUNT; ++i) {
    CHECK(strcmp(fbs_maze_algorithm_name(i), "unknown") != 0);
    CHECK(fbs_maze_algorithm_name(i)[0] != 0);
  }
  CHECK(strcmp(fbs_maze_algorithm_name(0), "backtracker") == 0);
  CHECK(strcmp(fbs_maze_algorithm_name(7), "growing_tree") == 0);
  CHECK(strcmp(fbs_maze_algorithm_name(ALGO_COUNT), "unknown") == 0);
  CHECK(strcmp(fbs_maze_algorithm_name(-1), "unknown") == 0);

  /* the direction encoding the whole module rests on */
  CHECK(FBS_MAZE_OPPOSITE(FBS_MAZE_N) == FBS_MAZE_S);
  CHECK(FBS_MAZE_OPPOSITE(FBS_MAZE_E) == FBS_MAZE_W);
  CHECK(FBS_MAZE_OPPOSITE(FBS_MAZE_S) == FBS_MAZE_N);
  CHECK(FBS_MAZE_OPPOSITE(FBS_MAZE_W) == FBS_MAZE_E);
  CHECK((FBS_MAZE_N | FBS_MAZE_E | FBS_MAZE_S | FBS_MAZE_W) == FBS_MAZE_DIRS);

  /* the default config really is usable with the default params */
  {
    fbs_maze *m = NULL;
    CHECK(fbs_maze_create(NULL, NULL, &m) == FBS_MAZE_OK);
    CHECK(fbs_maze_memory(m) == fbs_maze_memory_for(NULL));
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_width(m) == 16u && fbs_maze_height(m) == 16u);
    CHECK(fbs_maze_is_perfect(m) == 1);
    CHECK(params_equal(fbs_maze_params_of(m), &p));
    /* the config maximum is 257x257 by default */
    p.width = 257u;
    p.height = 257u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_OK);
    CHECK(fbs_maze_is_perfect(m) == 1);
    p.width = 258u;
    CHECK(fbs_maze_generate(m, &p) == FBS_MAZE_E_FULL);
    fbs_maze_destroy(m);
  }
}

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
  int i;
  const char *fast = getenv("FBS_MAZE_TEST_FAST");

  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--write-fixtures") == 0) {
      g_write_fixtures = 1;
    } else if (strcmp(argv[i], "--fixture-dir") == 0 && i + 1 < argc) {
      g_fixture_dir = argv[++i];
    } else {
      printf("usage: %s [--write-fixtures] [--fixture-dir DIR]\n", argv[0]);
      return 2;
    }
  }
  if (fast && fast[0] && fast[0] != '0') {
    g_fast = 1;
    printf("maze: FBS_MAZE_TEST_FAST is set; statistical loops and the maximum-size probe "
           "are scaled down\n");
  }

  test_mt1_connectivity();
  test_mt2_perfect();
  test_mt3_symmetry();
  test_mt4_reproducibility();
  test_mt5_sidewinder();
  test_mt5_eller_rows();
  test_mt5_growing_tree_identity();
  test_mt5_dead_end_texture();
  test_mt5_kruskal_hak_identity();
  test_mt5_border_render();
  test_mt6_braiding();
  test_mt7_degenerate();
  test_mt7_errors();
  test_mt8_solver();
  test_mt8_generated();
  test_mt9_shuffle_fairness();
  test_mt9_bounded_draw();
  test_mt9_rejection_branch();
  test_mt9_kruskal_cross_check();
  test_mt10_eller_runs();
  test_mt10_division_cuts();
  test_mt10_division_structure();
  test_mt11_round_trip();
  test_mt11_corruption();
  test_mt12_allocator();
  test_mt13_interleaved();
  test_mt13_stack_depth();
  test_mt14_render();
  test_names_and_defaults();

  printf("maze: %d checks passed, %d failed\n", g_checks - g_fails, g_fails);
  return g_fails > 100 ? 100 : g_fails;
}
