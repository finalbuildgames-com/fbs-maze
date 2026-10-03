/*
 * src/maze.c — FinalBuildSystems seeded maze generator. Implements
 * include/fbs/maze.h.
 *
 * Original work. Nothing is ported, wrapped or vendored: the owned Fab plugin
 * ("Maze Generator", LowkeyMe, MIT upstream at
 * fa3f913c64309f350e3f174a2f34e2ac03b0c27c, notice vendored at
 * third_party/lowkeyme/) is a *specification input only*; this file owes it
 * the framing (direction-bit grid, one seeded stream), not code. The
 * algorithms are textbook prior art (Jamis Buck's 2010-2011 series, linked by
 * the upstream README itself, and Wikipedia's "Maze generation algorithm");
 * the PRNG is public domain (Blackman & Vigna's xoshiro128** 1.1, Vigna's
 * splitmix64). The plugin's defects, labelled M-1 .. M-26 in the comments
 * below, are fixed structurally here, not copied.
 *
 * C99. Standard library only (<stdlib.h> for the default allocator,
 * <string.h> for memset/memcpy). No libm, no floating point anywhere, no
 * globals, no static mutable state, no recursion. One allocation per maze.
 *
 * ---------------------------------------------------------------------------
 * Internal contracts the public header only implies
 * ---------------------------------------------------------------------------
 *
 * PRNG. fbs_maze_rng_seed expands the 64-bit seed with splitmix64 (Vigna 2015)
 * into the four 32-bit words of the xoshiro128** state, in this exact order:
 *
 *     w0 = splitmix64(state);  s[0] = low32(w0);  s[1] = high32(w0);
 *     w1 = splitmix64(state);  s[2] = low32(w1);  s[3] = high32(w1);
 *
 * where `state` starts at the seed and splitmix64 pre-increments it by the
 * golden-ratio constant, as in the reference implementation. splitmix64 is a
 * bijection on 64-bit states, so two *consecutive* outputs can never both be
 * zero; the xoshiro state is therefore never all-zero, for any seed including
 * 0. That is the whole reason splitmix64 is here, and it is why no "if the
 * state is zero, fix it" branch exists (it would be unreachable code).
 *
 * fbs_maze_rng_next is xoshiro128** **1.1**: the scrambler is
 * `rotl(s[1] * 5, 7) * 9`. Version 1.0 scrambled s[0] and produces a different
 * stream; the version is pinned in the header, here, and by the golden vectors
 * in tests/fixtures/maze/rng-vectors.txt.
 *
 * fbs_maze_rng_below uses the PCG/Lemire threshold rejection loop, verbatim:
 *
 *     threshold = (0u - bound) % bound;    // == 2^32 mod bound
 *     do { x = next(); } while (x < threshold);
 *     return x % bound;
 *
 * i.e. the *low* 2^32 mod bound outputs are rejected. This is unbiased, has no
 * floating point, and terminates with probability 1. `bound == 0` returns 0 and
 * consumes nothing (documented in the header; mazelib's `x % 0` SIGFPEs here).
 * `bound == 1` has threshold 0, so it accepts immediately and *does* consume
 * one draw — only 0 short-circuits.
 *
 * DIRECTIONS. N=1, E=2, S=4, W=8, so FBS_MAZE_OPPOSITE is a 2-bit rotation.
 * Every scan that enumerates directions does so in N, E, S, W order — the
 * generators, the braider, the BFS and the render step — because the order is
 * observable in the output and therefore part of the contract.
 *
 * CELL BYTES. Low nibble only. The high nibble is reserved 0 and is never
 * written; algorithm state lives in a separate scratch array, which is the
 * structural fix for M-12. A border cell may carry an outward bit whose
 * neighbour does not exist (an entrance/exit): that is the only unmirrored
 * edge, and it is exactly the case that would index negatively upstream (M-7).
 *
 * ONE ALLOCATION. fbs_maze_create makes exactly one call to the allocator. The
 * block holds, in order: the fbs_maze header, the cells, then the algorithm
 * scratch (one byte-per-cell flag array, two uint32-per-cell arrays, a
 * 2*cells uint32 edge array for Kruskal, five max_width uint32 row arrays for
 * Eller and a bounded division stack), then the solver scratch (two
 * uint32-per-cell arrays: the BFS queue and one distance/parent field).
 * generate(), solve(), distances(), longest_path(), render() and serialize()
 * allocate nothing. Everything is sized from the config alone, so
 * fbs_maze_memory_for(cfg) can report the exact cost before allocating (M-22).
 *
 * SCRATCH THROUGH A const POINTER. is_connected(), solve(), distances() and
 * longest_path() take `const fbs_maze *`. They write through the *pointer
 * members* of the struct, which const-qualifies the pointers, not the arrays
 * they point at — legal C, no cast, and the logical value of the maze is
 * unchanged.
 *
 * ALGORITHMS. All eight are pure functions of (params) through the one seeded
 * stream. Draw counts and draw order are part of the output and are documented
 * at each routine.
 *
 *   BACKTRACKER, PRIM and GROWING_TREE are ONE routine, `maze_growing_tree`,
 *   called with newest_percent 100, 0 and params.newest_percent respectively.
 *   That is what makes the header's "newest_percent 100 == Backtracker, 0 ==
 *   Prim" a structural identity instead of a coincidence to be maintained by
 *   hand (witness MT-5), and it is mazelib's growing-tree framing: always
 *   choosing the newest active cell *is* the recursive backtracker; choosing
 *   uniformly at random *is* the randomised Prim texture ("many short dead
 *   ends"). The selection coin is drawn on every iteration in all three cases,
 *   so the three share one draw sequence exactly.
 *
 * SCHEMA. Magic "FBSM", version 1, rng_id 1,
 * 40 + width*height bytes, every field little-endian and written one at a
 * time. deserialize validates magic, version, flags, pad, sizes against both
 * `len` and the config, the algorithm enum, the three percents, rng_id, every
 * cell's high nibble, edge mirroring, and the two redundant counts — and
 * allocates nothing until all of that has passed.
 */

#include "fbs/maze.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Limits                                                                    */
/* ------------------------------------------------------------------------- */

#define MAZE_SCHEMA_VERSION 1u
#define MAZE_RNG_ID 1u /* xoshiro128** 1.1 seeded by splitmix64 */
#define MAZE_HEADER_BYTES 40u

#define MAZE_MAX_DIM 4096u
#define MAZE_MAX_CELLS (1u << 24)

#define MAZE_BLOCK_ALIGN 8u

static size_t maze_align_up(size_t v) {
  return (v + (MAZE_BLOCK_ALIGN - 1u)) & ~(size_t)(MAZE_BLOCK_ALIGN - 1u);
}

static void *maze_default_alloc(void *user, size_t bytes) {
  (void)user;
  return malloc(bytes);
}

static void maze_default_free(void *user, void *ptr) {
  (void)user;
  free(ptr);
}

/* ------------------------------------------------------------------------- */
/* PRNG: public ABI, because the maze bytes are defined by it                */
/* ------------------------------------------------------------------------- */

static uint32_t maze_rotl(uint32_t x, unsigned k) {
  return (uint32_t)((x << k) | (x >> (32u - k)));
}

/* splitmix64, Vigna 2015, public domain. Used for seeding only. */
static uint64_t maze_splitmix64(uint64_t *state) {
  uint64_t z;
  *state += 0x9E3779B97F4A7C15ULL;
  z = *state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void fbs_maze_rng_seed(fbs_maze_rng *r, uint64_t seed) {
  uint64_t state = seed;
  uint64_t w;
  if (!r) return;
  w = maze_splitmix64(&state);
  r->s[0] = (uint32_t)(w & 0xFFFFFFFFu);
  r->s[1] = (uint32_t)((w >> 32) & 0xFFFFFFFFu);
  w = maze_splitmix64(&state);
  r->s[2] = (uint32_t)(w & 0xFFFFFFFFu);
  r->s[3] = (uint32_t)((w >> 32) & 0xFFFFFFFFu);
}

uint32_t fbs_maze_rng_next(fbs_maze_rng *r) {
  uint32_t result, t;
  if (!r) return 0u;
  result = maze_rotl((uint32_t)(r->s[1] * 5u), 7u) * 9u;
  t = (uint32_t)(r->s[1] << 9);
  r->s[2] ^= r->s[0];
  r->s[3] ^= r->s[1];
  r->s[1] ^= r->s[2];
  r->s[0] ^= r->s[3];
  r->s[2] ^= t;
  r->s[3] = maze_rotl(r->s[3], 11u);
  return result;
}

uint32_t fbs_maze_rng_below(fbs_maze_rng *r, uint32_t bound) {
  uint32_t threshold, x;
  if (!r || bound == 0u) return 0u;
  threshold = (uint32_t)(0u - bound) % bound; /* == 2^32 mod bound */
  do {
    x = fbs_maze_rng_next(r);
  } while (x < threshold);
  return x % bound;
}

/* ------------------------------------------------------------------------- */
/* Internal representation                                                   */
/* ------------------------------------------------------------------------- */

struct fbs_maze {
  fbs_maze_allocator alloc;
  size_t block_size;

  uint32_t max_width;
  uint32_t max_height;
  uint32_t max_cells; /* max_width * max_height */
  uint32_t row_cap;   /* max_width */
  uint32_t div_cap;   /* division stack entries, 4 uint32 each */

  uint32_t width;
  uint32_t height;
  int generated;
  fbs_maze_params params;
  fbs_maze_rng rng;

  uint8_t *cells; /* max_cells bytes, the maze                                */

  /* algorithm scratch */
  uint8_t *flags;   /* max_cells bytes: per-cell visited state                */
  uint32_t *a32;    /* max_cells: active list / union-find parent             */
  uint32_t *b32;    /* max_cells: union-find size                             */
  uint32_t *edges;  /* 2 * max_cells: Kruskal's edge list                     */
  uint32_t *rparent; /* row_cap: Eller row union-find parent                  */
  uint32_t *rsize;   /* row_cap: Eller row union-find size                    */
  uint32_t *rhead;   /* row_cap: first column of each row set                 */
  uint32_t *rtail;   /* row_cap: last column of each row set                  */
  uint32_t *rnext;   /* row_cap: next column in the set, ascending            */
  uint32_t *dstack;  /* 4 * div_cap: recursive division regions (x,y,w,h)     */

  /* solver scratch */
  uint32_t *queue; /* max_cells: BFS queue                                    */
  uint32_t *aux;   /* max_cells: BFS distance or parent field                 */
};

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

static const uint8_t maze_dir_order[4] = {FBS_MAZE_N, FBS_MAZE_E, FBS_MAZE_S, FBS_MAZE_W};

static int maze_dir_index(int dir) {
  switch (dir) {
    case (int)FBS_MAZE_N: return 0;
    case (int)FBS_MAZE_E: return 1;
    case (int)FBS_MAZE_S: return 2;
    case (int)FBS_MAZE_W: return 3;
    default: return -1;
  }
}

/* 1 when the neighbour of (x,y) in direction `dir` is inside the grid, and
 * then *out is its cell index. Directions are N, E, S, W. */
static int maze_step(const fbs_maze *m, uint32_t x, uint32_t y, uint8_t dir, uint32_t *out) {
  switch (dir) {
    case FBS_MAZE_N:
      if (y == 0u) return 0;
      *out = (y - 1u) * m->width + x;
      return 1;
    case FBS_MAZE_E:
      if (x + 1u >= m->width) return 0;
      *out = y * m->width + x + 1u;
      return 1;
    case FBS_MAZE_S:
      if (y + 1u >= m->height) return 0;
      *out = (y + 1u) * m->width + x;
      return 1;
    default: /* FBS_MAZE_W */
      if (x == 0u) return 0;
      *out = y * m->width + x - 1u;
      return 1;
  }
}

static void maze_carve(fbs_maze *m, uint32_t cell, uint8_t dir, uint32_t neighbour) {
  m->cells[cell] = (uint8_t)(m->cells[cell] | dir);
  m->cells[neighbour] = (uint8_t)(m->cells[neighbour] | FBS_MAZE_OPPOSITE(dir));
}

static void maze_close(fbs_maze *m, uint32_t cell, uint8_t dir, uint32_t neighbour) {
  m->cells[cell] = (uint8_t)(m->cells[cell] & (uint8_t)~(unsigned)dir);
  m->cells[neighbour] = (uint8_t)(m->cells[neighbour] & (uint8_t)~(unsigned)FBS_MAZE_OPPOSITE(dir));
}

static unsigned maze_popcount4(uint8_t c) {
  return (unsigned)((c & 1u) + ((c >> 1) & 1u) + ((c >> 2) & 1u) + ((c >> 3) & 1u));
}

/* Fisher-Yates, the forward form (the M-3 fix for the plugin's biased naive
 * shuffle): swap a[i] with a[i + rng_below(n - i)] for i = 0 .. n-2. n-1 is
 * skipped because
 * rng_below(1) is a no-op that would still burn a draw. */
static void maze_shuffle(fbs_maze_rng *r, uint32_t *a, uint32_t n) {
  uint32_t i;
  for (i = 0u; i + 1u < n; ++i) {
    uint32_t j = i + fbs_maze_rng_below(r, n - i);
    uint32_t t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

/* ------------------------------------------------------------------------- */
/* Names, version, defaults                                                  */
/* ------------------------------------------------------------------------- */

const char *fbs_maze_status_name(int status) {
  switch (status) {
    case FBS_MAZE_OK: return "ok";
    case FBS_MAZE_E_INVALID: return "invalid";
    case FBS_MAZE_E_RANGE: return "range";
    case FBS_MAZE_E_FULL: return "full";
    case FBS_MAZE_E_UNREACHABLE: return "unreachable";
    case FBS_MAZE_E_SCHEMA: return "schema";
    case FBS_MAZE_E_TRUNCATED: return "truncated";
    case FBS_MAZE_E_STATE: return "state";
    case FBS_MAZE_E_MEMORY: return "memory";
    default: return "unknown";
  }
}

unsigned fbs_maze_version(void) { return FBS_MAZE_VERSION; }

const char *fbs_maze_algorithm_name(int algorithm) {
  switch (algorithm) {
    case FBS_MAZE_BACKTRACKER: return "backtracker";
    case FBS_MAZE_KRUSKAL: return "kruskal";
    case FBS_MAZE_PRIM: return "prim";
    case FBS_MAZE_ELLER: return "eller";
    case FBS_MAZE_SIDEWINDER: return "sidewinder";
    case FBS_MAZE_HUNT_AND_KILL: return "hunt_and_kill";
    case FBS_MAZE_DIVISION: return "division";
    case FBS_MAZE_GROWING_TREE: return "growing_tree";
    default: return "unknown";
  }
}

fbs_maze_config fbs_maze_config_default(void) {
  fbs_maze_config cfg;
  cfg.max_width = 257u;
  cfg.max_height = 257u;
  return cfg;
}

fbs_maze_params fbs_maze_params_default(void) {
  fbs_maze_params p;
  p.width = 16u;
  p.height = 16u;
  p.seed = 0u;
  p.algorithm = (uint8_t)FBS_MAZE_BACKTRACKER;
  p.east_bias_percent = 50u;
  p.newest_percent = 50u;
  p.braid_percent = 0u;
  return p;
}

/* ------------------------------------------------------------------------- */
/* Layout and lifetime                                                       */
/* ------------------------------------------------------------------------- */

static int maze_config_valid(const fbs_maze_config *c) {
  if (c->max_width == 0u || c->max_height == 0u) return 0;
  if (c->max_width > MAZE_MAX_DIM || c->max_height > MAZE_MAX_DIM) return 0;
  /* max_width * max_height <= 1 << 24, computed in 64 bits so it cannot wrap */
  if ((uint64_t)c->max_width * (uint64_t)c->max_height > (uint64_t)MAZE_MAX_CELLS) return 0;
  return 1;
}

enum {
  MAZE_OFF_CELLS = 0,
  MAZE_OFF_FLAGS,
  MAZE_OFF_A32,
  MAZE_OFF_B32,
  MAZE_OFF_EDGES,
  MAZE_OFF_RPARENT,
  MAZE_OFF_RSIZE,
  MAZE_OFF_RHEAD,
  MAZE_OFF_RTAIL,
  MAZE_OFF_RNEXT,
  MAZE_OFF_DSTACK,
  MAZE_OFF_QUEUE,
  MAZE_OFF_AUX,
  MAZE_OFF_COUNT
};

/* The division stack never holds more than (w + h - 1) regions: pushing both
 * halves and popping one is a depth-first walk of a binary tree, whose stack
 * is bounded by (depth + 1), and every split strictly decreases w + h of the
 * region it splits (a split of extent L yields extents a and b with a + b = L,
 * a >= 1, b >= 1). max_width + max_height + 2 is therefore generous. */
static uint32_t maze_div_cap(const fbs_maze_config *c) { return c->max_width + c->max_height + 2u; }

static size_t maze_block_layout(const fbs_maze_config *c, size_t off[MAZE_OFF_COUNT]) {
  size_t cells = (size_t)c->max_width * (size_t)c->max_height;
  size_t rows = (size_t)c->max_width;
  size_t divs = (size_t)maze_div_cap(c);
  size_t at = maze_align_up(sizeof(struct fbs_maze));

  off[MAZE_OFF_CELLS] = at;
  at = maze_align_up(at + cells);
  off[MAZE_OFF_FLAGS] = at;
  at = maze_align_up(at + cells);
  off[MAZE_OFF_A32] = at;
  at = maze_align_up(at + cells * sizeof(uint32_t));
  off[MAZE_OFF_B32] = at;
  at = maze_align_up(at + cells * sizeof(uint32_t));
  off[MAZE_OFF_EDGES] = at;
  at = maze_align_up(at + 2u * cells * sizeof(uint32_t));
  off[MAZE_OFF_RPARENT] = at;
  at = maze_align_up(at + rows * sizeof(uint32_t));
  off[MAZE_OFF_RSIZE] = at;
  at = maze_align_up(at + rows * sizeof(uint32_t));
  off[MAZE_OFF_RHEAD] = at;
  at = maze_align_up(at + rows * sizeof(uint32_t));
  off[MAZE_OFF_RTAIL] = at;
  at = maze_align_up(at + rows * sizeof(uint32_t));
  off[MAZE_OFF_RNEXT] = at;
  at = maze_align_up(at + rows * sizeof(uint32_t));
  off[MAZE_OFF_DSTACK] = at;
  at = maze_align_up(at + 4u * divs * sizeof(uint32_t));
  off[MAZE_OFF_QUEUE] = at;
  at = maze_align_up(at + cells * sizeof(uint32_t));
  off[MAZE_OFF_AUX] = at;
  at = maze_align_up(at + cells * sizeof(uint32_t));
  return at;
}

size_t fbs_maze_memory_for(const fbs_maze_config *cfg) {
  fbs_maze_config c = cfg ? *cfg : fbs_maze_config_default();
  size_t off[MAZE_OFF_COUNT];
  if (!maze_config_valid(&c)) return 0u;
  return maze_block_layout(&c, off);
}

size_t fbs_maze_memory(const fbs_maze *m) { return m ? m->block_size : (size_t)0; }

fbs_maze_status fbs_maze_create(const fbs_maze_config *cfg, const fbs_maze_allocator *alloc,
                                fbs_maze **out) {
  fbs_maze_config c;
  fbs_maze_allocator a;
  size_t off[MAZE_OFF_COUNT];
  size_t total;
  unsigned char *block;
  fbs_maze *m;

  if (!out) return FBS_MAZE_E_INVALID;
  c = cfg ? *cfg : fbs_maze_config_default();
  if (!maze_config_valid(&c)) return FBS_MAZE_E_RANGE;
  if (alloc) {
    if (!alloc->alloc || !alloc->free) return FBS_MAZE_E_INVALID;
    a = *alloc;
  } else {
    a.alloc = maze_default_alloc;
    a.free = maze_default_free;
    a.user = NULL;
  }

  total = maze_block_layout(&c, off);
  block = (unsigned char *)a.alloc(a.user, total);
  if (!block) return FBS_MAZE_E_MEMORY;
  memset(block, 0, total);

  m = (fbs_maze *)(void *)block;
  m->alloc = a;
  m->block_size = total;
  m->max_width = c.max_width;
  m->max_height = c.max_height;
  m->max_cells = (uint32_t)((uint64_t)c.max_width * (uint64_t)c.max_height);
  m->row_cap = c.max_width;
  m->div_cap = maze_div_cap(&c);
  m->width = 0u;
  m->height = 0u;
  m->generated = 0;
  m->params = fbs_maze_params_default();

  m->cells = block + off[MAZE_OFF_CELLS];
  m->flags = block + off[MAZE_OFF_FLAGS];
  m->a32 = (uint32_t *)(void *)(block + off[MAZE_OFF_A32]);
  m->b32 = (uint32_t *)(void *)(block + off[MAZE_OFF_B32]);
  m->edges = (uint32_t *)(void *)(block + off[MAZE_OFF_EDGES]);
  m->rparent = (uint32_t *)(void *)(block + off[MAZE_OFF_RPARENT]);
  m->rsize = (uint32_t *)(void *)(block + off[MAZE_OFF_RSIZE]);
  m->rhead = (uint32_t *)(void *)(block + off[MAZE_OFF_RHEAD]);
  m->rtail = (uint32_t *)(void *)(block + off[MAZE_OFF_RTAIL]);
  m->rnext = (uint32_t *)(void *)(block + off[MAZE_OFF_RNEXT]);
  m->dstack = (uint32_t *)(void *)(block + off[MAZE_OFF_DSTACK]);
  m->queue = (uint32_t *)(void *)(block + off[MAZE_OFF_QUEUE]);
  m->aux = (uint32_t *)(void *)(block + off[MAZE_OFF_AUX]);

  *out = m;
  return FBS_MAZE_OK;
}

void fbs_maze_destroy(fbs_maze *m) {
  fbs_maze_allocator a;
  if (!m) return;
  a = m->alloc;
  a.free(a.user, m);
}

/* ------------------------------------------------------------------------- */
/* Generators                                                                */
/* ------------------------------------------------------------------------- */

/* Growing tree — BACKTRACKER (newest 100), PRIM (newest 0) and GROWING_TREE
 * (newest = params.newest_percent) are this one routine, following mazelib's
 * growing-tree framing. Draw order per iteration: the selection coin
 * rng_below(100), then rng_below(len) when the coin picked "random", then
 * rng_below(candidates) when the chosen cell still has an unvisited
 * neighbour. The coin is drawn in every case, including the two extremes, so
 * all three algorithms consume one identical stream. The start cell is one
 * rng_below(width*height) draw. Removal from the active list is
 * swap-with-last, which for "newest" is a plain pop, which is exactly the
 * explicit-stack recursive backtracker (M-8: no recursion, and the list is
 * sized from the config). */
static void maze_growing_tree(fbs_maze *m, uint32_t newest_percent) {
  uint32_t w = m->width, h = m->height;
  uint32_t n = w * h;
  uint32_t *list = m->a32;
  uint32_t len = 0u;
  uint32_t start;

  memset(m->flags, 0, (size_t)n);
  start = fbs_maze_rng_below(&m->rng, n);
  m->flags[start] = 1u;
  list[len++] = start;

  while (len > 0u) {
    uint32_t idx;
    uint32_t c, x, y, k;
    uint32_t nbs[4];
    uint8_t dirs[4];
    unsigned cnt = 0u, i;

    idx = (fbs_maze_rng_below(&m->rng, 100u) < newest_percent)
              ? (len - 1u)
              : fbs_maze_rng_below(&m->rng, len);
    c = list[idx];
    x = c % w;
    y = c / w;
    for (i = 0u; i < 4u; ++i) {
      uint32_t nb;
      if (maze_step(m, x, y, maze_dir_order[i], &nb) && !m->flags[nb]) {
        dirs[cnt] = maze_dir_order[i];
        nbs[cnt] = nb;
        ++cnt;
      }
    }
    if (cnt == 0u) {
      list[idx] = list[len - 1u];
      --len;
      continue;
    }
    k = fbs_maze_rng_below(&m->rng, cnt);
    maze_carve(m, c, dirs[k], nbs[k]);
    m->flags[nbs[k]] = 1u;
    list[len++] = nbs[k];
  }
}

/* Kruskal — every interior edge, shuffled by Fisher-Yates (M-3), then a
 * union-find with union by size and iterative find with path halving (M-9).
 * The edge array is exactly (w-1)h + w(h-1) entries inside the single
 * allocation (M-10). Edge encoding: cell index << 1, low bit 0 = east,
 * 1 = south. Edges are generated in row-major order, east before south. */
static uint32_t maze_uf_find(uint32_t *parent, uint32_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]]; /* path halving */
    x = parent[x];
  }
  return x;
}

static void maze_kruskal(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t n = w * h;
  uint32_t *parent = m->a32, *size = m->b32;
  uint32_t count = 0u, i, x, y;

  for (i = 0u; i < n; ++i) {
    parent[i] = i;
    size[i] = 1u;
  }
  for (y = 0u; y < h; ++y) {
    for (x = 0u; x < w; ++x) {
      uint32_t c = y * w + x;
      if (x + 1u < w) m->edges[count++] = (uint32_t)(c << 1);
      if (y + 1u < h) m->edges[count++] = (uint32_t)((c << 1) | 1u);
    }
  }
  maze_shuffle(&m->rng, m->edges, count);

  for (i = 0u; i < count; ++i) {
    uint32_t e = m->edges[i];
    uint32_t c = e >> 1;
    uint8_t dir = (e & 1u) ? (uint8_t)FBS_MAZE_S : (uint8_t)FBS_MAZE_E;
    uint32_t nb = (e & 1u) ? (c + w) : (c + 1u);
    uint32_t ra = maze_uf_find(parent, c), rb = maze_uf_find(parent, nb);
    if (ra == rb) continue;
    if (size[ra] < size[rb]) {
      uint32_t t = ra;
      ra = rb;
      rb = t;
    }
    parent[rb] = ra;
    size[ra] += size[rb];
    maze_carve(m, c, dir, nb);
  }
}

/* Sidewinder — Buck's rule with the east bias exposed as a parameter (the
 * upstream coin is hard-coded 50/50). Row 0 is one corridor:
 * the run can only be closed by the east wall of the grid, and no draw is made
 * for it. In rows below, a draw is made at every column except the last
 * (where the run is forced closed); closing a run costs one more draw to pick
 * the cell that carves north. Passage count is (w-1) + (h-1)*w = w*h - 1. */
static void maze_sidewinder(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t x, y;
  uint32_t bias = m->params.east_bias_percent;

  for (y = 0u; y < h; ++y) {
    uint32_t run_start = 0u;
    for (x = 0u; x < w; ++x) {
      int close;
      if (x + 1u >= w) {
        close = 1;
      } else if (y == 0u) {
        close = 0;
      } else {
        close = (fbs_maze_rng_below(&m->rng, 100u) >= bias);
      }
      if (!close) {
        maze_carve(m, y * w + x, (uint8_t)FBS_MAZE_E, y * w + x + 1u);
        continue;
      }
      if (y > 0u) {
        uint32_t px = run_start + fbs_maze_rng_below(&m->rng, x - run_start + 1u);
        maze_carve(m, y * w + px, (uint8_t)FBS_MAZE_N, (y - 1u) * w + px);
      }
      run_start = x + 1u;
    }
  }
}

/* Eller — Buck's algorithm with a real union-find over the row (M-1) and one
 * rule for every row including the last (M-2).
 *
 * Per row: (a) for x = 0 .. w-2, if columns x and x+1 are in different sets,
 * merge them with probability 1/2 (rng_below(2)) and carve east; in the LAST
 * row every such pair is merged unconditionally, which is what makes the maze
 * connected. (b) For every set, in ascending order of its lowest column,
 * visit its columns in ascending order and carve south with probability 1/2
 * each; if the set carved none, one of its columns is chosen with
 * rng_below(count) and carved. (c) The next row inherits the sets of the
 * columns that carved south; every other column starts a fresh singleton set.
 *
 * The row union-find is rebuilt per row over column indices, so nothing scales
 * with height: this is the O(width)-memory property Eller exists for. */
static void maze_eller_reset_row(fbs_maze *m, uint32_t w) {
  uint32_t x;
  for (x = 0u; x < w; ++x) {
    m->rparent[x] = x;
    m->rsize[x] = 1u;
  }
}

static void maze_eller_union(fbs_maze *m, uint32_t a, uint32_t b) {
  uint32_t ra = maze_uf_find(m->rparent, a), rb = maze_uf_find(m->rparent, b);
  if (ra == rb) return;
  if (m->rsize[ra] < m->rsize[rb]) {
    uint32_t t = ra;
    ra = rb;
    rb = t;
  }
  m->rparent[rb] = ra;
  m->rsize[ra] += m->rsize[rb];
}

static void maze_eller(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t x, y;

  maze_eller_reset_row(m, w);
  for (y = 0u; y < h; ++y) {
    int last_row = (y + 1u == h);

    /* (a) horizontal merges */
    for (x = 0u; x + 1u < w; ++x) {
      if (maze_uf_find(m->rparent, x) == maze_uf_find(m->rparent, x + 1u)) continue;
      if (!last_row && fbs_maze_rng_below(&m->rng, 2u) == 0u) continue;
      maze_carve(m, y * w + x, (uint8_t)FBS_MAZE_E, y * w + x + 1u);
      maze_eller_union(m, x, x + 1u);
    }
    if (last_row) break;

    /* Bucket the columns of each set into an ascending singly linked list, so
       the sets can be visited in ascending order of their lowest column
       without an O(w^2) scan. */
    for (x = 0u; x < w; ++x) {
      m->rhead[x] = UINT32_MAX;
      m->rnext[x] = UINT32_MAX;
    }
    for (x = 0u; x < w; ++x) {
      uint32_t r = maze_uf_find(m->rparent, x);
      if (m->rhead[r] == UINT32_MAX) {
        m->rhead[r] = x;
      } else {
        m->rnext[m->rtail[r]] = x;
      }
      m->rtail[r] = x;
    }

    /* (b) at least one south passage per set */
    for (x = 0u; x < w; ++x) {
      uint32_t r = maze_uf_find(m->rparent, x);
      uint32_t col, count = 0u, carved = 0u;
      if (m->rhead[r] != x) continue; /* visit each set once, at its lowest column */
      for (col = x; col != UINT32_MAX; col = m->rnext[col]) {
        ++count;
        if (fbs_maze_rng_below(&m->rng, 2u) != 0u) {
          maze_carve(m, y * w + col, (uint8_t)FBS_MAZE_S, (y + 1u) * w + col);
          ++carved;
        }
      }
      if (carved == 0u) {
        uint32_t k = fbs_maze_rng_below(&m->rng, count);
        col = x;
        while (k-- > 0u) col = m->rnext[col];
        maze_carve(m, y * w + col, (uint8_t)FBS_MAZE_S, (y + 1u) * w + col);
      }
    }

    /* (c) carry the sets that came south into the next row */
    for (x = 0u; x < w; ++x) m->rhead[x] = UINT32_MAX;
    for (x = 0u; x < w; ++x) {
      if (m->cells[y * w + x] & FBS_MAZE_S) {
        uint32_t r = maze_uf_find(m->rparent, x);
        m->rtail[x] = m->rhead[r]; /* borrow rtail as "first carried column of r" */
        if (m->rhead[r] == UINT32_MAX) m->rhead[r] = x;
      } else {
        m->rtail[x] = UINT32_MAX;
      }
    }
    maze_eller_reset_row(m, w);
    for (x = 0u; x < w; ++x) {
      if (m->rtail[x] != UINT32_MAX) maze_eller_union(m, m->rtail[x], x);
    }
  }
}

/* Hunt-and-Kill — walk to a random unvisited neighbour until stuck, then scan
 * row-major for the first unvisited cell that touches a visited one and join
 * it to a random visited neighbour. The scan starts at `hunt_row`, the lowest
 * row that still holds an unvisited cell; rows below it are skipped only once
 * they are fully visited, so the scan order is identical to restarting from
 * (0,0) every time, without the cost. O(1) extra state per the header. */
static void maze_hunt_and_kill(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t n = w * h;
  uint32_t current, hunt_row = 0u;

  memset(m->flags, 0, (size_t)n);
  current = fbs_maze_rng_below(&m->rng, n);
  m->flags[current] = 1u;

  for (;;) {
    uint32_t x, y, k, i;
    int found = 0;

    /* kill: walk while an unvisited neighbour exists */
    for (;;) {
      uint32_t nbs[4];
      uint8_t dirs[4];
      unsigned cnt = 0u, d;
      x = current % w;
      y = current / w;
      for (d = 0u; d < 4u; ++d) {
        uint32_t nb;
        if (maze_step(m, x, y, maze_dir_order[d], &nb) && !m->flags[nb]) {
          dirs[cnt] = maze_dir_order[d];
          nbs[cnt] = nb;
          ++cnt;
        }
      }
      if (cnt == 0u) break;
      k = fbs_maze_rng_below(&m->rng, cnt);
      maze_carve(m, current, dirs[k], nbs[k]);
      m->flags[nbs[k]] = 1u;
      current = nbs[k];
    }

    /* hunt: advance past fully visited rows, then scan row-major */
    while (hunt_row < h) {
      uint32_t full = 1u;
      for (i = 0u; i < w; ++i) {
        if (!m->flags[hunt_row * w + i]) {
          full = 0u;
          break;
        }
      }
      if (!full) break;
      ++hunt_row;
    }
    for (y = hunt_row; y < h && !found; ++y) {
      for (x = 0u; x < w; ++x) {
        uint32_t c = y * w + x;
        uint32_t nbs[4];
        uint8_t dirs[4];
        unsigned cnt = 0u, d;
        if (m->flags[c]) continue;
        for (d = 0u; d < 4u; ++d) {
          uint32_t nb;
          if (maze_step(m, x, y, maze_dir_order[d], &nb) && m->flags[nb]) {
            dirs[cnt] = maze_dir_order[d];
            nbs[cnt] = nb;
            ++cnt;
          }
        }
        if (cnt == 0u) continue;
        k = fbs_maze_rng_below(&m->rng, cnt);
        maze_carve(m, c, dirs[k], nbs[k]);
        m->flags[c] = 1u;
        current = c;
        found = 1;
        break;
      }
    }
    if (!found) return;
  }
}

/* Recursive division, with an explicit stack (M-8) — start from a fully open
 * grid and add walls. Orientation is chosen from the aspect ratio at EVERY
 * level including the first (M-5): taller than wide bisects horizontally,
 * wider than tall bisects vertically, square flips a coin. The wall offset is
 * drawn from the full [0, extent-2] range (M-4), so a one-cell sub-region is
 * reachable, and the gap is drawn from the full cross extent. A region with
 * an extent of 1 is already a corridor because the grid started fully open,
 * which is the structural fix for M-6 (upstream carves nothing and returns a
 * disconnected grid). Draw order per split: the orientation coin only when the
 * region is square, then the wall offset, then the gap. Sub-regions are pushed
 * far-half first so the near half (top or left) is processed next. */
static void maze_division(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t x, y, sp = 0u;
  uint32_t *st = m->dstack;

  for (y = 0u; y < h; ++y) {
    for (x = 0u; x < w; ++x) {
      uint8_t c = 0u;
      if (y > 0u) c = (uint8_t)(c | FBS_MAZE_N);
      if (x + 1u < w) c = (uint8_t)(c | FBS_MAZE_E);
      if (y + 1u < h) c = (uint8_t)(c | FBS_MAZE_S);
      if (x > 0u) c = (uint8_t)(c | FBS_MAZE_W);
      m->cells[y * w + x] = c;
    }
  }

  st[0] = 0u;
  st[1] = 0u;
  st[2] = w;
  st[3] = h;
  sp = 1u;

  while (sp > 0u) {
    uint32_t rx, ry, rw, rh;
    int horizontal;
    --sp;
    rx = st[sp * 4u + 0u];
    ry = st[sp * 4u + 1u];
    rw = st[sp * 4u + 2u];
    rh = st[sp * 4u + 3u];
    if (rw < 2u || rh < 2u) continue;
    /* Proven unreachable (see maze_div_cap); kept so a future change to the
       split rule can never write past the stack. */
    if (sp + 2u > m->div_cap) continue;

    if (rh > rw) {
      horizontal = 1;
    } else if (rw > rh) {
      horizontal = 0;
    } else {
      horizontal = (fbs_maze_rng_below(&m->rng, 2u) != 0u);
    }

    if (horizontal) {
      uint32_t off = fbs_maze_rng_below(&m->rng, rh - 1u);
      uint32_t gap = fbs_maze_rng_below(&m->rng, rw);
      uint32_t wy = ry + off;
      for (x = 0u; x < rw; ++x) {
        if (x == gap) continue;
        maze_close(m, wy * w + rx + x, (uint8_t)FBS_MAZE_S, (wy + 1u) * w + rx + x);
      }
      st[sp * 4u + 0u] = rx;
      st[sp * 4u + 1u] = wy + 1u;
      st[sp * 4u + 2u] = rw;
      st[sp * 4u + 3u] = rh - off - 1u;
      ++sp;
      st[sp * 4u + 0u] = rx;
      st[sp * 4u + 1u] = ry;
      st[sp * 4u + 2u] = rw;
      st[sp * 4u + 3u] = off + 1u;
      ++sp;
    } else {
      uint32_t off = fbs_maze_rng_below(&m->rng, rw - 1u);
      uint32_t gap = fbs_maze_rng_below(&m->rng, rh);
      uint32_t wx = rx + off;
      for (y = 0u; y < rh; ++y) {
        if (y == gap) continue;
        maze_close(m, (ry + y) * w + wx, (uint8_t)FBS_MAZE_E, (ry + y) * w + wx + 1u);
      }
      st[sp * 4u + 0u] = wx + 1u;
      st[sp * 4u + 1u] = ry;
      st[sp * 4u + 2u] = rw - off - 1u;
      st[sp * 4u + 3u] = rh;
      ++sp;
      st[sp * 4u + 0u] = rx;
      st[sp * 4u + 1u] = ry;
      st[sp * 4u + 2u] = off + 1u;
      st[sp * 4u + 3u] = rh;
      ++sp;
    }
  }
}

/* Braiding, applied deterministically: cells are visited in ascending index.
 * A cell that is a dead end *at that moment* draws rng_below(100); if the
 * draw is below braid_percent, one of its closed interior walls is opened,
 * chosen with rng_below(count) over the closed walls in N, E, S, W order.
 * Degrees only ever rise, so a dead end that a previous opening already
 * relieved draws nothing. A dead end with no closed interior wall (the two
 * ends of a 1xN corridor) draws the coin and then stays a dead end: there is
 * nothing to open. */
static void maze_braid(fbs_maze *m) {
  uint32_t w = m->width, h = m->height;
  uint32_t n = w * h, i;
  uint32_t pct = m->params.braid_percent;

  for (i = 0u; i < n; ++i) {
    uint32_t x = i % w, y = i / w, k;
    uint32_t nbs[4];
    uint8_t dirs[4];
    unsigned cnt = 0u, d;
    if (maze_popcount4((uint8_t)(m->cells[i] & FBS_MAZE_DIRS)) != 1u) continue;
    if (fbs_maze_rng_below(&m->rng, 100u) >= pct) continue;
    for (d = 0u; d < 4u; ++d) {
      uint32_t nb;
      if (!maze_step(m, x, y, maze_dir_order[d], &nb)) continue;
      if (m->cells[i] & maze_dir_order[d]) continue;
      dirs[cnt] = maze_dir_order[d];
      nbs[cnt] = nb;
      ++cnt;
    }
    if (cnt == 0u) continue;
    k = fbs_maze_rng_below(&m->rng, cnt);
    maze_carve(m, i, dirs[k], nbs[k]);
  }
}

/* ------------------------------------------------------------------------- */
/* generate                                                                  */
/* ------------------------------------------------------------------------- */

/* Validation order, fixed so the statuses are predictable: NULL arguments,
 * then the algorithm enum (M-25), then zero width/height, then the three
 * percents, then the capacity of the maze object. Nothing is written until
 * every check has passed. */
static fbs_maze_status maze_params_check(const fbs_maze *m, const fbs_maze_params *p) {
  if ((int)p->algorithm >= (int)FBS_MAZE_ALGORITHM_COUNT) return FBS_MAZE_E_INVALID;
  if (p->width == 0u || p->height == 0u) return FBS_MAZE_E_RANGE;
  if (p->east_bias_percent > 100u || p->newest_percent > 100u || p->braid_percent > 100u)
    return FBS_MAZE_E_RANGE;
  if (p->width > m->max_width || p->height > m->max_height) return FBS_MAZE_E_FULL;
  return FBS_MAZE_OK;
}

fbs_maze_status fbs_maze_generate(fbs_maze *m, const fbs_maze_params *p) {
  fbs_maze_status st;

  if (!m || !p) return FBS_MAZE_E_INVALID;
  st = maze_params_check(m, p);
  if (st != FBS_MAZE_OK) return st;

  m->params = *p;
  m->width = p->width;
  m->height = p->height;
  memset(m->cells, 0, (size_t)p->width * (size_t)p->height);
  fbs_maze_rng_seed(&m->rng, p->seed);

  switch ((fbs_maze_algorithm)p->algorithm) {
    case FBS_MAZE_BACKTRACKER: maze_growing_tree(m, 100u); break;
    case FBS_MAZE_KRUSKAL: maze_kruskal(m); break;
    case FBS_MAZE_PRIM: maze_growing_tree(m, 0u); break;
    case FBS_MAZE_ELLER: maze_eller(m); break;
    case FBS_MAZE_SIDEWINDER: maze_sidewinder(m); break;
    case FBS_MAZE_HUNT_AND_KILL: maze_hunt_and_kill(m); break;
    case FBS_MAZE_DIVISION: maze_division(m); break;
    case FBS_MAZE_GROWING_TREE: maze_growing_tree(m, p->newest_percent); break;
    default: return FBS_MAZE_E_INVALID; /* unreachable: checked above */
  }
  if (p->braid_percent > 0u) maze_braid(m);

  m->generated = 1;
  return FBS_MAZE_OK;
}

/* ------------------------------------------------------------------------- */
/* Accessors                                                                 */
/* ------------------------------------------------------------------------- */

uint32_t fbs_maze_width(const fbs_maze *m) { return (m && m->generated) ? m->width : 0u; }
uint32_t fbs_maze_height(const fbs_maze *m) { return (m && m->generated) ? m->height : 0u; }

const fbs_maze_params *fbs_maze_params_of(const fbs_maze *m) {
  return (m && m->generated) ? &m->params : NULL;
}

const uint8_t *fbs_maze_cells(const fbs_maze *m) {
  return (m && m->generated) ? m->cells : NULL;
}

fbs_maze_status fbs_maze_cell(const fbs_maze *m, uint32_t x, uint32_t y, uint8_t *out) {
  if (!m || !out) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (x >= m->width || y >= m->height) return FBS_MAZE_E_INVALID;
  *out = m->cells[y * m->width + x];
  return FBS_MAZE_OK;
}

int fbs_maze_is_open(const fbs_maze *m, uint32_t x, uint32_t y, int dir) {
  if (!m) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (x >= m->width || y >= m->height) return FBS_MAZE_E_INVALID;
  if (maze_dir_index(dir) < 0) return FBS_MAZE_E_INVALID;
  return (m->cells[y * m->width + x] & (unsigned)dir) ? 1 : 0;
}

fbs_maze_status fbs_maze_set_passage(fbs_maze *m, uint32_t x, uint32_t y, int dir, int open) {
  uint32_t nb;
  if (!m) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (x >= m->width || y >= m->height) return FBS_MAZE_E_INVALID;
  if (maze_dir_index(dir) < 0) return FBS_MAZE_E_INVALID;
  if (!maze_step(m, x, y, (uint8_t)dir, &nb)) return FBS_MAZE_E_INVALID;
  if (open)
    maze_carve(m, y * m->width + x, (uint8_t)dir, nb);
  else
    maze_close(m, y * m->width + x, (uint8_t)dir, nb);
  return FBS_MAZE_OK;
}

/* ------------------------------------------------------------------------- */
/* Properties                                                                */
/* ------------------------------------------------------------------------- */

/* Interior passages only: each edge is counted once, from its west/north end,
 * and an outward bit on a border cell is never an east/south interior edge. */
static uint32_t maze_count_passages(const uint8_t *cells, uint32_t w, uint32_t h) {
  uint32_t x, y, count = 0u;
  for (y = 0u; y < h; ++y) {
    for (x = 0u; x < w; ++x) {
      uint8_t c = cells[y * w + x];
      if ((x + 1u < w) && (c & FBS_MAZE_E)) ++count;
      if ((y + 1u < h) && (c & FBS_MAZE_S)) ++count;
    }
  }
  return count;
}

static uint32_t maze_count_border(const uint8_t *cells, uint32_t w, uint32_t h) {
  uint32_t i, count = 0u;
  for (i = 0u; i < w; ++i) {
    if (cells[i] & FBS_MAZE_N) ++count;
    if (cells[(h - 1u) * w + i] & FBS_MAZE_S) ++count;
  }
  for (i = 0u; i < h; ++i) {
    if (cells[i * w] & FBS_MAZE_W) ++count;
    if (cells[i * w + w - 1u] & FBS_MAZE_E) ++count;
  }
  return count;
}

uint32_t fbs_maze_passage_count(const fbs_maze *m) {
  if (!m || !m->generated) return 0u;
  return maze_count_passages(m->cells, m->width, m->height);
}

uint32_t fbs_maze_border_opening_count(const fbs_maze *m) {
  if (!m || !m->generated) return 0u;
  return maze_count_border(m->cells, m->width, m->height);
}

uint32_t fbs_maze_dead_end_count(const fbs_maze *m) {
  uint32_t i, n, count = 0u;
  if (!m || !m->generated) return 0u;
  n = m->width * m->height;
  for (i = 0u; i < n; ++i) {
    if (maze_popcount4((uint8_t)(m->cells[i] & FBS_MAZE_DIRS)) == 1u) ++count;
  }
  return count;
}

/* BFS over interior passages, N/E/S/W neighbour order. Exactly one of `dist`
 * and `parent` is used as the visited marker (UINT32_MAX = unvisited); the
 * other may be NULL. Returns the number of cells reached. Border openings lead
 * nowhere and are ignored. Uses only m->queue. */
static uint32_t maze_bfs(const fbs_maze *m, uint32_t start, uint32_t *dist, uint32_t *parent) {
  uint32_t w = m->width, h = m->height;
  uint32_t n = w * h;
  uint32_t *mark = dist ? dist : parent;
  uint32_t head = 0u, tail = 0u, seen = 1u, i;

  for (i = 0u; i < n; ++i) mark[i] = UINT32_MAX;
  if (dist) dist[start] = 0u;
  if (parent) parent[start] = start;
  m->queue[tail++] = start;

  while (head < tail) {
    uint32_t c = m->queue[head++];
    uint32_t x = c % w, y = c / w;
    unsigned d;
    for (d = 0u; d < 4u; ++d) {
      uint32_t nb;
      if (!(m->cells[c] & maze_dir_order[d])) continue;
      if (!maze_step(m, x, y, maze_dir_order[d], &nb)) continue;
      if (mark[nb] != UINT32_MAX) continue;
      if (dist) dist[nb] = dist[c] + 1u;
      if (parent) parent[nb] = c;
      m->queue[tail++] = nb;
      ++seen;
    }
  }
  return seen;
}

int fbs_maze_is_connected(const fbs_maze *m) {
  if (!m || !m->generated) return 0;
  return maze_bfs(m, 0u, m->aux, NULL) == m->width * m->height ? 1 : 0;
}

int fbs_maze_is_perfect(const fbs_maze *m) {
  uint32_t n;
  if (!m || !m->generated) return 0;
  n = m->width * m->height;
  if (fbs_maze_passage_count(m) != n - 1u) return 0;
  return fbs_maze_is_connected(m);
}

/* ------------------------------------------------------------------------- */
/* Entrance / exit                                                           */
/* ------------------------------------------------------------------------- */

static fbs_maze_status maze_border_check(const fbs_maze *m, uint32_t x, uint32_t y, int dir) {
  uint32_t nb;
  if (!m) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (x >= m->width || y >= m->height) return FBS_MAZE_E_INVALID;
  if (maze_dir_index(dir) < 0) return FBS_MAZE_E_INVALID;
  /* the direction must point OUT of the grid */
  if (maze_step(m, x, y, (uint8_t)dir, &nb)) return FBS_MAZE_E_INVALID;
  return FBS_MAZE_OK;
}

fbs_maze_status fbs_maze_open_border(fbs_maze *m, uint32_t x, uint32_t y, int dir) {
  fbs_maze_status st = maze_border_check(m, x, y, dir);
  if (st != FBS_MAZE_OK) return st;
  m->cells[y * m->width + x] = (uint8_t)(m->cells[y * m->width + x] | (unsigned)dir);
  return FBS_MAZE_OK;
}

fbs_maze_status fbs_maze_close_border(fbs_maze *m, uint32_t x, uint32_t y, int dir) {
  fbs_maze_status st = maze_border_check(m, x, y, dir);
  if (st != FBS_MAZE_OK) return st;
  m->cells[y * m->width + x] =
      (uint8_t)(m->cells[y * m->width + x] & (uint8_t)~(unsigned)dir);
  return FBS_MAZE_OK;
}

/* Double BFS: farthest cell from cell 0 (lowest index on a tie), then the
 * farthest cell from that one (lowest index on a tie). Exact for a perfect
 * maze, which is a tree; for a braided maze it is the standard double-BFS
 * estimate the header prescribes. Requires a connected maze — a maze taken
 * apart with fbs_maze_set_passage reports E_UNREACHABLE. */
/* The cell at the greatest distance in m->aux, lowest index first. */
static uint32_t maze_pick_farthest(const fbs_maze *m, uint32_t fallback) {
  uint32_t n = m->width * m->height, i, best = fallback, bestd = 0u;
  for (i = 0u; i < n; ++i) {
    if (m->aux[i] == UINT32_MAX) continue;
    if (m->aux[i] > bestd) {
      bestd = m->aux[i];
      best = i;
    }
  }
  return best;
}

fbs_maze_status fbs_maze_longest_path(const fbs_maze *m, uint32_t *out_ax, uint32_t *out_ay,
                                      uint32_t *out_bx, uint32_t *out_by,
                                      uint32_t *out_distance) {
  uint32_t a, b, d;
  if (!m) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (maze_bfs(m, 0u, m->aux, NULL) != m->width * m->height) return FBS_MAZE_E_UNREACHABLE;
  a = maze_pick_farthest(m, 0u);
  (void)maze_bfs(m, a, m->aux, NULL);
  b = maze_pick_farthest(m, a);
  d = m->aux[b];
  if (out_ax) *out_ax = a % m->width;
  if (out_ay) *out_ay = a / m->width;
  if (out_bx) *out_bx = b % m->width;
  if (out_by) *out_by = b / m->width;
  if (out_distance) *out_distance = d;
  return FBS_MAZE_OK;
}

/* ------------------------------------------------------------------------- */
/* Solver                                                                    */
/* ------------------------------------------------------------------------- */

fbs_maze_status fbs_maze_solve(const fbs_maze *m, uint32_t sx, uint32_t sy, uint32_t ex,
                               uint32_t ey, uint32_t *out_path, size_t path_cap, size_t *out_len) {
  uint32_t start, end, c, len = 1u;
  size_t i;

  if (!m || !out_len) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (sx >= m->width || sy >= m->height || ex >= m->width || ey >= m->height)
    return FBS_MAZE_E_INVALID;
  if (!out_path && path_cap > 0u) return FBS_MAZE_E_INVALID;

  start = sy * m->width + sx;
  end = ey * m->width + ex;
  (void)maze_bfs(m, start, NULL, m->aux);
  if (m->aux[end] == UINT32_MAX) return FBS_MAZE_E_UNREACHABLE;

  for (c = end; c != start; c = m->aux[c]) ++len;
  if (path_cap < (size_t)len) {
    *out_len = (size_t)len;
    return FBS_MAZE_E_TRUNCATED;
  }
  i = (size_t)len;
  c = end;
  for (;;) {
    out_path[--i] = c;
    if (c == start) break;
    c = m->aux[c];
  }
  *out_len = (size_t)len;
  return FBS_MAZE_OK;
}

fbs_maze_status fbs_maze_distances(const fbs_maze *m, uint32_t sx, uint32_t sy, uint32_t *out_dist,
                                   size_t cap) {
  if (!m || !out_dist) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (sx >= m->width || sy >= m->height) return FBS_MAZE_E_INVALID;
  if (cap < (size_t)m->width * (size_t)m->height) return FBS_MAZE_E_TRUNCATED;
  (void)maze_bfs(m, sy * m->width + sx, out_dist, NULL);
  return FBS_MAZE_OK;
}

/* ------------------------------------------------------------------------- */
/* Render                                                                    */
/* ------------------------------------------------------------------------- */

size_t fbs_maze_render_size(const fbs_maze *m) {
  if (!m || !m->generated) return 0u;
  return ((size_t)m->width * 2u + 1u) * ((size_t)m->height * 2u + 1u);
}

fbs_maze_status fbs_maze_render(const fbs_maze *m, uint8_t *buf, size_t cap, uint32_t *out_w,
                                uint32_t *out_h) {
  size_t rw, rh, need;
  uint32_t x, y;

  if (!m) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (!buf && cap > 0u) return FBS_MAZE_E_INVALID;
  rw = (size_t)m->width * 2u + 1u;
  rh = (size_t)m->height * 2u + 1u;
  need = rw * rh;
  if (cap < need) {
    /* the required length is out_w * out_h; nothing else is written */
    if (out_w) *out_w = (uint32_t)rw;
    if (out_h) *out_h = (uint32_t)rh;
    return FBS_MAZE_E_TRUNCATED;
  }

  memset(buf, 0, need);
  for (y = 0u; y < m->height; ++y) {
    for (x = 0u; x < m->width; ++x) {
      uint8_t c = m->cells[y * m->width + x];
      size_t rx = (size_t)x * 2u + 1u, ry = (size_t)y * 2u + 1u;
      buf[ry * rw + rx] = 1u;
      /* Every set bit opens the wall slot next to the cell. An outward bit on
         a border cell lands on the border ring — a legal index by
         construction, which is the (2w+1)x(2h+1) answer to M-7. Interior bits
         are mirrored, so the same slot is written twice; that is harmless. */
      if (c & FBS_MAZE_N) buf[(ry - 1u) * rw + rx] = 1u;
      if (c & FBS_MAZE_E) buf[ry * rw + rx + 1u] = 1u;
      if (c & FBS_MAZE_S) buf[(ry + 1u) * rw + rx] = 1u;
      if (c & FBS_MAZE_W) buf[ry * rw + rx - 1u] = 1u;
    }
  }
  if (out_w) *out_w = (uint32_t)rw;
  if (out_h) *out_h = (uint32_t)rh;
  return FBS_MAZE_OK;
}

/* ------------------------------------------------------------------------- */
/* Serialization (schema "FBSM" v1, see SCHEMA at the top of this file)      */
/* ------------------------------------------------------------------------- */

static void maze_put_u16(unsigned char *p, unsigned v) {
  p[0] = (unsigned char)(v & 0xffu);
  p[1] = (unsigned char)((v >> 8) & 0xffu);
}

static void maze_put_u32(unsigned char *p, uint32_t v) {
  p[0] = (unsigned char)(v & 0xffu);
  p[1] = (unsigned char)((v >> 8) & 0xffu);
  p[2] = (unsigned char)((v >> 16) & 0xffu);
  p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void maze_put_u64(unsigned char *p, uint64_t v) {
  maze_put_u32(p, (uint32_t)(v & 0xffffffffu));
  maze_put_u32(p + 4, (uint32_t)((v >> 32) & 0xffffffffu));
}

static unsigned maze_get_u16(const unsigned char *p) {
  return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static uint32_t maze_get_u32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t maze_get_u64(const unsigned char *p) {
  return (uint64_t)maze_get_u32(p) | ((uint64_t)maze_get_u32(p + 4) << 32);
}

size_t fbs_maze_serialized_size(const fbs_maze *m) {
  if (!m || !m->generated) return 0u;
  return (size_t)MAZE_HEADER_BYTES + (size_t)m->width * (size_t)m->height;
}

fbs_maze_status fbs_maze_serialize(const fbs_maze *m, void *buf, size_t cap, size_t *out_len) {
  unsigned char *p;
  size_t need;

  if (!m || !out_len) return FBS_MAZE_E_INVALID;
  if (!m->generated) return FBS_MAZE_E_STATE;
  if (!buf && cap > 0u) return FBS_MAZE_E_INVALID;
  need = fbs_maze_serialized_size(m);
  if (cap < need) {
    *out_len = need;
    return FBS_MAZE_E_TRUNCATED;
  }

  p = (unsigned char *)buf;
  p[0] = 'F';
  p[1] = 'B';
  p[2] = 'S';
  p[3] = 'M';
  maze_put_u16(p + 4, MAZE_SCHEMA_VERSION);
  maze_put_u16(p + 6, 0u);
  maze_put_u32(p + 8, m->width);
  maze_put_u32(p + 12, m->height);
  maze_put_u64(p + 16, m->params.seed);
  p[24] = m->params.algorithm;
  p[25] = m->params.east_bias_percent;
  p[26] = m->params.newest_percent;
  p[27] = m->params.braid_percent;
  p[28] = (unsigned char)MAZE_RNG_ID;
  p[29] = 0u;
  p[30] = 0u;
  p[31] = 0u;
  maze_put_u32(p + 32, maze_count_passages(m->cells, m->width, m->height));
  maze_put_u32(p + 36, maze_count_border(m->cells, m->width, m->height));
  memcpy(p + MAZE_HEADER_BYTES, m->cells, (size_t)m->width * (size_t)m->height);
  *out_len = need;
  return FBS_MAZE_OK;
}

/* Every interior edge must be mirrored: bit d on (x,y) implies opposite(d) on
 * the neighbour. Only outward bits on border cells are exempt. */
static int maze_edges_mirrored(const unsigned char *cells, uint32_t w, uint32_t h) {
  uint32_t x, y;
  for (y = 0u; y < h; ++y) {
    for (x = 0u; x < w; ++x) {
      unsigned char c = cells[y * w + x];
      if (x + 1u < w) {
        unsigned char e = cells[y * w + x + 1u];
        if (((c & FBS_MAZE_E) != 0u) != ((e & FBS_MAZE_W) != 0u)) return 0;
      }
      if (y + 1u < h) {
        unsigned char s = cells[(y + 1u) * w + x];
        if (((c & FBS_MAZE_S) != 0u) != ((s & FBS_MAZE_N) != 0u)) return 0;
      }
    }
  }
  return 1;
}

fbs_maze_status fbs_maze_deserialize(const void *buf, size_t len, const fbs_maze_config *cfg,
                                     const fbs_maze_allocator *alloc, fbs_maze **out) {
  const unsigned char *p = (const unsigned char *)buf;
  fbs_maze_config c;
  fbs_maze *m = NULL;
  fbs_maze_status st;
  uint32_t w, h, i, n, passages, borders;

  if (!buf || !out) return FBS_MAZE_E_INVALID;
  if (len < (size_t)MAZE_HEADER_BYTES) return FBS_MAZE_E_SCHEMA;
  if (p[0] != 'F' || p[1] != 'B' || p[2] != 'S' || p[3] != 'M') return FBS_MAZE_E_SCHEMA;
  if (maze_get_u16(p + 4) != MAZE_SCHEMA_VERSION) return FBS_MAZE_E_SCHEMA;
  if (maze_get_u16(p + 6) != 0u) return FBS_MAZE_E_SCHEMA;

  w = maze_get_u32(p + 8);
  h = maze_get_u32(p + 12);
  if (w == 0u || h == 0u || w > MAZE_MAX_DIM || h > MAZE_MAX_DIM) return FBS_MAZE_E_SCHEMA;
  if ((uint64_t)w * (uint64_t)h > (uint64_t)MAZE_MAX_CELLS) return FBS_MAZE_E_SCHEMA;
  n = w * h;
  if (len != (size_t)MAZE_HEADER_BYTES + (size_t)n) return FBS_MAZE_E_SCHEMA;

  if ((int)p[24] >= (int)FBS_MAZE_ALGORITHM_COUNT) return FBS_MAZE_E_SCHEMA;
  if (p[25] > 100u || p[26] > 100u || p[27] > 100u) return FBS_MAZE_E_SCHEMA;
  if (p[28] != (unsigned char)MAZE_RNG_ID) return FBS_MAZE_E_SCHEMA;
  if (p[29] != 0u || p[30] != 0u || p[31] != 0u) return FBS_MAZE_E_SCHEMA;

  /* Capacity is checked before the cell scan: a blob that does not fit the
     caller's config is E_FULL whatever its contents say. */
  c = cfg ? *cfg : fbs_maze_config_default();
  if (!cfg) {
    c.max_width = w;
    c.max_height = h;
  } else {
    if (!maze_config_valid(&c)) return FBS_MAZE_E_RANGE;
    if (w > c.max_width || h > c.max_height) return FBS_MAZE_E_FULL;
  }

  for (i = 0u; i < n; ++i) {
    if ((p[MAZE_HEADER_BYTES + i] & 0xF0u) != 0u) return FBS_MAZE_E_SCHEMA; /* M-12 */
  }
  if (!maze_edges_mirrored(p + MAZE_HEADER_BYTES, w, h)) return FBS_MAZE_E_SCHEMA;
  passages = maze_count_passages(p + MAZE_HEADER_BYTES, w, h);
  borders = maze_count_border(p + MAZE_HEADER_BYTES, w, h);
  if (maze_get_u32(p + 32) != passages) return FBS_MAZE_E_SCHEMA;
  if (maze_get_u32(p + 36) != borders) return FBS_MAZE_E_SCHEMA;

  st = fbs_maze_create(&c, alloc, &m);
  if (st != FBS_MAZE_OK) return st;

  m->width = w;
  m->height = h;
  m->params.width = w;
  m->params.height = h;
  m->params.seed = maze_get_u64(p + 16);
  m->params.algorithm = p[24];
  m->params.east_bias_percent = p[25];
  m->params.newest_percent = p[26];
  m->params.braid_percent = p[27];
  memcpy(m->cells, p + MAZE_HEADER_BYTES, (size_t)n);
  fbs_maze_rng_seed(&m->rng, m->params.seed);
  m->generated = 1;

  *out = m;
  return FBS_MAZE_OK;
}
