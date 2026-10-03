
/* fbs/maze.h — FinalBuildSystems seeded maze generator. C99, engine
 * independent, no libm, no floating point, no recursion.
 *
 * Model (this document): a MAZE is a width x height grid of cells, one byte
 * each, whose low four bits say which of N/E/S/W passages are open. Generation
 * is a pure function of (params) -- algorithm, seed, size, bias knobs -- using
 * an explicitly specified PRNG (xoshiro128** 1.1 seeded through splitmix64) so
 * that the same params give byte-identical cells on every platform including
 * wasm32. One allocation, sized by the config, holds the cells, the algorithm
 * scratch and the solver scratch; nothing is allocated during generation. No
 * globals, no floating point, no libm, no recursion. Decision record:
 * docs/decisions/maze.md section 9. Errors leave outputs untouched except
 * FBS_MAZE_E_TRUNCATED (required length written).
 */
#ifndef FBS_MAZE_H
#define FBS_MAZE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FBS_MAZE_VERSION 100 /* major * 10000 + minor * 100 + patch */

/* ---- cells ----------------------------------------------------------- */
/* Ordered so that opposite(d) is a 2-bit rotation, not a switch. */
enum {
  FBS_MAZE_N = 1u, FBS_MAZE_E = 2u, FBS_MAZE_S = 4u, FBS_MAZE_W = 8u,
  FBS_MAZE_DIRS = 15u          /* the low nibble; the high nibble is reserved 0 */
};
#define FBS_MAZE_OPPOSITE(d) ((uint8_t)(((((unsigned)(d)) << 2) | (((unsigned)(d)) >> 2)) & 15u))

/* ---- status ---------------------------------------------------------- */

typedef enum fbs_maze_status {
  FBS_MAZE_OK            =  0,
  FBS_MAZE_E_INVALID     = -1,  /* NULL pointer, bad enum, bad direction, coordinates out of range */
  FBS_MAZE_E_RANGE       = -2,  /* width/height outside [1, config max], percent outside [0,100] */
  FBS_MAZE_E_FULL        = -3,  /* params exceed the capacity the maze was created with */
  FBS_MAZE_E_UNREACHABLE = -4,  /* solver: no path from start to end (impossible in a generated maze;
                                   possible after fbs_maze_set_passage edits) */
  FBS_MAZE_E_SCHEMA      = -5,  /* blob magic/version/length/consistency mismatch */
  FBS_MAZE_E_TRUNCATED   = -6,  /* output buffer too small; the required length is written */
  FBS_MAZE_E_STATE       = -7,  /* operation needs a generated maze */
  FBS_MAZE_E_MEMORY      = -8   /* allocator returned NULL */
} fbs_maze_status;

const char *fbs_maze_status_name(int status);
unsigned    fbs_maze_version(void);

typedef struct fbs_maze_allocator {
  void *(*alloc)(void *user, size_t bytes);
  void  (*free)(void *user, void *ptr);
  void  *user;
} fbs_maze_allocator;

/* ---- PRNG (public, because the maze bytes are defined by it) ---------- */
/* xoshiro128** 1.1, Blackman & Vigna 2018, public domain. State seeded by
   splitmix64 (Vigna 2015, public domain), which guarantees a non-zero state
   even for seed 0. Pinned by name AND version: xoshiro128** 1.0 scrambled
   s[0] instead of s[1] and produces a different stream. */
typedef struct fbs_maze_rng { uint32_t s[4]; } fbs_maze_rng;

void     fbs_maze_rng_seed (fbs_maze_rng *r, uint64_t seed);
uint32_t fbs_maze_rng_next (fbs_maze_rng *r);
/* Unbiased. bound == 0 returns 0 and consumes nothing (documented, not UB --
   cf. mazelib's SIGFPE, decision s3.1). Rejection, never modulo alone. */
uint32_t fbs_maze_rng_below(fbs_maze_rng *r, uint32_t bound);

/* ---- configuration --------------------------------------------------- */

typedef struct fbs_maze_config {
  uint32_t max_width;   /* 1..4096 */
  uint32_t max_height;  /* 1..4096; max_width * max_height <= 1u << 24 */
} fbs_maze_config;

fbs_maze_config fbs_maze_config_default(void); /* 257 x 257 */

typedef enum fbs_maze_algorithm {
  FBS_MAZE_BACKTRACKER   = 0, /* DFS: long corridors, few dead ends */
  FBS_MAZE_KRUSKAL       = 1, /* random spanning tree via union-find: many short dead ends */
  FBS_MAZE_PRIM          = 2, /* simplified Prim (growing tree, random selection): many short dead ends; identical to GROWING_TREE with newest_percent 0 */
  FBS_MAZE_ELLER         = 3, /* row at a time, O(width) memory */
  FBS_MAZE_SIDEWINDER    = 4, /* top row is one corridor; no upward dead ends */
  FBS_MAZE_HUNT_AND_KILL = 5, /* DFS-like texture, O(1) extra memory, O(n^2) time */
  FBS_MAZE_DIVISION      = 6, /* wall adder: rectangular rooms, long straight walls */
  FBS_MAZE_GROWING_TREE  = 7, /* newest_percent 100 == Backtracker, 0 == Prim, in between blends */
  FBS_MAZE_ALGORITHM_COUNT
} fbs_maze_algorithm;

const char *fbs_maze_algorithm_name(int algorithm);

typedef struct fbs_maze_params {
  uint32_t width;              /* >= 1 */
  uint32_t height;             /* >= 1 */
  uint64_t seed;
  uint8_t  algorithm;          /* fbs_maze_algorithm */
  uint8_t  east_bias_percent;  /* SIDEWINDER only: P(carve east). 0..100, default 50 */
  uint8_t  newest_percent;     /* GROWING_TREE only: P(pick newest). 0..100, default 50 */
  uint8_t  braid_percent;      /* all: P(open one wall of each dead end). 0..100, default 0 */
} fbs_maze_params;

fbs_maze_params fbs_maze_params_default(void); /* 16x16, seed 0, BACKTRACKER, 50/50/0 */

/* ---- lifetime -------------------------------------------------------- */
/* One allocation: cells + algorithm scratch + solver scratch, all sized from
   the config. Nothing is allocated by generate(), solve() or render(). */

typedef struct fbs_maze fbs_maze; /* opaque */

fbs_maze_status fbs_maze_create (const fbs_maze_config *cfg, const fbs_maze_allocator *alloc,
                                 fbs_maze **out);
void            fbs_maze_destroy(fbs_maze *m);
size_t          fbs_maze_memory (const fbs_maze *m);
size_t          fbs_maze_memory_for(const fbs_maze_config *cfg); /* budget without allocating */

/* ---- generation ------------------------------------------------------ */
/* Pure in (params): same params -> byte-identical cells, on every platform.
   Overwrites everything, including any border openings. */
fbs_maze_status fbs_maze_generate(fbs_maze *m, const fbs_maze_params *p);

uint32_t               fbs_maze_width (const fbs_maze *m);
uint32_t               fbs_maze_height(const fbs_maze *m);
const fbs_maze_params *fbs_maze_params_of(const fbs_maze *m);

/* Row-major, index = y * width + x. Returns a read-only view of width*height
   bytes; the high nibble of every byte is 0. */
const uint8_t  *fbs_maze_cells(const fbs_maze *m);
fbs_maze_status fbs_maze_cell (const fbs_maze *m, uint32_t x, uint32_t y, uint8_t *out);
/* 1 open, 0 closed, negative on error. */
int             fbs_maze_is_open(const fbs_maze *m, uint32_t x, uint32_t y, int dir);
/* Hand-editing for tests and level tools: sets or clears one interior passage
 * on BOTH cells (mirrored). E_INVALID when the neighbour is outside the grid
 * (use fbs_maze_open_border for border openings). Requires a generated maze. */
fbs_maze_status fbs_maze_set_passage(fbs_maze *m, uint32_t x, uint32_t y, int dir, int open);

/* ---- properties ------------------------------------------------------ */
/* Counted, not asserted. A perfect maze is connected AND has exactly
   cells-1 passages; braiding deliberately breaks the second half, and this
   API says so instead of a README claiming otherwise (see s5, M-1). */
uint32_t fbs_maze_passage_count(const fbs_maze *m); /* interior passages, border openings excluded */
uint32_t fbs_maze_dead_end_count(const fbs_maze *m); /* cells with exactly one set direction bit, border openings included */
int      fbs_maze_is_connected(const fbs_maze *m);  /* every cell reachable from (0,0) */
int      fbs_maze_is_perfect(const fbs_maze *m);    /* connected && passages == w*h - 1 */

/* ---- entrance / exit (absent upstream: bug M-21) --------------------- */
/* `dir` must point out of the grid from (x,y); the outward bit is set on that
   border cell and the render step turns it into a hole in the border ring.
   This is the case Algorithm.cpp:70/74 would index negatively (M-7). */
fbs_maze_status fbs_maze_open_border (fbs_maze *m, uint32_t x, uint32_t y, int dir);
fbs_maze_status fbs_maze_close_border(fbs_maze *m, uint32_t x, uint32_t y, int dir);
uint32_t        fbs_maze_border_opening_count(const fbs_maze *m);

/* Double BFS: the two cells at maximum shortest-path distance from each other
   in the (connected) maze. Deterministic tie-break: lowest cell index wins.
   Exact on a perfect maze, a lower-bound estimate on a braided one;
   E_UNREACHABLE on a disconnected maze. The natural default entrance/exit
   placement. */
fbs_maze_status fbs_maze_longest_path(const fbs_maze *m,
                                      uint32_t *out_ax, uint32_t *out_ay,
                                      uint32_t *out_bx, uint32_t *out_by,
                                      uint32_t *out_distance);

/* ---- solver ---------------------------------------------------------- */
/* BFS over cells; neighbour order is N, E, S, W (fixed, so paths are
   deterministic). Path is written in travel order, start first, as cell
   indices y*width+x. *out_len is the number of CELLS (path of one cell when
   start == end), never the ambiguous "cells of the rendered grid" the source
   reports (M-23). E_TRUNCATED writes the required length. */
fbs_maze_status fbs_maze_solve(const fbs_maze *m,
                               uint32_t sx, uint32_t sy, uint32_t ex, uint32_t ey,
                               uint32_t *out_path, size_t path_cap, size_t *out_len);

/* Full BFS distance field from one cell; UINT32_MAX for unreachable cells.
   `out_dist` must hold width*height entries. */
fbs_maze_status fbs_maze_distances(const fbs_maze *m, uint32_t sx, uint32_t sy,
                                   uint32_t *out_dist, size_t cap);

/* ---- render (what a mesh builder consumes) --------------------------- */
/* (2w+1) x (2h+1) bytes, row-major, 1 = floor, 0 = wall, with a full wall
   border that border openings punch through. This is Buck's "blockwise
   geometry"; the source instead emits (2w-1) and draws the ring separately
   (Algorithm.cpp:48-80, Maze.cpp:160-182), which is why even sizes waste a
   row and a column there (M-18). */
size_t          fbs_maze_render_size(const fbs_maze *m); /* (2w+1) * (2h+1) */
fbs_maze_status fbs_maze_render(const fbs_maze *m, uint8_t *buf, size_t cap,
                                uint32_t *out_w, uint32_t *out_h);

/* ---- serialization --------------------------------------------------- */
/* Deterministic: the same logical maze serializes to the same bytes on every
   platform. Round-trip is a fixed point at the byte level. */
size_t          fbs_maze_serialized_size(const fbs_maze *m); /* 40 + width*height */
fbs_maze_status fbs_maze_serialize  (const fbs_maze *m, void *buf, size_t cap, size_t *out_len);
fbs_maze_status fbs_maze_deserialize(const void *buf, size_t len, const fbs_maze_config *cfg,
                                     const fbs_maze_allocator *alloc, fbs_maze **out);

#ifdef __cplusplus
}
#endif
#endif /* FBS_MAZE_H */
