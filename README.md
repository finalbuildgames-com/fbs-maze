# fbs-maze

Seeded 2D grid maze generation in C99, with a BFS solver, a floor/wall render grid and a binary save format.

## What it does

You create a maze object once with a maximum size, then call `fbs_maze_generate` with a `fbs_maze_params` (width, height, 64-bit seed, algorithm, tuning percentages). The result is a row-major array of `width * height` bytes from `fbs_maze_cells`, where the low four bits of each byte say which of the N/E/S/W passages are open.

- Eight algorithms (`fbs_maze_algorithm`): recursive backtracker, Kruskal, Prim, Eller, sidewinder, hunt-and-kill, recursive division and growing tree. Backtracker and Prim are the growing tree at `newest_percent` 100 and 0.
- Tuning: `east_bias_percent` for sidewinder, `newest_percent` for growing tree, and `braid_percent` for any algorithm, which opens a wall at that share of dead ends to create loops.
- Entrances and exits: `fbs_maze_open_border` and `fbs_maze_close_border` on border cells. `fbs_maze_longest_path` finds the two cells farthest apart, which is a good default for start and goal.
- Queries: `fbs_maze_solve` writes the shortest path as cell indices, `fbs_maze_distances` fills a BFS distance field, and `fbs_maze_passage_count`, `fbs_maze_dead_end_count`, `fbs_maze_is_connected` and `fbs_maze_is_perfect` report structure.
- `fbs_maze_set_passage` edits single interior walls (both sides at once) for level tools.
- `fbs_maze_render` produces a `(2w+1) x (2h+1)` byte grid (1 = floor, 0 = wall) with a solid outer wall that border openings punch through. This is the grid a tile or mesh builder consumes.
- `fbs_maze_serialize` and `fbs_maze_deserialize` use a fixed little-endian format: a 40-byte header plus one byte per cell.
- The PRNG is public (`fbs_maze_rng_seed`, `fbs_maze_rng_next`, `fbs_maze_rng_below`): xoshiro128** 1.1 seeded through splitmix64, so other code can reproduce the same stream.

## When to use it

- You want reproducible mazes from a seed, for example to share a level by seed or regenerate it on a server.
- You need a rectangular grid of up to 4096 x 4096 cells and want the textures of the classic algorithms to choose from.
- You want to control memory: one allocation per maze, through your own allocator if you like.
- You want a solver and distance field without writing your own BFS.

## When not to use it

- Grids are rectangular with four-way connectivity only. There are no hex, circular, 3D or masked (non-rectangular) mazes.
- Limits are fixed: each side is 1 to 4096 cells and `max_width * max_height` is at most 2^24 cells. Params larger than the maze's configured capacity return `FBS_MAZE_E_FULL`; create a new maze object for a bigger size.
- Memory is sized for the configured maximum, not for the maze you generate: about 26 bytes per cell of capacity plus small per-row buffers. `fbs_maze_memory_for` reports the exact figure before you allocate.
- The solver is unweighted BFS. There are no movement costs or diagonal moves.
- On a braided maze (one with loops) `fbs_maze_longest_path` returns a lower-bound estimate, not the exact longest shortest path.
- Hunt-and-kill is O(n^2) in time.
- One maze object must not be used from two threads at once, even for read-only calls (see Design notes).
- There is no mesh output. `fbs_maze_render` gives a byte grid; turning it into geometry is up to you.

## Example

```c
#include <fbs/maze.h>
#include <stdio.h>
#include <stdlib.h>

#define W 16u
#define H 8u

int main(void) {
    fbs_maze_params p = fbs_maze_params_default();
    fbs_maze *maze = NULL;
    uint32_t path[W * H], gw = 0, gh = 0, x, y;
    size_t len = 0, i;
    uint8_t *grid = NULL;
    int ok = 0;

    p.width = W; p.height = H; p.seed = 42; p.algorithm = (uint8_t)FBS_MAZE_KRUSKAL;
    if (fbs_maze_create(NULL, NULL, &maze) != FBS_MAZE_OK) return 1; /* default 257 x 257 capacity */
    if (fbs_maze_generate(maze, &p) == FBS_MAZE_OK &&
        fbs_maze_open_border(maze, 0, 0, FBS_MAZE_W) == FBS_MAZE_OK &&
        fbs_maze_open_border(maze, W - 1, H - 1, FBS_MAZE_E) == FBS_MAZE_OK &&
        fbs_maze_solve(maze, 0, 0, W - 1, H - 1, path, W * H, &len) == FBS_MAZE_OK) {
        grid = (uint8_t *)malloc(fbs_maze_render_size(maze));
        ok = grid && fbs_maze_render(maze, grid, fbs_maze_render_size(maze), &gw, &gh) == FBS_MAZE_OK;
    }
    if (ok) {
        for (i = 0; i < len; ++i) { /* cell (x,y) is (2x+1, 2y+1) in the render grid */
            uint32_t cx = 2 * (path[i] % W) + 1, cy = 2 * (path[i] / W) + 1;
            grid[cy * gw + cx] = 2;
            if (i + 1 < len) { /* also mark the opening between this cell and the next */
                uint32_t nx = 2 * (path[i + 1] % W) + 1, ny = 2 * (path[i + 1] / W) + 1;
                grid[(cy + ny) / 2 * gw + (cx + nx) / 2] = 2;
            }
        }
        for (y = 0; y < gh; ++y) {
            for (x = 0; x < gw; ++x) putchar("# ."[grid[y * gw + x]]); /* wall, floor, path */
            putchar('\n');
        }
        printf("%s maze, path of %lu cells\n", fbs_maze_algorithm_name(p.algorithm), (unsigned long)len);
    }
    free(grid);
    fbs_maze_destroy(maze);
    return ok ? 0 : 1;
}
```

Build it by adding `add_executable(maze_demo main.c)` and `target_link_libraries(maze_demo PRIVATE fbs::maze)` to a project that includes this repository.

## Build and test

Requires a C99 compiler and CMake 3.16 or newer. The library uses only the standard C library (`malloc`/`free` for the default allocator, `memset`/`memcpy`); it has no floating point and no vendored code. On non-MSVC toolchains the generated CMake also links `libm`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --no-tests=error
```

CTest runs two tests. `maze` runs `tests/test_maze.c` against the golden files in `tests/fixtures/maze/`. `maze_example` runs `examples/basic.c` (built as `fbs_maze_example`), which creates a maze and prints the API version. Options: `FBS_BUILD_TESTS` and `FBS_BUILD_EXAMPLES`, both ON by default.

`tests/test_maze.c` checks, among other things:

- every algorithm, over ten sizes from 1x1 to 64x64 and 64 seeds each, gives a connected perfect maze (exactly `w*h - 1` passages, no cycle, checked by the test's own BFS and cycle finder);
- the same params give the same bytes 100 times in a row and regardless of maze capacity, and serialized mazes match the committed `*-17x11-5eed.bin` fixtures for all eight algorithms;
- the PRNG matches the golden vectors in `rng-vectors.txt`, the bounded draw is unbiased (chi-square tests), and the shuffle is fair;
- `braid_percent` 100 removes every dead end and keeps the maze connected;
- solver paths are valid walks whose length equals the BFS distance plus one;
- every error status, every truncation path, and that failed calls leave caller buffers untouched;
- a counting allocator sees exactly one allocation per maze and none from generate, solve, render or serialize;
- corrupted blobs are rejected by `fbs_maze_deserialize`;
- a 1024 x 1024 maze generates, solves, renders and round-trips. Set `FBS_MAZE_TEST_FAST=1` to scale the statistical loops and this size down.

To use it from another CMake project, use `add_subdirectory` or FetchContent and link `fbs::maze`:

```cmake
include(FetchContent)
set(FBS_BUILD_TESTS OFF)
set(FBS_BUILD_EXAMPLES OFF)
FetchContent_Declare(fbs_maze
  GIT_REPOSITORY https://github.com/finalbuildgames-com/fbs-maze.git
  GIT_TAG <full commit hash>) # pin a reviewed commit
FetchContent_MakeAvailable(fbs_maze)
target_link_libraries(your_game PRIVATE fbs::maze)
```

`cmake --install` copies the library, header and license files but no CMake package config file, so `find_package` is not supported. This repository ships the C library only; no engine bindings or adapters are included.

## Design notes

- **Determinism.** Generation is a pure function of the params. All randomness comes from one seeded xoshiro128** 1.1 stream, there is no floating point, and every scan uses a fixed N, E, S, W order, so the header specifies the same bytes on every platform. Serialization writes each field little-endian, one at a time. Solver ties are broken by the fixed neighbour order, and `fbs_maze_longest_path` ties go to the lowest cell index.
- **Memory.** `fbs_maze_create` makes exactly one allocation holding the cells and all generator and solver scratch. Nothing else allocates, apart from `fbs_maze_deserialize`, which creates a new maze. Pass a `fbs_maze_allocator` (alloc, free, user pointer) to route that allocation, or NULL for `malloc`/`free`. Generation uses no recursion.
- **Output buffers.** Path, distance, render and serialize buffers are caller-owned. When a buffer is too small, calls return `FBS_MAZE_E_TRUNCATED` and report the required size (`*out_len`, or `out_w * out_h` for render).
- **Threading.** There is no global or static mutable state, so separate maze objects can be used from separate threads. A single object is not thread-safe: `fbs_maze_solve`, `fbs_maze_distances`, `fbs_maze_longest_path`, `fbs_maze_is_connected` and `fbs_maze_is_perfect` take a `const fbs_maze *` but write the maze's internal scratch arrays.
- **Errors.** Functions return `fbs_maze_status` (0 on success, negative on error) and `fbs_maze_status_name` turns a code into text. On error, outputs are left untouched except for the truncation case above. Status-returning calls that need a generated maze return `FBS_MAZE_E_STATE` before the first successful generate; the count and size queries return 0.
- **Deserialization.** `fbs_maze_deserialize` checks the magic (`FBSM`), version, sizes, enum and percent ranges, reserved bits, wall mirroring and stored counts before allocating anything. With a NULL config, the new maze's capacity is exactly the stored size.
- **Border openings.** `fbs_maze_generate` overwrites the whole grid, including border openings, so open entrances again after each generate.
- **Versioning.** `FBS_MAZE_VERSION` and `fbs_maze_version()` give `major * 10000 + minor * 100 + patch` (currently 100, i.e. 0.1.0). The maze type is opaque; public structs have no size fields.

## License

MIT for Final Build Games' original code; see [LICENSE](LICENSE). The PRNG implements David Blackman and Sebastiano Vigna's public-domain xoshiro128** 1.1 and splitmix64. The "Maze Generator" Unreal plugin by LowkeyMe (MIT) was used as a reference only; none of its runtime source is included. Its license is reproduced in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
