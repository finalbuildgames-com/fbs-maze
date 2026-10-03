#include <fbs/maze.h>
#include <stdio.h>
int main(void) {
    fbs_maze_config config = fbs_maze_config_default();
    fbs_maze *context = NULL;
    if (fbs_maze_create(&config, NULL, &context) != 0) return 1;
    printf("API version: %u\n", fbs_maze_version());
    fbs_maze_destroy(context);
    return 0;
}
