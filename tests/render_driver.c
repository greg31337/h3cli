/* Test-only injection of a synthetic worker into the production runner. */
#include "src/cli/render.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    h3_request request = {0}; char error[512];
    if (!h3_request_parse_argv(&request, argc-2, argv+2, H3_SOURCE_NATIVE, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); h3_request_free(&request); return 2;
    }
    int status = h3_render_run(&request, argv[1], NULL);
    h3_request_free(&request); return status;
}
