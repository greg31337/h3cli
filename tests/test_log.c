#include "src/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, callbacks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void line(void *opaque) { CHECK(opaque == &callbacks); callbacks++; }

int main(void) {
    unsetenv("H3_PROFILE"); unsetenv("H3_VERBOSE");
    CHECK(!h3_log_verbose());
    int evaluated = 0;
    H3_VERBOSE("disabled %d\n", ++evaluated);
    CHECK(!evaluated);
    setenv("H3_VERBOSE", "1", 1);
    CHECK(h3_log_verbose() && !getenv("H3_PROFILE"));
    CHECK(h3_log_exchange_verbose(0) == -1);
    CHECK(!h3_log_verbose());
    int previous = h3_log_exchange_verbose(1);
    CHECK(previous == 0 && h3_log_verbose());
    h3_log_line_callback(line, &callbacks);
    H3_VERBOSE("verbose diagnostic fixture\n");
    CHECK(callbacks == 1);
    h3_log_exchange_verbose(previous);
    H3_VERBOSE("disabled diagnostic fixture\n");
    CHECK(callbacks == 1);
    h3_log_line_callback(NULL, NULL);
    h3_log_exchange_verbose(-1);
    unsetenv("H3_VERBOSE"); setenv("H3_PROFILE", "1", 1);
    CHECK(h3_log_verbose());
    unsetenv("H3_PROFILE");
    CHECK(!h3_log_verbose());
    printf("ok: %d diagnostic gating and request scope checks\n", checks);
    return 0;
}
