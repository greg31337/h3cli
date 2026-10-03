#include "src/memory.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
#define CHECK(c) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)
typedef struct { uint64_t bytes; int ok; } reading;
static int query(uint64_t *bytes, void *opaque) {
    reading *r = opaque; *bytes = r->bytes; return r->ok;
}
static void sample(h3_memory_snapshot *s,void *opaque) {
    reading *r=opaque;
    *s=(h3_memory_snapshot){.physical_total=UINT64_C(128)<<30,
        .available=r->bytes,.available_valid=r->ok,.process_valid=1,.system_valid=1};
}
static void copy_sample(h3_memory_snapshot *s,void *opaque){*s=*(h3_memory_snapshot*)opaque;}

int main(void) {
    unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
    unsetenv("H3_MEMORY_LIMIT_BYTES");
    uint64_t actual;
    if (h3_memory_available_bytes(&actual))
        printf("OS reclaimable headroom: %" PRIu64 " bytes\n", actual);
    CHECK(!h3_memory_available_bytes(NULL));
    CHECK(h3_memory_minimum_bytes() == H3_MIN_AVAILABLE_MEMORY);
    reading r = {H3_MIN_AVAILABLE_MEMORY, 1};
    h3_memory_set_test_query(query, &r);
    h3_memory_set_test_snapshot(sample,&r);
    char error[512] = {0};
    CHECK(h3_memory_check(0, "unit", error, sizeof(error)));
    CHECK(!h3_memory_check(1, "reserve boundary", error, sizeof(error)));
    CHECK(strstr(error, "reserve boundary") && strstr(error, "available=10737418240") &&
          strstr(error, "floor=10737418240") && strstr(error, "reserve=1 bytes"));
    r.bytes--;
    CHECK(!h3_memory_check(0, "Qwen", error, sizeof(error)));
    CHECK(strstr(error, "Qwen"));
    r.bytes = UINT64_MAX;
    CHECK(!h3_memory_check(UINT64_MAX, "overflow", error, sizeof(error)));
    CHECK(!h3_memory_check(UINT64_MAX - H3_MIN_AVAILABLE_MEMORY, "overflow cap", error, sizeof(error)));
    r.ok = 0;
    CHECK(!h3_memory_check(UINT64_MAX, "query failure", error, sizeof(error)));
    CHECK(strstr(error,"cannot query")&&h3_memory_error(error));
    r.ok=1;
    CHECK(!h3_memory_checkpoint(1, "user", error, sizeof(error)));
    CHECK(strstr(error, "cancelled during user"));
    r.ok = 1;
    setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES", "18446744073709551615", 1);
    CHECK(h3_memory_check(0, "max floor", error, sizeof(error)));
    CHECK(!h3_memory_check(1, "overflow floor", error, sizeof(error)));
    setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES", "16", 1);
    r.bytes = 17;
    CHECK(h3_memory_check(1, "override", error, sizeof(error)));
    CHECK(!h3_memory_check(2, "override", error, sizeof(error)));
    r.bytes = 15;
    CHECK(!h3_memory_checkpoint(1, "both", error, sizeof(error)));
    CHECK(strstr(error, "reclaimable physical memory") && strstr(error, "floor=16"));
    char tiny[2] = {'x', 'x'};
    CHECK(!h3_memory_check(0, "tiny", tiny, sizeof(tiny)) && tiny[1] == 0);
    CHECK(!h3_memory_check(0, NULL, NULL, 0));
    const char *invalid[] = {"-1", "18446744073709551616", "12junk", "", " 2"};
    for (size_t i = 0; i < sizeof(invalid)/sizeof(*invalid); i++) {
        setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES", invalid[i], 1);
        CHECK(h3_memory_minimum_bytes() == H3_MIN_AVAILABLE_MEMORY);
    }
    unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
    h3_memory_snapshot s={.physical_total=UINT64_C(128)<<30,.available=UINT64_C(50)<<30,
        .available_valid=1,.process_valid=1,.system_valid=1};
    h3_memory_set_test_snapshot(copy_sample,&s);
    CHECK(h3_memory_limit_bytes(&s)==UINT64_C(110000000000));
    s.physical_footprint=H3_MAX_PROCESS_MEMORY-1;
    CHECK(h3_memory_check(0,"below cap",error,sizeof(error)));
    CHECK(!h3_memory_check(2,"future allocation",error,sizeof(error)));
    s.physical_footprint++;
    CHECK(!h3_memory_check(0,"compressed process",error,sizeof(error)));
    CHECK(strstr(error,"process footprint=110.000 GB")&&strstr(error,"memory limit=110.000 GB"));
    s.physical_footprint=UINT64_C(60)<<30;
    CHECK(!h3_memory_check_gpu(0,H3_MAX_PROCESS_MEMORY,"Metal reservation",error,sizeof(error)));
    CHECK(h3_memory_check_gpu(0,s.physical_footprint,"overlapping accounting",error,sizeof(error)));
    s.process_valid=0;CHECK(!h3_memory_check(0,"failed process query",error,sizeof(error)));
    s.process_valid=1;s.available_valid=0;CHECK(!h3_memory_check(0,"failed headroom query",error,sizeof(error)));
    s.available_valid=1;s.physical_total=UINT64_C(64)<<30;
    CHECK(h3_memory_limit_bytes(&s)==(UINT64_C(54)<<30));
    s.physical_total=UINT64_C(8)<<30;CHECK(h3_memory_limit_bytes(&s)==0);
    CHECK(!h3_memory_check(0,"insufficient RAM",error,sizeof(error)));
    s.physical_total=UINT64_C(128)<<30;
    setenv("H3_MEMORY_LIMIT_BYTES","90000000000",1);CHECK(h3_memory_limit_bytes(&s)==UINT64_C(90000000000));
    setenv("H3_MEMORY_LIMIT_BYTES","200000000000",1);CHECK(h3_memory_limit_bytes(&s)==H3_MAX_PROCESS_MEMORY);
    setenv("H3_MEMORY_LIMIT_BYTES","0",1);CHECK(h3_memory_limit_bytes(&s)==H3_MAX_PROCESS_MEMORY);
    unsetenv("H3_MEMORY_LIMIT_BYTES");
    h3_memory_set_test_snapshot(NULL,NULL);
    h3_memory_set_test_query(NULL, NULL);
    printf("ok: %d memory query, floor, reserve, overflow, failure and cancellation checks\n", checks);
    return 0;
}
