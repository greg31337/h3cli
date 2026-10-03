#ifndef H3_DEVICE_H
#define H3_DEVICE_H
#include "src/h3.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Device selection is process-wide and must precede context creation. */
int h3_device_query(h3_device_info *info, char *error, size_t error_size);
int h3_device_configure(const char *index, const char *weight_mode,
                        char *error, size_t error_size);
/* Whether retaining optional GPU caches leaves this much room for the next
 * component. Unified-memory Metal preserves its existing cache policy. */
int h3_device_memory_fits(uint64_t next_component_bytes);
#ifdef __cplusplus
}
#endif
#endif
