#ifndef H3_SOL_H
#define H3_SOL_H
#include "src/host.h"
#include <stdint.h>
/* Host/Metal ABI: any protected row makes the entire block exact.
 * Temporal coordinates are target-video latent frames, independent of packing. */
typedef struct { uint32_t protect; int32_t first_frame,last_frame; uint32_t rows; } h3_sol_block;
typedef struct {
    uint32_t sequence, query_blocks, key_blocks, protected_rows;
    h3_sol_block *query, *key;
} h3_sol_layout;
int h3_sol_layout_build(const h3_layout *layout,unsigned query_block,unsigned key_block,
                        h3_sol_layout *out,char *error,size_t size);
void h3_sol_layout_free(h3_sol_layout *layout);
#endif
