#ifndef H3_APPROXIMATE_H
#define H3_APPROXIMATE_H
#include "src/h3.h"
#include "src/host.h"
int h3_approximate_layout_valid(const h3_params *params,int continuation,int context,
    const h3_layout *layout,char *error,size_t size);
int h3_approximate_params_valid(const h3_params *params,int lora,char *error,size_t size);
/* Backend-independent reference policy, also used with reconstructed saved kinds. */
int h3_approximate_references_valid(const h3_params *params,char *error,size_t size);
int h3_adaptive_lower_bound(const h3_params *params,char *error,size_t size);
int h3_adaptive_preflight(int mode,char *error,size_t size);
float h3_adaptive_threshold(const h3_params *params);
int h3_adaptive_max_hits(const h3_params *params);
int h3_adaptive_controls_valid(const h3_params *params,char *error,size_t size);
int h3_adaptive_controls_match(const h3_params *requested,const h3_params *saved);
int h3_warmups_valid(const h3_params *params,char *error,size_t size);
int h3_warmups_match(const h3_params *requested,const h3_params *saved);
#endif
