#ifndef H3_CUDA_DISPATCH_H
#define H3_CUDA_DISPATCH_H
/* Architecture policy is deliberately separate from H3 model semantics. New
 * kernels enter this table only after component/continuation/resume qualification.
 * Still and low-level operator consumers retain portable implementations. */
typedef enum { H3_CUDA_PORTABLE = 0, H3_CUDA_SM120_ATTENTION, H3_CUDA_SM90_ATTENTION } h3_cuda_kernel_family;
typedef struct {
    int architecture;
    h3_cuda_kernel_family attention, linear, convolution;
    const char *name;
} h3_cuda_dispatch;
static inline h3_cuda_dispatch h3_cuda_dispatch_select(int sm, int portable_only) {
    h3_cuda_dispatch choice = {sm,H3_CUDA_PORTABLE,H3_CUDA_PORTABLE,H3_CUDA_PORTABLE,"portable"};
    if (portable_only) return choice;
    switch (sm) {
        case 90:
            choice.attention=H3_CUDA_SM90_ATTENTION;
            choice.name="SM90 fixed-head attention (portable fallback)";
            return choice;
        case 120:
            choice.attention=H3_CUDA_SM120_ATTENTION;
            choice.name="SM120 fixed-head attention (portable fallback)";
            return choice;
        case 86: case 89: return choice;
        default: return choice; /* Forward-compatible PTX primitive fallback. */
    }
}
#endif
