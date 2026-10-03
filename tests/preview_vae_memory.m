#import <Metal/Metal.h>
#include <stdint.h>
uint64_t h3_test_metal_allocated(void) {
    @autoreleasepool { return MTLCreateSystemDefaultDevice().currentAllocatedSize; }
}
