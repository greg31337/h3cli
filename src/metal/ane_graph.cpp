#include "src/metal/ane.h"
#include "third_party/vpipe-ane/ane-emitter.h"
#include <cstdio>
#include <exception>

int h3_ane_emit(const char *directory, int rows, int k, int n, int tile,
                char *error, size_t error_size) {
    try {
        vpipe::AneGraphSpec spec;
        spec.M=rows; spec.K=k; spec.N=n; spec.bk=tile; spec.bn=tile;
        spec.runtime_weights=true; spec.weights_out_in=true;
        std::string message;
        bool ok=vpipe::AneEmitter::emit(spec, {}, directory, &message);
        if(!ok && error && error_size) std::snprintf(error,error_size,"%s",message.c_str());
        return ok;
    } catch(const std::exception& e) {
        if(error && error_size) std::snprintf(error,error_size,"ANE graph: %s",e.what());
        return 0;
    }
}
