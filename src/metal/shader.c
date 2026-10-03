#include "src/metal/shader.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "src/platform.h"
#include "src/runtime/runtime.h"

static char *join(const char *directory, const char *name) {
    size_t a = strlen(directory), b = strlen(name);
    if (b > SIZE_MAX - a - 2) return NULL;
    char *path = malloc(a + b + 2);
    if (path) snprintf(path, a + b + 2, "%s/%s", directory, name);
    return path;
}

/* A present but unreadable CWD file is an error, not permission to select a
 * different shader. Keep lookup and compilation failures distinct. */
static int readable(const char *path) {
    struct stat status;
    if (stat(path, &status)) return 0;
    if (!S_ISREG(status.st_mode)) { errno = EINVAL; return 0; }
    return access(path, R_OK) == 0;
}


char *h3_shader_resolve(const char *supplied, char *error, size_t error_size) {
    if (error && error_size) *error = '\0';
#if defined(__APPLE__) && defined(H3_PACKAGE_RUNTIME)
    (void)supplied;
    const char *fixed = h3_runtime_shader_path();
    if (fixed && readable(fixed)) return strdup(fixed);
    if (error && error_size) snprintf(error, error_size, "packaged shader is unavailable");
    return NULL;
#endif
    const char *override = getenv("H3_SHADER_PATH");
    const char *name = override ? override :
                       supplied ? supplied : "src/metal/shaders.metal";
    if (override || !*name || *name == '/') {
        if (*name && readable(name)) {
            char *result = strdup(name);
            if (result) return result;
            errno = ENOMEM;
        } else if (!*name) errno = EINVAL;
        if (error && error_size)
            snprintf(error, error_size, "cannot load shader %s%s: %s; "
                     "provide a readable shader file%s",
                     override ? "H3_SHADER_PATH=" : "", name, strerror(errno),
                     override ? " (explicit override; no fallback)" : "");
        return NULL;
    }
    char *cwd = getcwd(NULL, 0);
    char *local = cwd ? join(cwd, name) : NULL;
    free(cwd);
    if (!local) goto memory_error;
    if (readable(local)) return local;
    int local_error = errno;
    if (local_error != ENOENT && local_error != ENOTDIR) {
        if (error && error_size)
            snprintf(error, error_size, "cannot load shader %s; attempted CWD %s: %s",
                     name, local, strerror(local_error));
        free(local);
        return NULL;
    }
    char *exe = h3_executable_path();
    char *nearby = NULL;
    char *installed = NULL;
    if (exe) {
        char *slash = strrchr(exe, '/');
        if (slash) {
            *slash = '\0'; nearby = join(exe, name);
            slash = strrchr(exe, '/');
            if (slash && !strcmp(slash + 1, "bin")) {
                *slash = '\0'; installed = join(exe, name);
            }
        }
        free(exe);
    }
    if (nearby && readable(nearby)) {
        char *canonical = realpath(nearby, NULL);
        free(local);
        free(installed);
        if (canonical) { free(nearby); return canonical; }
        return nearby;
    }
    int nearby_error = errno;
    if (installed && (nearby_error == ENOENT || nearby_error == ENOTDIR) && readable(installed)) {
        char *canonical = realpath(installed, NULL);
        free(local);
        free(nearby);
        if (canonical) { free(installed); return canonical; }
        return installed;
    }
    if (error && error_size)
        snprintf(error, error_size, "cannot load shader %s; attempted CWD %s (%s), "
                 "executable-relative path %s (%s), installation path %s; preserve the shader's relative path "
                 "or set H3_SHADER_PATH",
                 name, local, strerror(local_error), nearby ? nearby : "<unavailable>",
                 nearby ? strerror(nearby_error) : "cannot discover executable",
                 installed ? installed : "<not a bin/ installation>");
    free(local);
    free(nearby);
    free(installed);
    return NULL;
memory_error:
    if (error && error_size)
        snprintf(error, error_size, "cannot resolve shader %s: cannot obtain CWD or allocate path", name);
    return NULL;
}
