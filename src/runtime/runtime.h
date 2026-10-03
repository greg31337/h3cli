#ifndef H3_RUNTIME_H
#define H3_RUNTIME_H
#if defined(__APPLE__) && defined(H3_PACKAGE_RUNTIME)
const char *h3_runtime_shader_path(void);
#endif
/* Package-only canonical identities; ordinary native builds return NULL/input. */
#if (defined(__linux__) || defined(__APPLE__)) && defined(H3_PACKAGE_RUNTIME)
const char *h3_runtime_dependency_id(const char *environment_name);
const char *h3_runtime_environment_entry(const char *entry);
#else
static inline const char *h3_runtime_dependency_id(const char *name) {(void)name; return 0;}
static inline const char *h3_runtime_environment_entry(const char *entry) {return entry;}
#endif
#endif
