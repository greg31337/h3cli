#include "src/geometry.h"
#include "src/cli/render.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *headings[] = {
    "subject_definitions:", "summary:", "retention_analysis:",
    "detailed_description:", "overall_soundscape:", "non_diegetic_music:"
};

static void *allocate(size_t size) {
    void *p = calloc(1, size);
    if (!p) { fputs("h3cli: render: out of memory\n", stderr); exit(1); }
    return p;
}
static char *copy(const char *s) {
    char *p = allocate(strlen(s) + 1);
    strcpy(p, s);
    return p;
}
char *h3_render_format(const char *format, ...) {
    char *s = NULL;
    va_list args;
    va_start(args, format);
    int n = vasprintf(&s, format, args);
    va_end(args);
    if (n < 0) { fputs("h3cli: render: out of memory\n", stderr); exit(1); }
    return s;
}
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n-1])) s[--n] = 0;
    return s;
}
static void append(char **body, const char *text) {
    char *s = h3_render_format("%s%s\n", *body ? *body : "", text);
    free(*body); *body = s;
}
static void strip(char **body) {
    if (*body) {
        char *s = copy(trim(*body));
        free(*body); *body = s;
    }
}
char *h3_render_absolute(const char *base, const char *value) {
    char *expanded;
    if (value[0] == '~' && (!value[1] || value[1] == '/')) {
        const char *home_dir = getenv("HOME");
        expanded = h3_render_format("%s%s", home_dir ? home_dir : "~", value + 1);
    } else expanded = copy(value);
    char *absolute = expanded[0] == '/' ? copy(expanded) : h3_render_format("%s/%s", base, expanded);
    free(expanded);
    /* Lexically normalize paths even when the destination does not exist. */
    char *result = allocate(strlen(absolute) + 2), *save = NULL;
    strcpy(result, "/");
    for (char *part = strtok_r(absolute, "/", &save); part; part = strtok_r(NULL, "/", &save)) {
        if (!strcmp(part, ".")) continue;
        if (!strcmp(part, "..")) {
            char *slash = strrchr(result, '/');
            if (slash != result) *slash = 0; else result[1] = 0;
        } else {
            if (result[1]) strcat(result, "/");
            strcat(result, part);
        }
    }
    free(absolute);
    return result;
}
static char *stem(const char *name) {
    char *s = copy(name), *dot = strrchr(s, '.');
    if (dot && dot != s) *dot = 0;
    return s;
}
static int utf8(const unsigned char *s, size_t length) {
    for (size_t i = 0; i < length; ) {
        unsigned c = s[i++], code, minimum; size_t trailing;
        if (c < 128) continue;
        if (c >= 0xc2 && c <= 0xdf) { code = c & 31; trailing = 1; minimum = 128; }
        else if (c >= 0xe0 && c <= 0xef) { code = c & 15; trailing = 2; minimum = 2048; }
        else if (c >= 0xf0 && c <= 0xf4) { code = c & 7; trailing = 3; minimum = 65536; }
        else return 0;
        if (trailing > length - i) return 0;
        while (trailing--) {
            c = s[i++]; if ((c & 0xc0) != 0x80) return 0;
            code = (code << 6) | (c & 63);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return 0;
    }
    return 1;
}
char *h3_render_chain_name(const h3_render_segment *s) {
    char *base = stem(s->name), *name = h3_render_format("%s-merged.mp4", base);
    free(base); return name;
}
static int forbidden(const char *name) {
    static const char *names[] = {
        "prompt", "output", "continue-from", "save-av-state", "keep-continuation-prefix",
        "help", "info", "server", "still", "state-only", "decode-av-state",
        "decode-still-latent", "save-still-latent", "resume-sampler-state",
        "stop-after-step", "preview-on-stop", "upscale-state", "inspect-upscale-state",
        "download-models", "list-models", "render", "output-dir", "dry-run"
    };
    for (size_t i = 0; i < sizeof(names)/sizeof(*names); i++)
        if (!strcmp(name, names[i])) return 1;
    return 0;
}
static const char *canonical(const char *s) {
    while (*s == '-') s++;
    if (!strcmp(s, "refimage")) return "ref-image";
    if (!strcmp(s, "refvideo")) return "ref-video";
    if (!strcmp(s, "refaudio")) return "ref-audio";
    if (!strcmp(s, "d")) return "model-dir";
    if (!strcmp(s, "p")) return "prompt";
    if (!strcmp(s, "o")) return "output";
    return s;
}
/* Shell-style quoting only; no expansion or shell execution. */
static char **tokens(const char *s, int *count, const char **arguments, char *error, size_t size) {
    char **v = allocate(1026 * sizeof(*v));
    char *token = allocate(strlen(s) + 1);
    *count = 0;
    if (arguments) *arguments = NULL;
    while (*s) {
        while (isspace((unsigned char)*s)) s++;
        if (!*s || *s == '#') break;
        size_t n = 0; char quote = 0;
        while (*s && (quote || (!isspace((unsigned char)*s) && *s != '#'))) {
            char c = *s++;
            /* Keep the raw argument spelling for quote-aware reference lists,
             * including directives written as #ref-image="a,b.png",c.png. */
            if (arguments && !*count && !*arguments && c == '=' && !quote) *arguments = s;
            if ((c == '\'' || c == '"') && (!quote || quote == c)) {
                quote = quote ? 0 : c; continue;
            }
            if (c == '\\' && quote != '\'') {
                if (!*s) goto malformed;
                if (quote == '"' && *s != '"' && *s != '\\' && *s != '$' && *s != '`')
                    token[n++] = c;
                c = *s++;
            }
            token[n++] = c;
        }
        if (quote) goto malformed;
        if (*count >= 1024) { snprintf(error, size, "too many directive tokens"); goto fail; }
        if (arguments && !*count && !*arguments) *arguments = s;
        token[n] = 0; v[(*count)++] = copy(token);
        if (*s == '#') break;
    }
    free(token); return v;
malformed:
    snprintf(error, size, "unterminated quotation or escape in directive");
fail:
    free(token); h3_argv_free(v); return NULL;
}
static int clone_settings(h3_request *to, const h3_request *from, int references,
                          char *error, size_t size) {
    for (size_t i = 0; i < from->count; i++) {
        const h3_option_value *v = &from->values[i];
        if (!references && v->option->path == H3_PATH_MEDIA) continue;
        if (!h3_request_add(to, v->option->name, v->value, v->second, v->source, error, size)) return 0;
    }
    return 1;
}
static char *option_path(const h3_option_descriptor *o, const char *base, const char *s) {
    if (!s) return NULL;
    if (!strcmp(o->name, "lora")) {
        char *p = copy(s), *colon = strrchr(p, ':');
        double scale;
        if (colon && lj_number(colon + 1, &scale)) *colon = 0;
        else colon = NULL;
        char *absolute = h3_render_absolute(base, p);
        char *result = colon ? h3_render_format("%s:%s", absolute, colon + 1) : copy(absolute);
        free(p); free(absolute); return result;
    }
    return o->path != H3_PATH_NONE || !strcmp(o->name, "models-path") ?
        h3_render_absolute(base, s) : copy(s);
}
static int set_option(h3_request *r, const char *name, const char *v, const char *second,
                      const char *base, unsigned char *touched, char *error, size_t size) {
    const h3_option_descriptor *o = h3_option_find(name);
    if (forbidden(name)) {
        snprintf(error, size, "--%s is managed by --render or does not render a complete video", name); return 0;
    }
    if (!o) { snprintf(error, size, "unknown directive/option --%s", name); return 0; }
    if (!h3_option_validate(o, v, error, size)) return 0;
    static const char *groups[][4] = {
        {"frames", "seconds", NULL}, {"preview-vae", "no-preview-vae", NULL},
        {"ffmpeg-crf", "output-quality", "lossless-video", NULL}
    };
    for (size_t i = 0; i < 3; i++) {
        int matches = 0;
        for (size_t j = 0; groups[i][j]; j++) matches |= !strcmp(groups[i][j], name);
        if (matches) for (size_t j = 0; groups[i][j]; j++)
            if (strcmp(groups[i][j], name)) h3_request_remove(r, groups[i][j]);
    }
    size_t index = (size_t)(o - h3_cli_options);
    if (!o->repeatable || !touched[index]) h3_request_remove(r, name);
    touched[index] = 1;
    char *value = option_path(o, base, v), *other = option_path(o, base, second);
    int ok = h3_request_add(r, name, value, other, H3_SOURCE_NATIVE, error, size);
    free(value); free(other); return ok;
}
static const char *value(const h3_render_segment *s, const char *name) {
    const h3_option_value *v = h3_request_get(&s->settings, name);
    return v ? v->value : h3_option_find(name)->default_value;
}
static int integer(const h3_render_segment *s, const char *name) { return atoi(value(s, name)); }
static int has(const h3_render_segment *s, const char *name) { return h3_request_get(&s->settings, name) != NULL; }
static int reference_mode(const h3_render_segment *s) {
    for (size_t i = 0; i < s->settings.count; i++)
        if (!strncmp(s->settings.values[i].option->name, "ref-", 4) &&
            s->settings.values[i].option->path == H3_PATH_MEDIA) return 1;
    return 0;
}
static int seed_set(h3_render_plan *p, const char *text, char *error, size_t size) {
    if (p->seed_count) { snprintf(error, size, "use #seed-set only once"); return 0; }
    char *list = copy(text), *next = list;
    int ok = 0;
    for (;;) {
        char *item = next, *comma = strchr(item, ',');
        if (comma) { *comma = 0; next = comma + 1; }
        item = trim(item);
        if (!h3_option_validate(h3_option_find("seed"), item, error, size)) {
            snprintf(error, size, "#seed-set requires comma-separated unsigned 64-bit integers"); break;
        }
        uint64_t offset = (uint64_t)strtoull(item, NULL, 10);
        for (size_t i = 0; i < p->seed_count; i++) if (p->seed_set[i] == offset) {
            snprintf(error, size, "duplicate #seed-set offset: %" PRIu64, offset); goto done;
        }
        if (p->seed_count == 1024) { snprintf(error, size, "#seed-set supports at most 1024 offsets"); break; }
        uint64_t *grown = realloc(p->seed_set, (p->seed_count + 1) * sizeof(*grown));
        if (!grown) { snprintf(error, size, "out of memory"); break; }
        p->seed_set = grown; p->seed_set[p->seed_count++] = offset;
        if (!comma) { ok = 1; break; }
    }
done:
    free(list); return ok;
}
static int add_reference(h3_render_plan *p, const char *alias, const char *path,
                         const char *base, char *error, size_t size) {
    if (!alias || !*alias || !path || !*path) goto invalid;
    for (const unsigned char *q = (const unsigned char *)alias; *q; q++)
        if (*q <= 32 || *q == 127 || *q == '/' || *q == '\\' || *q == '#') goto invalid;
    for (size_t i = 0; i < p->reference_count; i++) if (!strcmp(alias, p->references[i].name)) {
        snprintf(error, size, "duplicate #ref alias: %s", alias); return 0;
    }
    h3_render_reference *grown = realloc(p->references, (p->reference_count + 1) * sizeof(*grown));
    if (!grown) { snprintf(error, size, "out of memory"); return 0; }
    p->references = grown;
    p->references[p->reference_count++] = (h3_render_reference){copy(alias), h3_render_absolute(base, path)};
    return 1;
invalid:
    snprintf(error, size, "use #ref ALIAS FILE_PATH; alias must be a nonempty name without whitespace, # or directories"); return 0;
}
static const char *reference_path(const h3_render_plan *p, const char *path) {
    if (path) for (size_t i = 0; i < p->reference_count; i++)
        if (!strcmp(path, p->references[i].name)) return p->references[i].path;
    return path;
}
static int reference_option(const h3_option_descriptor *o) {
    return o && o->path == H3_PATH_MEDIA && o->repeatable && !strncmp(o->name, "ref-", 4);
}
/* Commas separate entries, not quoted/escaped filename characters. Paired
 * references retain their VIDEO AUDIO arguments within each entry. */
static int set_references(const h3_render_plan *p, h3_request *settings,
                          const h3_option_descriptor *o, const char *arguments,
                          const char *base, unsigned char *touched, char *error, size_t size) {
    char *list = copy(arguments), *start = list, quote = 0;
    int ok = 0;
    for (char *end = list; ; end++) {
        char c = *end;
        if (c == '\\' && quote != '\'' && end[1]) { end++; continue; }
        if ((c == '\'' || c == '"') && (!quote || quote == c)) {
            quote = quote ? 0 : c; continue;
        }
        if (c && (quote || (c != ',' && c != '#'))) continue;
        *end = 0;
        int count = 0;
        char **v = tokens(start, &count, NULL, error, size);
        if (!v) break;
        if (count != o->arity) {
            snprintf(error, size, "#%s expects %d %s per comma-separated entry", o->name,
                     o->arity, o->arity == 1 ? "path" : "paths (VIDEO AUDIO)");
            h3_argv_free(v); break;
        }
        int added = set_option(settings, o->name, reference_path(p, v[0]),
                               o->arity == 2 ? reference_path(p, v[1]) : NULL,
                               base, touched, error, size);
        h3_argv_free(v);
        if (!added) break;
        if (c != ',') { ok = 1; break; }
        start = end + 1;
    }
    free(list); return ok;
}
static void finish_prompt(h3_render_plan *p, unsigned seen) {
    h3_render_segment *s = &p->segments[p->count - 1];
    strip(&s->preamble);
    s->prompt = copy(s->preamble ? s->preamble : "");
    for (size_t j = 0; j < 6; j++) {
        if (!(seen & (1u << j)) && s->continuing && p->count > 1 &&
            p->segments[p->count - 2].sections[j])
            s->sections[j] = copy(p->segments[p->count - 2].sections[j]);
        strip(&s->sections[j]);
        if (!s->sections[j] || !*s->sections[j]) continue;
        char *next = h3_render_format("%s%s%s\n%s", s->prompt, *s->prompt ? "\n\n" : "",
                                      headings[j], s->sections[j]);
        free(s->prompt); s->prompt = next;
    }
}
static int parse(h3_render_plan *p, const h3_request *defaults, const char *base,
                 char *error, size_t size) {
    char *text = copy(p->source_text), *cursor = text;
    if (!strncmp(cursor, "\xef\xbb\xbf", 3)) cursor += 3;
    unsigned char *touched = allocate(h3_cli_option_count);
    h3_request global_references = {0};
    unsigned seen = 0; int active_heading = -1, prompt_started = 0, number = 0, ok = 0;
    h3_render_segment *s = NULL;
    while (*cursor) {
        number++;
        char *line = cursor, *end = strchr(cursor, '\n');
        if (end) { *end = 0; cursor = end + 1; } else cursor += strlen(cursor);
        size_t len = strlen(line);
        if (len && line[len-1] == '\r') line[len-1] = 0;
        char *clean_copy = copy(line), *clean = trim(clean_copy);
        if (!strncmp(clean, "###", 3) || !strncmp(clean, "!###", 4)) {
            if (s) finish_prompt(p, seen);
            int skipped = clean[0] == '!';
            char *label = trim(clean + (skipped ? 4 : 3));
            len = strlen(label);
            if (len > 3 && !strcmp(label + len - 3, "###") && isspace((unsigned char)label[len-4])) {
                label[len-3] = 0; label = trim(label);
            }
            int valid = *label && strcmp(label, ".") && strcmp(label, "..");
            for (const unsigned char *q = (unsigned char *)label; *q; q++)
                if (*q < 32 || *q == 127 || *q == '/' || *q == '\\') valid = 0;
            if (!valid || strlen(label) > 236) {
                snprintf(error, size, "use a filename after ###, without directories or control characters");
                free(clean_copy); goto done;
            }
            h3_render_segment *grown = realloc(p->segments, (p->count + 1) * sizeof(*grown));
            if (!grown) { free(clean_copy); snprintf(error, size, "out of memory"); goto done; }
            p->segments = grown; s = &p->segments[p->count]; memset(s, 0, sizeof(*s));
            s->line = number; s->skipped = skipped;
            s->label = copy(label);
            len = strlen(label);
            s->name = h3_render_format("%s%s", label, len >= 4 && !strcasecmp(label + len - 4, ".mp4") ? "" : ".mp4");
            const h3_request *previous = p->count ? &p->segments[p->count-1].settings : defaults;
            p->count++;
            if (!clone_settings(&s->settings, previous, p->count == 1, error, size)) { free(clean_copy); goto done; }
            memset(touched, 0, h3_cli_option_count);
            for (size_t i = 0; i < global_references.count; i++) {
                const h3_option_value *v = &global_references.values[i];
                if (!set_option(&s->settings, v->option->name, v->value, v->second,
                                base, touched, error, size)) { free(clean_copy); goto done; }
            }
            seen = 0; active_heading = -1; prompt_started = 0;
        } else if (!strncmp(clean, "##", 2) || (!*clean && !prompt_started)) {
            /* Comments do not become prompt text. */
        } else if (*clean == '#') {
            int count = 0; const char *arguments = NULL;
            char **v = tokens(clean + 1, &count, &arguments, error, size);
            if (!v) { free(clean_copy); goto done; }
            if (!count) { h3_argv_free(v); free(clean_copy); continue; }
            char *equals = strchr(v[0], '=');
            const char *inline_value = NULL;
            if (equals) { *equals = 0; inline_value = equals + 1; }
            const char *name = canonical(v[0]);
            const h3_option_descriptor *option = h3_option_find(name);
            int arity = count - 1 + (inline_value != NULL);
            const char *first = inline_value ? inline_value : count > 1 ? v[1] : NULL;
            const char *second = inline_value ? (count > 1 ? v[1] : NULL) : (count > 2 ? v[2] : NULL);
            int good = 1;
            if (!strcmp(name, "nomerge")) {
                if (arity) { snprintf(error, size, "#nomerge takes no arguments"); good = 0; }
                else p->nomerge = 1;
            } else if (!strcmp(name, "overlay-segment-name") || !strcmp(name, "seed-set") || !strcmp(name, "ref")) {
                if (s) { snprintf(error, size, "#%s belongs before the first segment definition", name); good = 0; }
                else if (!strcmp(name, "overlay-segment-name")) {
                    if (arity) { snprintf(error, size, "#overlay-segment-name takes no arguments"); good = 0; }
                    else p->overlay_segment_name = 1;
                } else if (!strcmp(name, "ref")) {
                    if (arity != 2) { snprintf(error, size, "use #ref ALIAS FILE_PATH"); good = 0; }
                    else good = add_reference(p, first, second, base, error, size);
                } else {
                    char *list = copy(inline_value ? inline_value : "");
                    for (int j = 1; j < count; j++) {
                        char *joined = h3_render_format("%s%s%s", list, *list ? " " : "", v[j]);
                        free(list); list = joined;
                    }
                    good = seed_set(p, list, error, size); free(list);
                }
            } else if (reference_option(option) && (!s || !prompt_started)) {
                good = set_references(p, s ? &s->settings : &global_references, option,
                                      arguments, base, touched, error, size);
            } else if (!s) { snprintf(error, size, "start the first video with ### NAME or !### NAME"); good = 0; }
            else if (prompt_started) { snprintf(error, size, "put directives before the prompt; use \\# for a literal hash"); good = 0; }
            else if (!strcmp(name, "continue")) {
                if (arity || s->continuing) { snprintf(error, size, "use #continue once, with no value"); good = 0; }
                else s->continuing = 1;
            } else if (!strcmp(name, "size")) {
                char *dimensions = arity == 1 ? copy(first) : copy("");
                char *separator = strpbrk(dimensions, "xX");
                if (separator) *separator++ = 0;
                if (!separator || !h3_option_validate(h3_option_find("width"), dimensions, error, size) ||
                    !h3_option_validate(h3_option_find("height"), separator, error, size)) {
                    snprintf(error, size, "use #size WIDTHxHEIGHT"); good = 0;
                } else {
                    good = set_option(&s->settings, "width", dimensions, NULL, base, touched, error, size) &&
                           set_option(&s->settings, "height", separator, NULL, base, touched, error, size);
                }
                free(dimensions);
            } else if (!strcmp(name, "unset")) {
                const char *target = arity == 1 ? canonical(first) : "";
                const h3_option_descriptor *o = h3_option_find(target);
                if (!strcmp(target, "size")) {
                    h3_request_remove(&s->settings, "width"); h3_request_remove(&s->settings, "height");
                } else if (!o || forbidden(target)) { snprintf(error, size, "cannot unset #%s", target); good = 0; }
                else { h3_request_remove(&s->settings, target); touched[o - h3_cli_options] = 0; }
            } else {
                const h3_option_descriptor *o = h3_option_find(name);
                if (o && o->arity != arity) { snprintf(error, size, "#%s expects %d values", name, o->arity); good = 0; }
                else {
                    if (o && o->path == H3_PATH_MEDIA) {
                        first = reference_path(p, first); second = reference_path(p, second);
                    }
                    good = set_option(&s->settings, name, first, second, base, touched, error, size);
                }
            }
            h3_argv_free(v);
            if (!good) { free(clean_copy); goto done; }
        } else if (!s) {
            snprintf(error, size, "start the first video with ### NAME or !### NAME"); free(clean_copy); goto done;
        } else {
            int heading = -1;
            for (int j = 0; j < 6; j++) if (!strcmp(clean, headings[j])) heading = j;
            if (heading >= 0) {
                if (seen & (1u << heading)) { snprintf(error, size, "duplicate prompt heading %s", headings[heading]); free(clean_copy); goto done; }
                seen |= 1u << heading; active_heading = heading;
            } else {
                char *literal = line;
                while (isspace((unsigned char)*literal)) literal++;
                if (literal[0] == '\\' && literal[1] == '#') memmove(literal, literal + 1, strlen(literal));
                append(active_heading < 0 ? &s->preamble : &s->sections[active_heading], line);
            }
            prompt_started = 1;
        }
        free(clean_copy);
    }
    if (!s) snprintf(error, size, "no video sections found");
    else { finish_prompt(p, seen); ok = 1; }
done:
    if (!ok) {
        char *message = copy(error);
        snprintf(error, size, "%s:%d: %s", p->source, number, message); free(message);
    }
    h3_request_free(&global_references);
    free(touched); free(text); return ok;
}
static void free_segment(h3_render_segment *s) {
    free(s->name); free(s->label); free(s->prompt); free(s->preamble);
    for (size_t j = 0; j < 6; j++) free(s->sections[j]);
    h3_request_free(&s->settings);
}
/* Expand from the unmodified settings, so inherited seeds never accumulate
 * offsets and every sequence starts with its own independent first segment. */
static int expand_seeds(h3_render_plan *p, char *error, size_t size) {
    if (!p->seed_count) return 1;
    if (p->count > SIZE_MAX / p->seed_count / sizeof(*p->segments)) {
        snprintf(error, size, "seed-set render plan is too large"); return 0;
    }
    size_t count = p->count * p->seed_count;
    h3_render_segment *expanded = allocate(count * sizeof(*expanded));
    for (size_t run = 0; run < p->seed_count; run++) for (size_t i = 0; i < p->count; i++) {
        const h3_render_segment *s = &p->segments[i];
        uint64_t seed = (uint64_t)strtoull(value(s, "seed"), NULL, 10), offset = p->seed_set[run];
        if (seed > UINT64_MAX - offset) {
            snprintf(error, size, "%s:%d: seed %" PRIu64 " + offset %" PRIu64 " exceeds unsigned 64-bit range",
                     p->source, s->line, seed, offset); goto fail;
        }
        h3_render_segment *to = &expanded[run * p->count + i];
        to->line = s->line; to->continuing = s->continuing; to->skipped = s->skipped;
        to->frames = s->frames; to->delivered = s->delivered; to->width = s->width; to->height = s->height;
        to->seed_offset = offset;
        to->name = h3_render_format("seed-%" PRIu64 "/%s", offset, s->name);
        to->label = copy(s->label); to->prompt = copy(s->prompt);
        if (!clone_settings(&to->settings, &s->settings, 1, error, size)) goto fail;
        h3_request_remove(&to->settings, "seed");
        char decimal[32]; snprintf(decimal, sizeof(decimal), "%" PRIu64, seed + offset);
        if (!h3_request_add(&to->settings, "seed", decimal, NULL, H3_SOURCE_NATIVE, error, size)) goto fail;
    }
    for (size_t i = 0; i < p->count; i++) free_segment(&p->segments[i]);
    free(p->segments); p->segments = expanded; p->count = count; return 1;
fail:
    for (size_t i = 0; i < count; i++) free_segment(&expanded[i]);
    free(expanded); return 0;
}
static int validate(h3_render_plan *p, char *error, size_t size) {
    for (size_t i = 0; i < p->count; i++) {
        h3_render_segment *s = &p->segments[i];
#define FAIL(...) do { snprintf(error, size, __VA_ARGS__); goto invalid; } while (0)
        if (s->continuing && !i) FAIL("#continue needs a preceding segment");
        if (s->continuing && s->skipped != p->segments[i-1].skipped)
            FAIL("continuation chain mixes skipped (!###) and rendered (###) segments");
        if (s->skipped) continue;
        if (!s->prompt || !*s->prompt) FAIL("empty prompt after inheritance");
        s->width = integer(s, "width"); s->height = integer(s, "height");
        if (h3_canvas_native_profile(s->width,s->height,H3_NATIVE_CUDA)<0) FAIL("size must use multiples of 32 within the native pixel limit");
        int rw = integer(s, "render-width"), rh = integer(s, "render-height");
        if ((rw || rh) && (rw < 32 || rh < 32 || rw % 32 || rh % 32 || rw > s->width || rh > s->height ||
            (int64_t)rw * s->height != (int64_t)rh * s->width)) FAIL("internal render size must fit and match the output aspect ratio");
        double requested = integer(s, "frames");
        if (has(s, "seconds")) requested = round(strtod(value(s, "seconds"), NULL) * 24.0);
        if (requested < 1 || requested > 362) FAIL("frame count must align within the native 22..362 generation range");
        int frames = (int)requested; if (frames < 5) frames = 5;
        s->frames = frames + (17 - (frames - 5) % 17) % 17; s->delivered = s->frames;
        if (s->frames < 22) FAIL("frame count must align within the native 22..362 generation range");
        for (size_t j = 0; j < s->settings.count; j++) {
            const h3_option_value *v = &s->settings.values[j];
            if (v->option->path != H3_PATH_MEDIA) continue;
            struct stat st;
            if (stat(v->value, &st) || !S_ISREG(st.st_mode) ||
                (v->second && (stat(v->second, &st) || !S_ISREG(st.st_mode)))) FAIL("missing #%s file: %s", v->option->name, v->value);
        }
        if (s->continuing) {
            h3_render_segment *previous = &p->segments[i-1];
            int context = integer(s, "continue-context");
            if (context < 39 || (context - 39) % 51 || context >= s->frames || context > previous->frames)
                FAIL("context must be 39 + 51*k, fit the previous state, and leave new frames");
            s->delivered -= context;
            const char *same[] = {"width", "height", "render-width", "render-height", "model-dir", "models-path"};
            for (size_t j = 0; j < sizeof(same)/sizeof(*same); j++)
                if (strcmp(value(s, same[j]), value(previous, same[j]))) FAIL("#%s changes within a continuation chain", same[j]);
            if (reference_mode(s) != reference_mode(previous)) FAIL("continuation cannot switch between Ref2VA and FL2VA; repeat references");
            if (has(s, "first-frame") || has(s, "last-frame")) FAIL("continuation does not support first/last-frame anchors");
            if (integer(s, "layers") != 50 || has(s, "token-reduction")) FAIL("continuation requires 50 layers and token reduction off");
            int core = integer(s, "core-reuse"), bridge = !strcmp(value(s, "continue-mode"), "bridge");
            if (core != 1 && !(bridge && (core == 4 || core == 6))) FAIL("unsupported core-reuse for continuation");
        }
        continue;
invalid: {
            char *message = copy(error);
            snprintf(error, size, "%s:%d: %s: %s", p->source, s->line, s->name, message); free(message); return 0;
        }
#undef FAIL
    }
    sj_value *names = sj_array();
    for (size_t i = 0; i < p->count; i++) {
        if (p->segments[i].skipped) continue;
        sj_add(names, NULL, sj_string(p->segments[i].name));
        if (!p->nomerge && !p->segments[i].continuing && i+1 < p->count && p->segments[i+1].continuing) {
            char *name = h3_render_chain_name(&p->segments[i]); sj_add(names, NULL, sj_string(name)); free(name);
        }
    }
    if (names->count && !p->nomerge) sj_add(names, NULL, sj_string(p->merged));
    for (size_t i = 0; i < names->count; i++) for (size_t j = 0; j < i; j++)
        if (!strcasecmp(sj_text(names->items[i]), sj_text(names->items[j]))) {
            snprintf(error, size, "output filename collides with another output: %s", sj_text(names->items[i]));
            sj_free(names); return 0;
        }
    sj_free(names); return 1;
}
int h3_render_plan_read(h3_render_plan *p, const h3_request *request, char *error, size_t size) {
    h3_request defaults = {0};
    char *cwd = getcwd(NULL, 0), *base = NULL;
    unsigned char *touched = allocate(h3_cli_option_count);
    int ok = 0;
    const h3_option_value *input = h3_request_get(request, "render"), *output = h3_request_get(request, "output-dir");
    if (!cwd || !input || !input->value || !*input->value || (output && !*output->value)) {
        snprintf(error, size, "--render requires a filename; --output-dir must be nonempty"); goto done;
    }
    char *absolute = h3_render_absolute(cwd, input->value);
    p->source = realpath(absolute, NULL); free(absolute);
    if (!p->source) { snprintf(error, size, "cannot read render script: %s", strerror(errno)); goto done; }
    base = copy(p->source); *strrchr(base, '/') = 0;
    if (!*base) { free(base); base = copy("/"); }
    char *name = stem(strrchr(p->source, '/') + 1);
    p->merged = h3_render_format("%s.mp4", name);
    p->output = output ? h3_render_absolute(cwd, output->value) : h3_render_format("%s/%s-render", base, name);
    p->work = h3_render_format("%s/.render", p->output); free(name);
    p->dry_run = h3_request_get(request, "dry-run") != NULL;
    for (size_t i = 0; i < request->count; i++) {
        const h3_option_value *v = &request->values[i];
        if (!strcmp(v->option->name, "render") || !strcmp(v->option->name, "output-dir") || !strcmp(v->option->name, "dry-run")) continue;
        if (!set_option(&defaults, v->option->name, v->value, v->second, cwd, touched, error, size)) goto done;
    }
    FILE *file = fopen(p->source, "rb");
    if (!file) { snprintf(error, size, "cannot open script: %s", strerror(errno)); goto done; }
    struct stat st;
    if (fstat(fileno(file), &st) || !S_ISREG(st.st_mode) || st.st_size < 0 || st.st_size > 64*1024*1024) {
        fclose(file); snprintf(error, size, "script must be a regular text file of at most 64 MiB"); goto done;
    }
    size_t length = (size_t)st.st_size;
    p->source_text = allocate(length + 1);
    int read_ok = fread(p->source_text, 1, length, file) == length && !ferror(file);
    if (fclose(file)) read_ok = 0;
    if (!read_ok || memchr(p->source_text, 0, length)) { snprintf(error, size, "cannot read text script or embedded NUL byte"); goto done; }
    if (!utf8((const unsigned char *)p->source_text, length)) { snprintf(error, size, "render script must be UTF-8 text"); goto done; }
    ok = parse(p, &defaults, base, error, size) && validate(p, error, size) && expand_seeds(p, error, size);
done:
    free(cwd); free(base); free(touched); h3_request_free(&defaults); return ok;
}
void h3_render_plan_free(h3_render_plan *p) {
    for (size_t i = 0; i < p->count; i++) {
        free_segment(&p->segments[i]);
    }
    for (size_t i = 0; i < p->reference_count; i++) { free(p->references[i].name); free(p->references[i].path); }
    free(p->references); free(p->seed_set);
    free(p->segments); free(p->source); free(p->source_text); free(p->output); free(p->work); free(p->merged);
    memset(p, 0, sizeof(*p));
}
int h3_render_request(const h3_render_plan *p, size_t index, h3_request *request, char *error, size_t size) {
    const h3_render_segment *s = &p->segments[index];
    if (!clone_settings(request, &s->settings, 1, error, size)) return 0;
    if (!s->continuing) {
        for (size_t i = 0; i < h3_cli_option_count; i++)
            if (!strncmp(h3_cli_options[i].name, "continue-", 9)) h3_request_remove(request, h3_cli_options[i].name);
    }
    char *state = h3_render_format("%s/%04zu.h3av", p->work, index);
    char *output = h3_render_format("%s/%04zu.mp4", p->work, index);
    int ok = h3_request_add(request, "prompt", s->prompt, NULL, H3_SOURCE_MANAGED, error, size) &&
             h3_request_add(request, "save-av-state", state, NULL, H3_SOURCE_MANAGED, error, size) &&
             h3_request_add(request, "output", output, NULL, H3_SOURCE_MANAGED, error, size);
    if (s->continuing && ok) {
        char *previous = h3_render_format("%s/%04zu.h3av", p->work, index-1);
        ok = h3_request_add(request, "continue-from", previous, NULL, H3_SOURCE_MANAGED, error, size); free(previous);
    }
    free(state); free(output); return ok;
}
sj_value *h3_render_plan_json(const h3_render_plan *p, const char *binary) {
    sj_value *report = sj_object(), *segments = sj_array(), *merges = sj_array();
    sj_add(report, "status", sj_string("planned")); sj_add(report, "input", sj_string(p->source));
    sj_add(report, "output", sj_string(p->output)); sj_add(report, "nomerge", sj_bool(p->nomerge));
    sj_add(report, "overlay_segment_name", sj_bool(p->overlay_segment_name));
    sj_value *seeds = sj_array(), *aliases = sj_object();
    for (size_t i = 0; i < p->seed_count; i++) sj_add(seeds, NULL, sj_uint(p->seed_set[i]));
    for (size_t i = 0; i < p->reference_count; i++) sj_add(aliases, p->references[i].name, sj_string(p->references[i].path));
    sj_add(report, "seed_set", seeds); sj_add(report, "references", aliases);
    sj_add(report, "segments", segments); sj_add(report, "merged", merges);
    for (size_t i = 0; i < p->count; i++) {
        const h3_render_segment *s = &p->segments[i]; sj_value *row = sj_object();
        sj_add(row, "name", sj_string(s->name)); sj_add(row, "line", sj_int(s->line));
        sj_add(row, "label", sj_string(s->label));
        sj_add(row, "seed", sj_uint((uint64_t)strtoull(value(s, "seed"), NULL, 10)));
        if (p->seed_count) sj_add(row, "seed_offset", sj_uint(s->seed_offset));
        sj_add(row, "prompt", sj_string(s->prompt)); sj_add(row, "continuing", sj_bool(s->continuing));
        sj_add(row, "status", sj_string(s->skipped ? "skipped" : "planned"));
        sj_add(row, "frames", sj_int(s->frames)); sj_add(row, "delivered", sj_int(s->delivered));
        sj_add(row, "width", sj_int(s->width)); sj_add(row, "height", sj_int(s->height));
        char *video = h3_render_format("%s/%s", p->output, s->name);
        sj_add(row, "video", sj_string(video)); free(video);
        if (!s->skipped) {
            h3_request request = {0}; char error[256]; int argc = 0;
            if (!h3_render_request(p, i, &request, error, sizeof(error))) { h3_request_free(&request); sj_free(row); sj_free(report); return NULL; }
            char **argv = h3_request_argv(&request, &argc);
            sj_value *command = sj_array();
            for (int j = 0; argv && j < argc; j++) sj_add(command, NULL, sj_string(j ? argv[j] : binary));
            h3_argv_free(argv); h3_request_free(&request); sj_add(row, "command", command);
        }
        sj_add(segments, NULL, row);
    }
    if (!p->nomerge) {
        for (size_t i = 0; i < p->count; ) {
            size_t end = i + 1;
            while (end < p->count && p->segments[end].continuing) end++;
            if (!p->segments[i].skipped && end > i+1) {
                char *name = h3_render_chain_name(&p->segments[i]); sj_value *row = sj_object(), *inputs = sj_array();
                sj_add(row, "name", sj_string(name)); sj_add(row, "status", sj_string("planned"));
                if (p->seed_count) sj_add(row, "seed_offset", sj_uint(p->segments[i].seed_offset));
                for (size_t j = i; j < end; j++) sj_add(inputs, NULL, sj_string(p->segments[j].name));
                sj_add(row, "segments", inputs); sj_add(merges, NULL, row); free(name);
            }
            i = end;
        }
        size_t runs = p->seed_count ? p->seed_count : 1, per_run = p->count / runs;
        for (size_t run = 0; run < runs; run++) {
            sj_value *inputs = sj_array();
            for (size_t i = run * per_run; i < (run + 1) * per_run; i++)
                if (!p->segments[i].skipped) sj_add(inputs, NULL, sj_string(p->segments[i].name));
            if (inputs->count) {
                char *name = p->seed_count ? h3_render_format("seed-%" PRIu64 "/%s", p->seed_set[run], p->merged) : copy(p->merged);
                sj_value *row = sj_object(); sj_add(row, "name", sj_string(name)); free(name);
                if (p->seed_count) sj_add(row, "seed_offset", sj_uint(p->seed_set[run]));
                sj_add(row, "status", sj_string("planned")); sj_add(row, "segments", inputs); sj_add(merges, NULL, row);
            } else sj_free(inputs);
        }
    }
    return report;
}
