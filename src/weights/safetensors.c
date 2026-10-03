#include "src/weights/safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define H3_ST_MAX_HEADER (256u * 1024u * 1024u)

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
} h3_json_cursor;

static int h3_json_fail(h3_json_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size) {
        snprintf(cursor->error, cursor->error_size, "%s", message);
    }
    return 0;
}

static void h3_json_ws(h3_json_cursor *cursor) {
    while (cursor->at < cursor->end &&
           (*cursor->at == ' ' || *cursor->at == '\n' ||
            *cursor->at == '\r' || *cursor->at == '\t')) cursor->at++;
}

static int h3_json_take(h3_json_cursor *cursor, char expected) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != expected) {
        return h3_json_fail(cursor, "malformed safetensors JSON header");
    }
    cursor->at++;
    return 1;
}

static int h3_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static char *h3_json_string(h3_json_cursor *cursor) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        h3_json_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t maximum = (size_t)(cursor->end - cursor->at);
    char *result = malloc(maximum + 1);
    if (!result) {
        h3_json_fail(cursor, "out of memory parsing safetensors header");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        unsigned char value = (unsigned char)*cursor->at++;
        if (value == '\\') {
            if (cursor->at >= cursor->end) goto malformed;
            value = (unsigned char)*cursor->at++;
            switch (value) {
                case '"': result[length++] = '"'; break;
                case '\\': result[length++] = '\\'; break;
                case '/': result[length++] = '/'; break;
                case 'b': result[length++] = '\b'; break;
                case 'f': result[length++] = '\f'; break;
                case 'n': result[length++] = '\n'; break;
                case 'r': result[length++] = '\r'; break;
                case 't': result[length++] = '\t'; break;
                case 'u': {
                    if (cursor->end - cursor->at < 4) goto malformed;
                    int codepoint = 0;
                    for (int index = 0; index < 4; index++) {
                        int digit = h3_hex(cursor->at[index]);
                        if (digit < 0) goto malformed;
                        codepoint = codepoint * 16 + digit;
                    }
                    cursor->at += 4;
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (cursor->end - cursor->at < 6 || cursor->at[0] != '\\' ||
                            cursor->at[1] != 'u') goto malformed;
                        int low = 0;
                        for (int index = 2; index < 6; index++) {
                            int digit = h3_hex(cursor->at[index]);
                            if (digit < 0) goto malformed;
                            low = low * 16 + digit;
                        }
                        if (low < 0xdc00 || low > 0xdfff) goto malformed;
                        cursor->at += 6;
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) goto malformed;
                    /* Public tensor names/metadata use C strings. */
                    if (!codepoint) goto malformed;
                    if (codepoint < 0x80) {
                        result[length++] = (char)codepoint;
                    } else if (codepoint < 0x800) {
                        result[length++] = (char)(0xc0 | (codepoint >> 6));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else if (codepoint < 0x10000) {
                        result[length++] = (char)(0xe0 | (codepoint >> 12));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else {
                        result[length++] = (char)(0xf0 | (codepoint >> 18));
                        result[length++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    }
                    break;
                }
                default: goto malformed;
            }
        } else {
            if (value < 0x20) goto malformed;
            result[length++] = (char)value;
            if (value >= 0x80) {
                int extra = value >= 0xc2 && value <= 0xdf ? 1 :
                            value >= 0xe0 && value <= 0xef ? 2 :
                            value >= 0xf0 && value <= 0xf4 ? 3 : -1;
                if (extra < 0 || cursor->end - cursor->at < extra) goto malformed;
                unsigned char first = (unsigned char)*cursor->at;
                if ((value == 0xe0 && first < 0xa0) || (value == 0xed && first >= 0xa0) ||
                    (value == 0xf0 && first < 0x90) || (value == 0xf4 && first >= 0x90)) goto malformed;
                for (int i = 0; i < extra; i++) {
                    unsigned char next = (unsigned char)*cursor->at++;
                    if ((next & 0xc0) != 0x80) goto malformed;
                    result[length++] = (char)next;
                }
            }
        }
    }
    if (cursor->at >= cursor->end) goto malformed;
    cursor->at++;
    result[length] = '\0';
    char *compact = realloc(result, length + 1);
    return compact ? compact : result;

malformed:
    free(result);
    h3_json_fail(cursor, "malformed JSON string escape");
    return NULL;
}

static int h3_json_uint(h3_json_cursor *cursor, uint64_t *result) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at < '0' || *cursor->at > '9') {
        return h3_json_fail(cursor, "expected unsigned JSON integer");
    }
    if (*cursor->at == '0' && cursor->at + 1 < cursor->end &&
        cursor->at[1] >= '0' && cursor->at[1] <= '9')
        return h3_json_fail(cursor, "leading zero in JSON integer");
    uint64_t value = 0;
    while (cursor->at < cursor->end && *cursor->at >= '0' && *cursor->at <= '9') {
        unsigned digit = (unsigned)(*cursor->at - '0');
        if (value > (UINT64_MAX - digit) / 10) {
            return h3_json_fail(cursor, "integer overflow in safetensors header");
        }
        value = value * 10 + digit;
        cursor->at++;
    }
    *result = value;
    return 1;
}

/* Metadata is a flat string-to-string map, never arbitrary nested JSON. */
static int h3_parse_metadata(h3_json_cursor *cursor, h3_st_header *header) {
    if (!h3_json_take(cursor, '{')) return 0;
    h3_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == '}') {
        cursor->at++;
        return 1;
    }
    char **keys = NULL;
    size_t count = 0;
    int ok = 0;
    for (;;) {
        char *key = h3_json_string(cursor);
        if (!key) break;
        int duplicate = 0;
        for (size_t i = 0; i < count; i++)
            if (!strcmp(keys[i], key)) duplicate = 1;
        char **next = duplicate ? NULL : realloc(keys, (count + 1) * sizeof(*keys));
        if (!next) {
            free(key);
            h3_json_fail(cursor, duplicate ? "duplicate metadata key" : "out of memory");
            break;
        }
        keys = next;
        keys[count++] = key;
        if (!h3_json_take(cursor, ':')) break;
        char *value = h3_json_string(cursor);
        if (!value) break;
        char **values = realloc(header->metadata_values, count * sizeof(*values));
        if (!values) { free(value); h3_json_fail(cursor, "out of memory"); break; }
        header->metadata_values = values;
        values[header->metadata_count++] = value;
        h3_json_ws(cursor);
        if (cursor->at < cursor->end && *cursor->at == '}') {
            cursor->at++;
            ok = 1;
            break;
        }
        if (!h3_json_take(cursor, ',')) break;
    }
    if (ok) header->metadata_keys = keys;
    else { for (size_t i = 0; i < count; i++) free(keys[i]); free(keys); }
    return ok;
}

/* Bounds checked before converting an offset to off_t or touching payloads. */
int h3_st_file_range(uint64_t offset, uint64_t bytes, uint64_t file_size) {
    return offset <= INT64_MAX && bytes <= (uint64_t)INT64_MAX - offset &&
           offset <= file_size && bytes <= file_size - offset;
}

int h3_st_zero_copy_plan(uint64_t offset, size_t bytes, h3_dtype dtype,
                         uint64_t file_size, size_t page, size_t buffer_limit,
                         size_t *mapping_bytes, char *reason, size_t reason_size) {
    const char *why = NULL;
    size_t item = h3_dtype_size(dtype), rounded = 0;
    if (mapping_bytes) *mapping_bytes = 0;
    if (!h3_st_file_range(offset, bytes, file_size)) why = "tensor range exceeds file or off_t";
    else if (!item || bytes % item) why = "invalid dtype or tensor byte length";
    else if (!bytes) why = "empty tensor";
    else if (offset % item) why = "tensor offset is not dtype aligned";
    else if (!page || (page & (page - 1))) why = "invalid mapping page size";
    else if (offset % page) why = "tensor offset is not Metal page aligned";
    else if (bytes > SIZE_MAX - (page - 1)) why = "mapping length overflows";
    else {
        rounded = (bytes + page - 1) & ~(page - 1);
        if (rounded > buffer_limit || rounded > (uint64_t)INT64_MAX - offset)
            why = "mapping length exceeds buffer or offset limit";
    }
    if (reason && reason_size) snprintf(reason, reason_size, "%s", why ? why : "eligible");
    if (why) return 0;
    if (mapping_bytes) *mapping_bytes = rounded;
    return 1;
}

static h3_dtype h3_parse_dtype(const char *name) {
    if (!strcmp(name, "BOOL")) return H3_DTYPE_BOOL;
    if (!strcmp(name, "I8")) return H3_DTYPE_I8;
    if (!strcmp(name, "U8")) return H3_DTYPE_U8;
    if (!strcmp(name, "I16")) return H3_DTYPE_I16;
    if (!strcmp(name, "U16")) return H3_DTYPE_U16;
    if (!strcmp(name, "F16")) return H3_DTYPE_F16;
    if (!strcmp(name, "BF16")) return H3_DTYPE_BF16;
    if (!strcmp(name, "I32")) return H3_DTYPE_I32;
    if (!strcmp(name, "U32")) return H3_DTYPE_U32;
    if (!strcmp(name, "F32")) return H3_DTYPE_F32;
    if (!strcmp(name, "I64")) return H3_DTYPE_I64;
    if (!strcmp(name, "U64")) return H3_DTYPE_U64;
    if (!strcmp(name, "F64")) return H3_DTYPE_F64;
    return H3_DTYPE_UNKNOWN;
}

size_t h3_dtype_size(h3_dtype dtype) {
    switch (dtype) {
        case H3_DTYPE_BOOL:
        case H3_DTYPE_I8:
        case H3_DTYPE_U8: return 1;
        case H3_DTYPE_I16:
        case H3_DTYPE_U16:
        case H3_DTYPE_F16:
        case H3_DTYPE_BF16: return 2;
        case H3_DTYPE_I32:
        case H3_DTYPE_U32:
        case H3_DTYPE_F32: return 4;
        case H3_DTYPE_I64:
        case H3_DTYPE_U64:
        case H3_DTYPE_F64: return 8;
        case H3_DTYPE_UNKNOWN: return 0;
    }
    return 0;
}

const char *h3_dtype_name(h3_dtype dtype) {
    switch (dtype) {
        case H3_DTYPE_BOOL: return "BOOL";
        case H3_DTYPE_I8: return "I8";
        case H3_DTYPE_U8: return "U8";
        case H3_DTYPE_I16: return "I16";
        case H3_DTYPE_U16: return "U16";
        case H3_DTYPE_F16: return "F16";
        case H3_DTYPE_BF16: return "BF16";
        case H3_DTYPE_I32: return "I32";
        case H3_DTYPE_U32: return "U32";
        case H3_DTYPE_F32: return "F32";
        case H3_DTYPE_I64: return "I64";
        case H3_DTYPE_U64: return "U64";
        case H3_DTYPE_F64: return "F64";
        case H3_DTYPE_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

static int h3_parse_shape(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '[')) return 0;
    h3_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == ']') {
        cursor->at++;
        tensor->ndim = 0;
        return 1;
    }
    int dimensions = 0;
    for (;;) {
        uint64_t value;
        if (dimensions >= 8) return h3_json_fail(cursor, "tensor rank exceeds 8");
        if (!h3_json_uint(cursor, &value)) return 0;
        tensor->shape[dimensions++] = value;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated shape");
        if (*cursor->at == ']') {
            cursor->at++;
            tensor->ndim = dimensions;
            return 1;
        }
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected shape comma");
    }
}

static int h3_parse_offsets(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '[') ||
        !h3_json_uint(cursor, &tensor->data_begin) ||
        !h3_json_take(cursor, ',') ||
        !h3_json_uint(cursor, &tensor->data_end) ||
        !h3_json_take(cursor, ']')) return 0;
    return 1;
}

static int h3_parse_tensor(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '{')) return 0;
    int have_dtype = 0;
    int have_shape = 0;
    int have_offsets = 0;
    for (;;) {
        h3_json_ws(cursor);
        if (cursor->at < cursor->end && *cursor->at == '}') {
            cursor->at++;
            break;
        }
        char *key = h3_json_string(cursor);
        if (!key || !h3_json_take(cursor, ':')) {
            free(key);
            return 0;
        }
        int ok = 1;
        if (!strcmp(key, "dtype")) {
            if (have_dtype) { free(key); return h3_json_fail(cursor, "duplicate dtype"); }
            char *value = h3_json_string(cursor);
            if (!value) ok = 0;
            else {
                tensor->dtype = h3_parse_dtype(value);
                have_dtype = 1;
                if (tensor->dtype == H3_DTYPE_UNKNOWN)
                    ok = h3_json_fail(cursor, "unknown tensor dtype");
            }
            free(value);
        } else if (!strcmp(key, "shape")) {
            if (have_shape) { free(key); return h3_json_fail(cursor, "duplicate shape"); }
            ok = h3_parse_shape(cursor, tensor);
            have_shape = ok;
        } else if (!strcmp(key, "data_offsets")) {
            if (have_offsets) { free(key); return h3_json_fail(cursor, "duplicate data_offsets"); }
            ok = h3_parse_offsets(cursor, tensor);
            have_offsets = ok;
        } else {
            ok = h3_json_fail(cursor, "unknown tensor descriptor field");
        }
        free(key);
        if (!ok) return 0;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated tensor entry");
        if (*cursor->at == '}') continue;
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected tensor comma");
        h3_json_ws(cursor);
        if (cursor->at < cursor->end && *cursor->at == '}')
            return h3_json_fail(cursor, "trailing tensor comma");
    }
    if (!have_dtype || !have_shape || !have_offsets) {
        return h3_json_fail(cursor, "incomplete tensor descriptor");
    }
    return 1;
}

static int h3_validate_tensor(h3_json_cursor *cursor, h3_st_tensor *tensor,
                              uint64_t data_start, uint64_t file_size) {
    if (data_start > file_size || data_start > INT64_MAX ||
        tensor->data_end < tensor->data_begin ||
        tensor->data_end > file_size - data_start) {
        return h3_json_fail(cursor, "tensor data offsets exceed file");
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] && elements > UINT64_MAX / tensor->shape[dimension]) {
            return h3_json_fail(cursor, "tensor element count overflow");
        }
        elements *= tensor->shape[dimension];
    }
    size_t item_size = h3_dtype_size(tensor->dtype);
    if (item_size && elements > UINT64_MAX / item_size) {
        return h3_json_fail(cursor, "tensor byte count overflow");
    }
    uint64_t expected = elements * item_size;
    if (expected != tensor->data_end - tensor->data_begin) {
        return h3_json_fail(cursor, "tensor shape does not match byte offsets");
    }
    tensor->file_offset = data_start + tensor->data_begin;
    return 1;
}

static int h3_append_tensor(h3_st_header *header, h3_st_tensor tensor,
                            size_t *capacity, h3_json_cursor *cursor) {
    if (header->tensor_count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 64;
        h3_st_tensor *values = realloc(header->tensors, next * sizeof(*values));
        if (!values) return h3_json_fail(cursor, "out of memory indexing tensors");
        header->tensors = values;
        *capacity = next;
    }
    header->tensors[header->tensor_count++] = tensor;
    return 1;
}

static int h3_tensor_name_order(const void *a, const void *b) {
    const h3_st_tensor *x = *(const h3_st_tensor *const *)a;
    const h3_st_tensor *y = *(const h3_st_tensor *const *)b;
    return strcmp(x->name, y->name);
}
static int h3_tensor_range_order(const void *a, const void *b) {
    const h3_st_tensor *x = *(const h3_st_tensor *const *)a;
    const h3_st_tensor *y = *(const h3_st_tensor *const *)b;
    if (x->data_begin != y->data_begin) return x->data_begin < y->data_begin ? -1 : 1;
    return x->data_end < y->data_end ? -1 : x->data_end > y->data_end;
}
static int h3_validate_index(h3_st_header *header, h3_json_cursor *cursor) {
    size_t n = header->tensor_count;
    const h3_st_tensor **sorted = malloc((n ? n : 1) * sizeof(*sorted));
    if (!sorted) return h3_json_fail(cursor, "out of memory validating tensors");
    for (size_t i = 0; i < n; i++) sorted[i] = &header->tensors[i];
    qsort(sorted, n, sizeof(*sorted), h3_tensor_name_order);
    for (size_t i = 1; i < n; i++) {
        if (!strcmp(sorted[i-1]->name, sorted[i]->name)) {
            free(sorted);
            return h3_json_fail(cursor, "duplicate tensor name");
        }
    }
    qsort(sorted, n, sizeof(*sorted), h3_tensor_range_order);
    uint64_t end = 0;
    for (size_t i = 0; i < n; i++) {
        if (sorted[i]->data_begin != end) {
            free(sorted);
            return h3_json_fail(cursor, "overlapping tensor ranges or unindexed payload gap");
        }
        end = sorted[i]->data_end;
    }
    free(sorted);
    if (end != header->file_size - 8 - header->header_size)
        return h3_json_fail(cursor, "unindexed trailing safetensors payload");
    return 1;
}

static uint64_t h3_u64_le(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

int h3_st_read_header(const char *path, h3_st_header *header,
                      char *error, size_t error_size) {
    if (!path || !header) return 0;
    memset(header, 0, sizeof(*header));
    if (error && error_size) error[0] = '\0';
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    struct stat status;
    unsigned char prefix[8];
    if (fstat(descriptor, &status) != 0 || status.st_size < 8 ||
        pread(descriptor, prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid file", path);
        close(descriptor);
        return 0;
    }
    uint64_t file_size = (uint64_t)status.st_size;
    uint64_t header_size = h3_u64_le(prefix);
    if (!header_size || header_size > H3_ST_MAX_HEADER ||
        !h3_st_file_range(8, header_size, file_size)) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid header size", path);
        close(descriptor);
        return 0;
    }
    char *json = malloc((size_t)header_size + 1);
    if (!json || pread(descriptor, json, (size_t)header_size, 8) != (ssize_t)header_size) {
        if (error && error_size) snprintf(error, error_size, "%s: cannot read header", path);
        free(json);
        close(descriptor);
        return 0;
    }
    close(descriptor);
    json[header_size] = '\0';
    header->path = strdup(path);
    header->file_size = file_size;
    header->header_size = header_size;
    if (!header->path) {
        free(json);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    h3_json_cursor cursor = {json, json + header_size, error, error_size};
    size_t capacity = 0;
    int have_metadata = 0;
    if (*json != '{' || !h3_json_take(&cursor, '{')) {
        h3_json_fail(&cursor, "safetensors header must begin with an object");
        goto fail;
    }
    h3_json_ws(&cursor);
    if (cursor.at < cursor.end && *cursor.at == '}') cursor.at++;
    else for (;;) {
        char *name = h3_json_string(&cursor);
        if (!name || !h3_json_take(&cursor, ':')) {
            free(name);
            goto fail;
        }
        if (!strcmp(name, "__metadata__")) {
            free(name);
            if (have_metadata++) {
                h3_json_fail(&cursor, "duplicate metadata entry");
                goto fail;
            }
            if (!h3_parse_metadata(&cursor, header)) goto fail;
        } else {
            h3_st_tensor tensor;
            memset(&tensor, 0, sizeof(tensor));
            tensor.name = name;
            if (!h3_parse_tensor(&cursor, &tensor) ||
                !h3_validate_tensor(&cursor, &tensor, 8 + header_size, file_size) ||
                !h3_append_tensor(header, tensor, &capacity, &cursor)) {
                free(tensor.name);
                goto fail;
            }
        }
        h3_json_ws(&cursor);
        if (cursor.at >= cursor.end) {
            h3_json_fail(&cursor, "unterminated safetensors object");
            goto fail;
        }
        if (*cursor.at == '}') {
            cursor.at++;
            break;
        }
        if (*cursor.at++ != ',') {
            h3_json_fail(&cursor, "expected top-level comma");
            goto fail;
        }
    }
    h3_json_ws(&cursor);
    if (cursor.at != cursor.end) {
        h3_json_fail(&cursor, "trailing data in safetensors header");
        goto fail;
    }
    if (!h3_validate_index(header, &cursor)) goto fail;
    free(json);
    return 1;

fail:
    free(json);
    h3_st_free_header(header);
    return 0;
}

void h3_st_free_header(h3_st_header *header) {
    if (!header) return;
    for (size_t index = 0; index < header->tensor_count; index++) {
        free(header->tensors[index].name);
    }
    for (size_t i = 0; i < header->metadata_count; i++) {
        if (header->metadata_keys) free(header->metadata_keys[i]);
        free(header->metadata_values[i]);
    }
    free(header->metadata_keys); free(header->metadata_values);
    free(header->tensors);
    free(header->path);
    memset(header, 0, sizeof(*header));
}

const h3_st_tensor *h3_st_find(const h3_st_header *header, const char *name) {
    if (!header || !name) return NULL;
    for (size_t index = 0; index < header->tensor_count; index++) {
        if (!strcmp(header->tensors[index].name, name)) return &header->tensors[index];
    }
    return NULL;
}

uint64_t h3_st_tensor_elements(const h3_st_tensor *tensor) {
    if (!tensor) return 0;
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] != 0 &&
            elements > UINT64_MAX / tensor->shape[dimension]) return 0;
        elements *= tensor->shape[dimension];
    }
    return elements;
}

int h3_st_read_data(const h3_st_header *header, const h3_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!header || !header->path || !tensor || (!data && bytes != 0)) {
        if (error && error_size) snprintf(error, error_size, "invalid tensor read request");
        return 0;
    }
    if (tensor->data_end < tensor->data_begin ||
        !h3_st_file_range(tensor->file_offset, bytes, header->file_size)) {
        if (error && error_size) snprintf(error, error_size, "invalid tensor file range");
        return 0;
    }
    uint64_t tensor_bytes = tensor->data_end - tensor->data_begin;
    if (tensor_bytes != bytes) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: expected %llu bytes, got %zu",
                     tensor->name ? tensor->name : "tensor",
                     (unsigned long long)tensor_bytes, bytes);
        }
        return 0;
    }
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: %s", header->path, strerror(errno));
        }
        return 0;
    }
    struct stat status;
    if (fstat(descriptor, &status) || status.st_size < 0 ||
        !h3_st_file_range(tensor->file_offset, bytes, (uint64_t)status.st_size)) {
        if (error && error_size) snprintf(error, error_size, "%s: tensor range exceeds current file", header->path);
        close(descriptor);
        return 0;
    }
    size_t done = 0;
    while (done < bytes) {
        size_t remaining = bytes - done;
        size_t chunk = remaining < (size_t)(1u << 30) ? remaining :
                                                            (size_t)(1u << 30);
        ssize_t count = pread(descriptor, (unsigned char *)data + done, chunk,
                              (off_t)(tensor->file_offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            if (error && error_size) {
                snprintf(error, error_size, "%s: cannot read %s: %s", header->path,
                         tensor->name ? tensor->name : "tensor",
                         count < 0 ? strerror(errno) : "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        done += (size_t)count;
    }
    close(descriptor);
    return 1;
}

static int h3_has_suffix(const char *value, const char *suffix) {
    size_t length = strlen(value);
    size_t suffix_length = strlen(suffix);
    return length >= suffix_length && !strcmp(value + length - suffix_length, suffix);
}

int h3_st_inventory_dir(const char *directory, h3_component_info *info,
                        char *error, size_t error_size) {
    if (!directory || !info) return 0;
    memset(info, 0, sizeof(*info));
    DIR *stream = opendir(directory);
    if (!stream) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", directory, strerror(errno));
        return 0;
    }
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (entry->d_name[0] == '.' || !h3_has_suffix(entry->d_name, ".safetensors")) continue;
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        char *path = malloc(length);
        if (!path) {
            if (error && error_size) snprintf(error, error_size, "out of memory");
            closedir(stream);
            return 0;
        }
        snprintf(path, length, "%s/%s", directory, entry->d_name);
        h3_st_header header;
        int ok = h3_st_read_header(path, &header, error, error_size);
        free(path);
        if (!ok) {
            closedir(stream);
            return 0;
        }
        info->files++;
        info->bytes += header.file_size;
        info->tensors += header.tensor_count;
        for (size_t index = 0; index < header.tensor_count; index++) {
            info->tensor_bytes += header.tensors[index].data_end -
                                  header.tensors[index].data_begin;
        }
        h3_st_free_header(&header);
    }
    closedir(stream);
    if (!info->files) {
        if (error && error_size) snprintf(error, error_size, "%s: no safetensors files", directory);
        return 0;
    }
    return 1;
}
