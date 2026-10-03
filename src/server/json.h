#ifndef H3_SERVER_JSON_H
#define H3_SERVER_JSON_H
#include "src/weights/lora_json.h"
#include <stdint.h>
typedef lj_value sj_value;
#define sj_get lj_get
#define sj_free lj_free
#define sj_parse lj_parse
sj_value *sj_object(void);
sj_value *sj_array(void);
sj_value *sj_string(const char *text);
sj_value *sj_bool(int value);
sj_value *sj_int(int64_t value);
sj_value *sj_uint(uint64_t value);
sj_value *sj_real(double value);
sj_value *sj_null(void);
int sj_add(sj_value *parent,const char *key,sj_value *owned);
int sj_set(sj_value *object,const char *key,sj_value *owned);
int sj_write(FILE *file,const sj_value *value);
void sj_sort(sj_value *value);
char *sj_dump(const sj_value *value);
sj_value *sj_clone(const sj_value *value);
const char *sj_text(const sj_value *value);
const char *sj_field(const sj_value *value,const char *key);
int sj_u64(const sj_value *value,uint64_t *out);
int sj_i64(const sj_value *value,int64_t *out);
#endif
