#ifndef H3_RUNTIME_MACOS_H
#define H3_RUNTIME_MACOS_H
#import <Foundation/Foundation.h>
#include <sys/stat.h>
#include <stdint.h>
/* Internal Objective-C API shared by the bootstrap and package-only core. */
_Noreturn void h3_mac_fail(NSString *message);
NSString *h3_mac_sha(const void *bytes, size_t size);
NSString *h3_mac_file_sha(int fd);
NSDictionary *h3_mac_json(NSData *data);
NSData *h3_mac_encode(id value);
int h3_mac_directory(NSString *path, BOOL create);
int h3_mac_parent(int root, NSString *path, BOOL create);
BOOL h3_mac_path(NSString *path);
uint64_t h3_mac_number(id value, uint64_t maximum);
NSArray *h3_mac_stamp(struct stat *st);
NSData *h3_mac_read(int root, NSString *path, size_t maximum);
void h3_mac_write(int root, NSString *path, NSData *data, mode_t mode);
NSDictionary *h3_mac_verify(int root, NSDictionary *expected);
void h3_mac_clean_environment(void);
#endif
