#import <Foundation/Foundation.h>
#include "src/conditioning/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

/* JSON corpus in, JSON token IDs/decoded text out. Also builds against the
 * pre-fix tokenizer for exact before/after comparisons. No model/GPU needed. */
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s tokenizer.json corpus.json\n", argv[0]);
            return 2;
        }
        char error[512];
        h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
        if (!tokenizer) { fprintf(stderr, "%s\n", error); return 1; }
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
        NSArray *corpus = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
        if (![corpus isKindOfClass:NSArray.class]) return 2;
        NSMutableArray *results = [NSMutableArray array];
        for (NSDictionary *item in corpus) {
            NSString *text = item[@"text"];
            uint32_t *ids = NULL;
            size_t count = 0;
            if (!h3_tokenizer_encode(tokenizer, text.UTF8String, 0, &ids, &count,
                                     error, sizeof(error))) {
                fprintf(stderr, "%s\n", error); return 1;
            }
            char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
            if (!decoded) { fprintf(stderr, "%s\n", error); return 1; }
            NSMutableArray *numbers = [NSMutableArray array];
            for (size_t i = 0; i < count; i++) [numbers addObject:@(ids[i])];
            [results addObject:@{@"name":item[@"name"], @"text":text, @"ids":numbers,
                                 @"decoded":[NSString stringWithUTF8String:decoded]}];
            free(decoded); h3_tokenizer_ids_free(ids);
        }
        h3_tokenizer_free(tokenizer);
        NSData *json = [NSJSONSerialization dataWithJSONObject:results
                             options:NSJSONWritingPrettyPrinted error:NULL];
        fwrite(json.bytes, 1, json.length, stdout); putchar('\n');
        return 0;
    }
}
