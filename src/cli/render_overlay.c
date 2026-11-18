#include "src/cli/render.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* Five-by-seven bitmap lettering keeps badges available in the bundled media
 * tools, which do not require drawtext, font discovery or external font files. */
static const unsigned char letters[95][7] = {
    ['!'-32]={4,4,4,4,4,0,4}, ['"'-32]={10,10,10,0,0,0,0},
    ['#'-32]={10,10,31,10,31,10,10}, ['$'-32]={4,15,20,14,5,30,4},
    ['%'-32]={24,25,2,4,8,19,3}, ['&'-32]={12,18,20,8,21,18,13},
    ['\''-32]={4,4,8,0,0,0,0}, ['('-32]={2,4,8,8,8,4,2},
    [')'-32]={8,4,2,2,2,4,8}, ['*'-32]={0,21,14,31,14,21,0},
    ['+'-32]={0,4,4,31,4,4,0}, [','-32]={0,0,0,0,4,4,8},
    ['-'-32]={0,0,0,31,0,0,0}, ['.'-32]={0,0,0,0,0,12,12},
    ['/'-32]={1,1,2,4,8,16,16},
    ['0'-32]={14,17,19,21,25,17,14}, ['1'-32]={4,12,4,4,4,4,14},
    ['2'-32]={14,17,1,2,4,8,31}, ['3'-32]={30,1,1,14,1,1,30},
    ['4'-32]={2,6,10,18,31,2,2}, ['5'-32]={31,16,16,30,1,1,30},
    ['6'-32]={14,16,16,30,17,17,14}, ['7'-32]={31,1,1,2,4,4,4},
    ['8'-32]={14,17,17,14,17,17,14}, ['9'-32]={14,17,17,15,1,1,14},
    [':'-32]={0,12,12,0,12,12,0}, [';'-32]={0,12,12,0,4,4,8},
    ['<'-32]={2,4,8,16,8,4,2}, ['='-32]={0,0,31,0,31,0,0},
    ['>'-32]={8,4,2,1,2,4,8}, ['?'-32]={14,17,1,2,4,0,4},
    ['@'-32]={14,17,23,21,23,16,14},
    ['A'-32]={14,17,17,31,17,17,17}, ['B'-32]={30,17,17,30,17,17,30},
    ['C'-32]={14,17,16,16,16,17,14}, ['D'-32]={30,17,17,17,17,17,30},
    ['E'-32]={31,16,16,30,16,16,31}, ['F'-32]={31,16,16,30,16,16,16},
    ['G'-32]={14,17,16,23,17,17,15}, ['H'-32]={17,17,17,31,17,17,17},
    ['I'-32]={14,4,4,4,4,4,14}, ['J'-32]={7,2,2,2,2,18,12},
    ['K'-32]={17,18,20,24,20,18,17}, ['L'-32]={16,16,16,16,16,16,31},
    ['M'-32]={17,27,21,21,17,17,17}, ['N'-32]={17,25,21,19,17,17,17},
    ['O'-32]={14,17,17,17,17,17,14}, ['P'-32]={30,17,17,30,16,16,16},
    ['Q'-32]={14,17,17,17,21,18,13}, ['R'-32]={30,17,17,30,20,18,17},
    ['S'-32]={15,16,16,14,1,1,30}, ['T'-32]={31,4,4,4,4,4,4},
    ['U'-32]={17,17,17,17,17,17,14}, ['V'-32]={17,17,17,17,17,10,4},
    ['W'-32]={17,17,17,21,21,21,10}, ['X'-32]={17,17,10,4,10,17,17},
    ['Y'-32]={17,17,10,4,4,4,4}, ['Z'-32]={31,1,2,4,8,16,31},
    ['['-32]={14,8,8,8,8,8,14}, ['\\'-32]={16,16,8,4,2,1,1},
    [']'-32]={14,2,2,2,2,2,14}, ['^'-32]={4,10,17,0,0,0,0},
    ['_'-32]={0,0,0,0,0,0,31}, ['`'-32]={8,4,2,0,0,0,0},
    ['a'-32]={0,0,14,1,15,17,15}, ['b'-32]={16,16,22,25,17,17,30},
    ['c'-32]={0,0,14,17,16,17,14}, ['d'-32]={1,1,13,19,17,17,15},
    ['e'-32]={0,0,14,17,31,16,14}, ['f'-32]={6,9,8,28,8,8,8},
    ['g'-32]={0,15,17,17,15,1,14}, ['h'-32]={16,16,22,25,17,17,17},
    ['i'-32]={4,0,12,4,4,4,14}, ['j'-32]={2,0,6,2,2,18,12},
    ['k'-32]={16,16,18,20,24,20,18}, ['l'-32]={12,4,4,4,4,4,14},
    ['m'-32]={0,0,26,21,21,17,17}, ['n'-32]={0,0,22,25,17,17,17},
    ['o'-32]={0,0,14,17,17,17,14}, ['p'-32]={0,30,17,17,30,16,16},
    ['q'-32]={0,15,17,17,15,1,1}, ['r'-32]={0,0,22,25,16,16,16},
    ['s'-32]={0,0,15,16,14,1,30}, ['t'-32]={8,8,28,8,8,9,6},
    ['u'-32]={0,0,17,17,17,19,13}, ['v'-32]={0,0,17,17,17,10,4},
    ['w'-32]={0,0,17,17,21,21,10}, ['x'-32]={0,0,17,10,4,10,17},
    ['y'-32]={0,17,17,17,15,1,14}, ['z'-32]={0,0,31,2,4,8,31},
    ['{'-32]={2,4,4,8,4,4,2}, ['|'-32]={4,4,4,4,4,4,4},
    ['}'-32]={8,4,4,2,4,4,8}, ['~'-32]={0,0,8,21,2,0,0}
};

int h3_render_badge(const char *path, const char *label, int width, int height,
                    int *margin, char *error, size_t size) {
    /* The planner has validated UTF-8. Non-ASCII characters retain their
     * identity as codepoint labels instead of ambiguous missing-glyph boxes. */
    char text[2048]; size_t length = 0;
    for (const unsigned char *p = (const unsigned char *)label; *p; ) {
        unsigned c = *p++;
        if (c < 128) text[length++] = (char)c;
        else {
            int trailing = c < 224 ? 1 : c < 240 ? 2 : 3;
            c &= trailing == 1 ? 31u : trailing == 2 ? 15u : 7u;
            while (trailing--) c = (c << 6) | (*p++ & 63u);
            int n = snprintf(text + length, sizeof(text) - length, "[U+%04X]", c);
            if (n < 0 || (size_t)n >= sizeof(text) - length) goto invalid;
            length += (size_t)n;
        }
        if (length >= sizeof(text) - 16) goto invalid;
    }
    if (!length || width < 32 || height < 32) goto invalid;
    *margin = width < 256 || height < 256 ? 2 : 16;
    int pad = height < 128 ? 2 : 8, scale = 4, columns = 0, rows = 0;
    int available_w = width - 2 * (*margin + pad);
    int available_h = height / 3 - 2 * pad;
    if (available_h < 7) available_h = 7;
    for (; scale >= 1; scale--) {
        columns = (available_w + scale) / (6 * scale);
        rows = (available_h + scale) / (8 * scale);
        if (columns > 0 && rows > 0 && (size_t)columns * (size_t)rows >= length) break;
    }
    if (!scale) {
        scale = 1; columns = (available_w + 1) / 6; rows = (available_h + 1) / 8;
        if (columns < 1 || rows < 1) goto invalid;
        length = (size_t)columns * (size_t)rows;
        for (size_t i = 0; i < 3 && i < length; i++) text[length - 1 - i] = '.';
    }
    if ((size_t)columns > length) columns = (int)length;
    rows = (int)((length + (size_t)columns - 1) / (size_t)columns);
    int w = (columns * 6 - 1) * scale + 2 * pad;
    int h = (rows * 8 - 1) * scale + 2 * pad;
    size_t bytes = (size_t)w * (size_t)h * 3;
    unsigned char *pixels = malloc(bytes);
    if (!pixels) { snprintf(error, size, "out of memory creating segment badge"); return 0; }
    memset(pixels, 24, bytes);
    for (size_t i = 0; i < length; i++) {
        unsigned c = (unsigned char)text[i];
        if (c < 32 || c > 126) c = '?';
        int x = pad + (int)(i % (size_t)columns) * 6 * scale;
        int y = pad + (int)(i / (size_t)columns) * 8 * scale;
        for (int row = 0; row < 7; row++) for (int col = 0; col < 5; col++)
            if (letters[c - 32][row] & (1u << (4 - col)))
                for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) {
                    size_t at = ((size_t)(y + row * scale + dy) * (size_t)w + (size_t)(x + col * scale + dx)) * 3;
                    memset(pixels + at, 255, 3);
                }
    }
    FILE *file = fopen(path, "wb");
    int ok = file && fprintf(file, "P6\n%d %d\n255\n", w, h) > 0 && fwrite(pixels, 1, bytes, file) == bytes;
    if (file && fclose(file)) ok = 0;
    free(pixels);
    if (!ok) snprintf(error, size, "cannot write segment badge %s: %s", path, strerror(errno));
    return ok;
invalid:
    snprintf(error, size, "cannot fit segment badge into %dx%d video", width, height); return 0;
}
