#ifndef THIRD_PARTY_BLOOMFILTER_BLOOMFILTER_H
#define THIRD_PARTY_BLOOMFILTER_BLOOMFILTER_H

#include <stdlib.h>
#include <inttypes.h>

struct bloomfilter {
    uint64_t  magic_num;
    uint64_t  m;
    uint64_t  k;
    uint64_t  count;
    unsigned char bit_vector[1];
};

int bloomfilter_check(struct bloomfilter* filter);

void
bloomfilter_init(struct bloomfilter *bloomfilter, uint64_t m, uint64_t k);

int
bloomfilter_set(struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_set_nocheck(struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_get(struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_dump(struct bloomfilter *bloomfilter, const void *path);

int
bloomfilter_load(struct bloomfilter **bloomfilter, const void *path);

int
bloomfilter_get_hash(struct bloomfilter *bloomfilter, const void *key, size_t len, char *dst);

uint64_t
char_to_little_endian_64bits(unsigned char *bytes);

uint32_t
char_to_little_endian_32bits(unsigned char *bytes);

#endif /* __BLOOMFILTER_H__ */
