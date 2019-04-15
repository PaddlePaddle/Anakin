#ifndef THIRD_PARTY_BLOOMFILTER_MURMUR3_H 
#define THIRD_PARTY_BLOOMFILTER_MURMUR3_H

#include <stdlib.h>
#include <stdint.h>

void
murmur3_hash32(const void *key, size_t len, uint32_t seed, void *out);
void
murmurhash3_x64_128(const void * key, const int len, const uint32_t seed, void * out);

#endif 
