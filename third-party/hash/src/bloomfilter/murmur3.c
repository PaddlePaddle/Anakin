#include "bloomfilter/murmur3.h"

#define ROTL32(x, r)	(((x) << (r)) | ((x) >> (32 - (r))))
#define ROTL64(x, r)	(((x) << (r)) | ((x) >> (64 - (r))))
#define BIG_CONSTANT(x) (x##LLU)

uint32_t fmix32(uint32_t h) {
    return h;
}

//uint64_t getblock64(const uint64_t * p, int i) {
//	return p[i];
//}

uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

void murmur3_hash32(const void *key, size_t len, uint32_t seed, void *out) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    int i = 0;
    uint32_t k1 = 0;
    uint32_t h1 = seed;

    const uint8_t *data = (const uint8_t *) key;
    const int nblocks = len >> 2;

    const uint32_t *blocks = (const uint32_t *) (data + nblocks * 4);
    const uint8_t *tail = (const uint8_t *) (data + nblocks * 4);

    for (i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    switch (len & 3) {
    case 3:
        k1 ^= tail[2] << 16;
        break;
    case 2:
        k1 ^= tail[1] << 8;
        break;
    case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        break;
    };

    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    *(uint32_t*) out = h1;
}

void murmurhash3_x64_128(const void * key, const int len, const uint32_t seed, void * out) {
    const uint8_t * data = (const uint8_t*) key;
    const int nblocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;
    int i = 0;

    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

    //----------
    // body

    const uint64_t * blocks = (const uint64_t *) (data);

    uint64_t k1;
    uint64_t k2;

    for (i = 0; i < nblocks; i++) {
        k1 = blocks[i * 2 + 0];
        k2 = blocks[i * 2 + 1];

        k1 *= c1;
        k1 = ROTL64(k1, 31);
        k1 *= c2;
        h1 ^= k1;

        h1 = ROTL64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = ROTL64(k2, 33);
        k2 *= c1;
        h2 ^= k2;

        h2 = ROTL64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*) (data + nblocks * 16);
    uint64_t nk1 = 0;
    uint64_t nk2 = 0;
    //no break here!!!
    switch (len & 15) {
        case 15:
            nk2 ^= ((uint64_t) tail[14]) << 48;
        case 14:
            nk2 ^= ((uint64_t) tail[13]) << 40;
        case 13:
            nk2 ^= ((uint64_t) tail[12]) << 32;
        case 12:
            nk2 ^= ((uint64_t) tail[11]) << 24;
        case 11:
            nk2 ^= ((uint64_t) tail[10]) << 16;
        case 10:
            nk2 ^= ((uint64_t) tail[9]) << 8;
        case 9:
            nk2 ^= ((uint64_t) tail[8]) << 0;
            nk2 *= c2;
            nk2 = ROTL64(nk2, 33);
            nk2 *= c1;
            h2 ^= nk2;
        case 8:
            nk1 ^= ((uint64_t) tail[7]) << 56;
        case 7:
            nk1 ^= ((uint64_t) tail[6]) << 48;
        case 6:
            nk1 ^= ((uint64_t) tail[5]) << 40;
        case 5:
            nk1 ^= ((uint64_t) tail[4]) << 32;
        case 4:
            nk1 ^= ((uint64_t) tail[3]) << 24;
        case 3:
            nk1 ^= ((uint64_t) tail[2]) << 16;
        case 2:
            nk1 ^= ((uint64_t) tail[1]) << 8;
        case 1:
            nk1 ^= ((uint64_t) tail[0]) << 0;
            nk1 *= c1;
            nk1 = ROTL64(nk1, 31);
            nk1 *= c2;
            h1 ^= nk1;
    };

    //----------
    // finalization

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    ((uint64_t*) out)[0] = h1;
    ((uint64_t*) out)[1] = h2;
}
