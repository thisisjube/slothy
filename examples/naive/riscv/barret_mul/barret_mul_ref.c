#include <stdint.h>
#include <stdio.h>
#include <math.h>

uint32_t barret_mul(uint32_t a, uint32_t b, uint32_t n, uint32_t k, uint64_t C) {
    uint64_t z = a*b;
    uint64_t t = (a*C) >> k;
    uint32_t result = z-n*t;
    return result;
}

uint64_t calc_const(uint32_t b, uint64_t R, uint32_t n) {
    return (b*R)/n;
}

int main() {
    uint32_t a = 179;
    uint32_t b = 123;
    uint32_t n = 13;
    uint32_t k = 32;
    uint64_t R = 1ULL << k;
    uint64_t C = calc_const(b, R, n);

    printf("Constant is %lu\n", C);
    printf("%u\n", barret_mul(a, b, n, k, C));
    return 0;
}