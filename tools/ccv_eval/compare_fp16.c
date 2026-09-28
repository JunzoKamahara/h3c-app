// Compare two contiguous-FP16 files, report relative L2, max abs diff, PSNR.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float f16_to_f32(unsigned short h) {
    unsigned sign = (h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1F;
    unsigned mant = h & 0x3FFu;
    unsigned bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp++; }
            mant &= 0x3FFu;
            bits = sign | ((127 - 15 + 1 - exp) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s a.bin b.bin\n", argv[0]); return 2; }
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[2], "rb");
    if (!fa || !fb) { fprintf(stderr, "cannot open input\n"); return 2; }
    fseek(fa, 0, SEEK_END); long sa = ftell(fa); fseek(fa, 0, SEEK_SET);
    fseek(fb, 0, SEEK_END); long sb = ftell(fb); fseek(fb, 0, SEEK_SET);
    if (sa != sb) { fprintf(stderr, "size mismatch: %ld vs %ld\n", sa, sb); return 2; }
    size_t count = (size_t)sa / 2;
    unsigned short *a = malloc(sa), *b = malloc(sb);
    fread(a, 2, count, fa); fread(b, 2, count, fb);
    double sum_sq_diff = 0, sum_sq_a = 0, max_abs_diff = 0, max_abs_a = 0;
    for (size_t i = 0; i < count; i++) {
        double va = f16_to_f32(a[i]), vb = f16_to_f32(b[i]);
        double d = va - vb;
        sum_sq_diff += d * d;
        sum_sq_a += va * va;
        if (fabs(d) > max_abs_diff) max_abs_diff = fabs(d);
        if (fabs(va) > max_abs_a) max_abs_a = fabs(va);
    }
    double relative_l2 = sqrt(sum_sq_diff / (sum_sq_a > 1e-30 ? sum_sq_a : 1e-30));
    double rmse = sqrt(sum_sq_diff / count);
    double psnr = 20.0 * log10(max_abs_a / (rmse > 1e-12 ? rmse : 1e-12));
    printf("count=%zu relative_l2=%.8g max_abs_diff=%.6g max_abs_a=%.6g rmse=%.6g psnr_db=%.2f\n",
           count, relative_l2, max_abs_diff, max_abs_a, rmse, psnr);
    return 0;
}
