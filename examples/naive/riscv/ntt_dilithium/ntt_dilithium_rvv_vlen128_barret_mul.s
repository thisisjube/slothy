///
/// Copyright (c) 2025 Justus Bergermann
/// SPDX-License-Identifier: MIT
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
///

.macro barret_mul dst, a, b, barret_const
    vmul.vx \dst, \a, \b                            // z = a*b = coefficient (vect) * root (scalar)
    vmulhu.vx vtmp0, \a, \barret_const              // t = (a * barret_const) >> k
    vnmsac.vx \dst, modulus, vtmp0                  // z = z - n * t
.endm

.macro ct_butterfly a, b, root, barret_const
    barret_mul vtmp1, b, root, barret_const         // vtmp1 = root * b mod modulus
    vsub.vv \b, \a, vtmp1                           // b = b - vtmp1
    vadd.vv \a, \a, vtmp1                           // a = a + vtmp1
.endm

.macro barret_mul_v vdst, va, vb, vbarret_const
    vmul.vv \vdst, \va, \vb                         // z = a*b = coefficient (vect) * root (scalar)
    vmulhu.vv vtmp0, \va, \vbarret_const            // t = (a * barret_const) >> k
    vnmsac.vx \vdst, vmodulus, vtmp0                // z = z - n * t
.endm

.macro ct_butterfly_v va, vb, vroot, vbarret_const
    barret_mul_v vtmp1, vb, vroot, vbarret_const    // vtmp1 = vroot * vb mod modulus
    vsub.vv \vb, \va, vtmp1                         // b = b - vtmp1
    vadd.vv \va, \va, vtmp1                         // a = a + vtmp1
.endm

.macro transpose4 data, ptr
    vsseg4e32.v \data, (\ptr)
    vl4re32.v \data, (\ptr)  // loads data into data0, data1, data2, data3
    // alternative:
    // vle32.v data0, ptr
    // vle32.v data1, ptr+1 etc.
.endm

.macro load_roots_1234  root_ptr, barretc_1, barretc_2, barretc_3, barretc_4, barretc_5,
                        root6, barretc_6, root7, barretc_7, root8, barretc_8, root9, barretc_9,
                        root10, barretc_10, root11, barretc_11, root12, barretc_12, root13,
                        barretc_13, root14, barretc_14, root15, barretc_15
    lw \barretc_1,  (0*8+4)(\root_ptr)
    lw \barretc_2,  (1*8+4)(\root_ptr)
    lw \barretc_3,  (2*8+4)(\root_ptr)
    lw \barretc_4,  (3*8+4)(\root_ptr)
    lw \barretc_5,  (4*8+4)(\root_ptr)
    lw \root6,      (5*8)(\root_ptr)
    lw \barretc_6,  (5*8+4)(\root_ptr)
    lw \root7,      (6*8)(\root_ptr)
    lw \barretc_7,  (6*8+4)(\root_ptr)
    lw \root8,      (7*8)(\root_ptr)
    lw \barretc_8,  (7*8+4)(\root_ptr)
    lw \root9,      (8*8)(\root_ptr)
    lw \barretc_9,  (8*8+4)(\root_ptr)
    lw \root10,     (9*8)(\root_ptr)
    lw \barretc_10, (9*8+4)(\root_ptr)
    lw \root11,     (10*8)(\root_ptr)
    lw \barretc_11, (10*8+4)(\root_ptr)
    lw \root12,     (11*8)(\root_ptr)
    lw \barretc_12, (11*8+4)(\root_ptr)
    lw \root13,     (12*8)(\root_ptr)
    lw \barretc_13, (12*8+4)(\root_ptr)
    lw \root14,     (13*8)(\root_ptr)
    lw \barretc_14, (13*8+4)(\root_ptr)
    lw \root15,     (14*8)(\root_ptr)
    lw \barretc_15, (14*8+4)(\root_ptr)
.endm


.macro load_roots_5678  xroot1, xbarretc_1, xroot2, xbarretc_2, xroot3, xbarretc_3,
                        vroot1, vbarretc_1, vroot2, vbarretc_2,  vroot3, vbarretc_3,
                        root_ptr
    lw \xroot1,     (0*8)(\root_ptr)
    lw \xbarretc_1, (0*8+4)(\root_ptr)
    lw \xroot2,     (1*8)(\root_ptr)
    lw \xbarretc_2, (1*8+4)(\root_ptr)
    lw \xroot3,     (2*8)(\root_ptr)
    lw \xbarretc_3  (2*8+4)(\root_ptr)
    addi \root_ptr, \root_ptr, 3*8  // increment root_ptr manually here, bc vle32 does not allow offsets
    vle32.v \vroot1, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
    vle32.v \vbarretc_1, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
    vle32.v \vroot2, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
    vle32.v \vbarretc_2, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
    vle32.v \vroot3, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
    vle32.v \vbarretc_3, (\root_ptr)
    addi \root_ptr, \root_ptr, 16
.endm

.macro push_stack
    addi sp, sp, -8*15
    sd s0,  0*8(sp)
    sd s1,  1*8(sp)
    sd s2,  2*8(sp)
    sd s3,  3*8(sp)
    sd s4,  4*8(sp)
    sd s5,  5*8(sp)
    sd s6,  6*8(sp)
    sd s7,  7*8(sp)
    sd s8,  8*8(sp)
    sd s9,  9*8(sp)
    sd s10, 10*8(sp)
    sd s11, 11*8(sp)
    sd gp,  12*8(sp)
    sd tp,  13*8(sp)
    sd ra,  14*8(sp)

.macro pop_stack
    ld s0,  0*8(sp)
    ld s1,  1*8(sp)
    ld s2,  2*8(sp)
    ld s3,  3*8(sp)
    ld s4,  4*8(sp)
    ld s5,  5*8(sp)
    ld s6,  6*8(sp)
    ld s7,  7*8(sp)
    ld s8,  8*8(sp)
    ld s9,  9*8(sp)
    ld s10, 10*8(sp)
    ld s11, 11*8(sp)
    ld gp,  12*8(sp)
    ld tp,  13*8(sp)
    ld ra,  14*8(sp)
    addi sp, sp, 8*15
.data

.p2align 4
roots:
#include "ntt_dilithium_1234_5478_twiddles_barret_mul.s"
.text
    .global ntt_dilithium_1234_5678
    .global _ntt_dilithium_1234_5678
.p2align 4

ntt_dilithium_1234_5678:
_ntt_dilithium_1234_5678:
    push_stack // save scalar regs here

    in          .req x1
    count       .req x3
    modulus     .req x4
    root_ptr    .req x5
    xtmp        .req x6

    data0   .req v0
    data1   .req v1
    data2   .req v2
    data3   .req v3
    data4   .req v4
    data5   .req v5
    data6   .req v6
    data7   .req v7
    data8   .req v8
    data9   .req v9
    data10  .req v10
    data11  .req v11
    data12  .req v12
    data13  .req v13
    data14  .req v14
    data15  .req v15

    // load first 5 roots only on demand due to limited number of registers
    barretc_1   .req x7     // root1 = \psi^bitinverse(1), root2 = \psi^bitinverse(2) ...
    barretc_2   .req x8     // barretc_1 = floor((root1 << k)\modulus) ...
    barretc_3   .req x9
    barretc_4   .req x10
    barretc_5   .req x11
    root6       .req x12
    barretc_6   .req x13
    root7       .req x14
    barretc_7   .req x15
    root8       .req x16
    barretc_8   .req x17
    root9       .req x18
    barretc_9   .req x19
    root10      .req x20
    barretc_10  .req x21
    root11      .req x22
    barretc_11  .req x23
    root12      .req x24
    barretc_12  .req x25
    root13      .req x26
    barretc_13  .req x27
    root14      .req x28
    barretc_14  .req x29
    root15      .req x30
    barretc_15  .req x31

    vtmp0    .req v16  // used by barret_mul
    vtmp1    .req v17  // used by ct_butterfly
    vtmp2    .req v18  // free to use
    vmodulus .req v19  // vectorized modulus

    li modulus, 8380417  // load dilithium modulus. Get modulus from memory in future?
    vmv.v.x vmodulus, modulus // copy modulus into vector register

    .equ L_STRIDE, 64  // Load Stride = distance between coefficients pairs of four
    .equ S_STRIDE, 16  // Shift Stride = distance of coefficients between two iterations

    la root_ptr, roots  // load address of roots in memory into root_ptr

    load_roots_1234 root_ptr, barretc_1, barretc_2, barretc_3, barretc_4, barretc_5,
    root6, barretc_6, root7, barretc_7, root8, barretc_8, root9, barretc_9, root10, barretc_10,
    root11, barretc_11, root12, barretc_12, root13, barretc_13, root14, barretc_14, root15, barretc_15

    li count, 4

    .p2align 2
layer1234_start:
    // Load 64 coefficients. For VLEN = 128 each register holds 4*4 byte = 32 bit coefficients. Hence, 16 regs required
    // Base register must be incremented by L_STRIDE = 64 byte to load the correct coefficient pairs.

    // make this a loop?
    vle32.v data0, (in)
    addi in, in, L_STRIDE
    vle32.v data1, (in)
    addi in, in, L_STRIDE
    vle32.v data2, (in)
    addi in, in, L_STRIDE
    vle32.v data3, (in)
    addi in, in, L_STRIDE
    vle32.v data4, (in)
    addi in, in, L_STRIDE
    vle32.v data5, (in)
    addi in, in, L_STRIDE
    vle32.v data6, (in)
    addi in, in, L_STRIDE
    vle32.v data7, (in)
    addi in, in, L_STRIDE
    vle32.v data8, (in)
    addi in, in, L_STRIDE
    vle32.v data9, (in)
    addi in, in, L_STRIDE
    vle32.v data10, (in)
    addi in, in, L_STRIDE
    vle32.v data11, (in)
    addi in, in, L_STRIDE
    vle32.v data12, (in)
    addi in, in, L_STRIDE
    vle32.v data13, (in)
    addi in, in, L_STRIDE
    vle32.v data14, (in)
    addi in, in, L_STRIDE
    vle32.v data15, (in)

    lw xtmp, 0*8(root_ptr)  // xtmp = root1

    // Merge 4 layers (interleaved)
    // level 1 - Stride = 128*32 byte -> BF(a0,a128), BF(a16, a144) ...
    ct_butterfly data0, data8, xtmp, barretc_1
    ct_butterfly data1, data9, xtmp, barretc_1
    ct_butterfly data2, data10, xtmp, barretc_1
    ct_butterfly data3, data11, xtmp, barretc_1
    ct_butterfly data4, data12, xtmp, barretc_1
    ct_butterfly data5, data13, xtmp, barretc_1
    ct_butterfly data6, data14, xtmp, barretc_1
    ct_butterfly data7, data15, xtmp, barretc_1

    lw xtmp, 1*8(root_ptr)  // xtmp = root2

    // level 2 - Stride = 64*32 byte -> BF(a0, a64), BF(16, 80) ...
    ct_butterfly data0, data4, xtmp, barretc_2
    ct_butterfly data1, data5, xtmp, barretc_2
    ct_butterfly data2, data6, xtmp, barretc_2
    ct_butterfly data3, data7, xtmp, barretc_2

    lw xtmp, 2*8(root_ptr)  // xtmp = root3

    ct_butterfly data8, data12, xtmp, barretc_3
    ct_butterfly data9, data13, xtmp, barretc_3
    ct_butterfly data10, data14, xtmp, barretc_3
    ct_butterfly data11, data15, xtmp, barretc_3

    lw xtmp, 3*8(root_ptr)  // xtmp = root4

    // level 3 - Stride = 32*32 byte -> BF(a0, a32), BF(a16, a48) ...
    ct_butterfly data0, data2, xtmp, barretc_4
    ct_butterfly data1, data3, xtmp, barretc_4

    lw xtmp, 4*8(root_ptr)  // xtmp = root5

    ct_butterfly data4, data6, xtmp, barretc_5
    ct_butterfly data5, data7, xtmp, barretc_5
    ct_butterfly data8, data10, root6, barretc_6
    ct_butterfly data9, data11, root6, barretc_6
    ct_butterfly data12, data14, root6, barretc_7
    ct_butterfly data13, data15, root7, barretc_7

    // level 4 - Stride = 16*32 byte -> BF(a0, a16), BF(a32, a48) ...
    ct_butterfly data0, data1, root8, barretc_8
    ct_butterfly data2, data3, root9, barretc_9
    ct_butterfly data4, data5, root10, barretc_10
    ct_butterfly data6, data7, root11, barretc_11
    ct_butterfly data8, data9, root12, barretc_12
    ct_butterfly data10, data11, root13, barretc_13
    ct_butterfly data12, data13, root14, barretc_14
    ct_butterfly data14, data15, root15, barretc_15

    addi in, in, -15*L_STRIDE  // decrement pointer to original value

    // make this a loop?
    // store results
    vse32.v data0, (in)
    addi in, in, L_STRIDE
    vse32.v data1, (in)
    addi in, in, L_STRIDE
    vse32.v data2, (in)
    addi in, in, L_STRIDE
    vse32.v data3, (in)
    addi in, in, L_STRIDE
    vse32.v data4, (in)
    addi in, in, L_STRIDE
    vse32.v data5, (in)
    addi in, in, L_STRIDE
    vse32.v data6, (in)
    addi in, in, L_STRIDE
    vse32.v data7, (in)
    addi in, in, L_STRIDE
    vse32.v data8, (in)
    addi in, in, L_STRIDE
    vse32.v data9, (in)
    addi in, in, L_STRIDE
    vse32.v data10, (in)
    addi in, in, L_STRIDE
    vse32.v data11, (in)
    addi in, in, L_STRIDE
    vse32.v data12, (in)
    addi in, in, L_STRIDE
    vse32.v data13, (in)
    addi in, in, L_STRIDE
    vse32.v data14, (in)
    addi in, in, L_STRIDE
    vse32.v data15, (in)

    addi in, in, S_STRIDE  // load next coeffient pairs, all shifted by 16 bytes
layer1234_end:
    addi count, count, -1
    bnez count, layer1234_start

    .unreq barretc_1
    .unreq barretc_2
    .unreq barretc_3
    .unreq barretc_4
    .unreq barretc_5
    .unreq root6

    xroot1      .req x7
    xbarretc_1  .req x8
    xroot2      .req x9
    xbarretc_2  .req x10
    xroot3      .req x11
    xbarretc_3  .req x12

    vroot1      .req v19
    vbarretc_1  .req v20
    vroot2      .req v21
    vbarretc2   .req v22
    vroot3      .req v23
    vbarretc_3  .req v24
    addi in, in, -4*S_STRIDE    // reset in pointer to original value, has been updated 4 x S_STRIDE in the previous loop
                                // other implementation saved original in to stack, maybe consider that ...
    li count, 16
    addi root_ptr, root_ptr, 15*8 // point to twiddles for layer5679, starting with root16

    .equ L_STRIDE, 16
    .equ S_STRIDE, 64  // check again

    .p2align 2
layer5678_start:
    vle32.v data0, (in)
    addi in, in, L_STRIDE
    vle32.v data1, (in)
    addi in, in, L_STRIDE
    vle32.v data2, (in)
    addi in, in, L_STRIDE
    vle32.v data3, (in)

    addi in, in, -3*L_STRIDE  // decrement pointer to original value

    load_roots_5678 xroot1, xbarretc_1, xroot2, xbarretc_2, xroot3, xbarretc_3,
                         vroot1, vbarretc_1, vroot2, vbarretc_2, vroot3, vbarretc_3,
                         root_ptr

    // level 5+6
    ct_butterfly data0, data2, xroot1, xbarretc_1  // Stride = 8*32 byte -> BF(a0, a8), BF(a4, a12) ...
    ct_butterfly data1, data3, xroot1, xbarretc_1
    ct_butterfly data0, data1, xroot2, xbarretc_2  // Stride = 4*32 byte -> BF(a0, a4), BF(a8, a12) ...
    ct_butterfly data2, data3, xroot3, xbarretc_3

    sub sp, sp, 64  // allocate 64 byte of memory for 4 vector registers
    transpose4 data0, sp  // necessary bc coefficients required for ct_butterfly would be in same regs otherwise
    add sp, sp, 64  // free memory

    // level 7+8
    ct_butterfly_v data0, data2, vroot1, vbarretc_1  // Stride = 2*32 byte -> BF(a0, a2), BF(a4, a6) ...
    ct_butterfly_v data1, data3, vroot1, vbarretc_1
    ct_butterfly_v data0, data1, vroot2, vbarretc_2  // Stride = 1*32 byte -> BF(a0, a1), BF(a2, a3) ...
    ct_butterfly_v data2, data3, vroot3, vbarretc_3

    // store results, transpose back before. Corresponds to https://fprox.substack.com/i/139455473/x-matrix-transpose-using-strided-vector-stores
    vsseg4e32.v data0, in  // Store packed vector of 4*4-byte segments from data0, data1, data2, data3 to memory

    // alternative store, might be faster. Corresponds to https://fprox.substack.com/i/139455473/x-matrix-transpose-using-segmented-vector-stores
    // Benchmarks: https://pastebin.com/gQB76kgy
    //li vtmp2, L_STRIDE
    //vsse32.v data0, (in), vtmp2
    //addi in, in, L_STRIDE
    //vsse32.v data1, (in), vtmp2
    //addi in, in, L_STRIDE
    //vsse32.v data2, (in), vtmp2
    //addi in, in, L_STRIDE
    //vsse32.v data3, (in), vtmp2
    // addi in, in, -3*L_STRIDE  // decrement pointer to original value

    addi in, in, S_STRIDE  // load next coeffient pairs, all shifted by 64 bytes
layer5678_end:
    addi count, count, -1
    bnez count, layer5678_start

    pop_stack
    ret

// TODOs:
// - figure out what roots + constants to load for layer5678 and how to store them in memory (vectors required for layer 7-8)
// - check how to load roots + constants from memory to registers
// - save regs where necessary
// - make big load/ store sequences macro, loop or something
// - check pointer inc/ decr and move right after
// documentation
    // correct comments in BF byte/ bit