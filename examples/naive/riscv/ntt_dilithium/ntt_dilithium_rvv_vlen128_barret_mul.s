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
    vmul.vx \dst, \a, \b                // z = a*b = coefficient (vect) * root (scalar)
    vmulhu.vx tmp0, \a, \barret_const   // t = (a * barret_const) >> k
    vnmsac.vx \dst, modulus, tmp0       // z = z - n * t
.endm

.macro ct_butterfly a, b, root, barret_const
    barret_mul tmp1, b, root, barret_const                  // tmp1 = root * b
    vadd.vv \a, \a, tmp1                                    // a = a + tmp1
    vsub.vv \b, \a, tmp1                                    // b = b - tmp1
.endm

.data
.p2align 4
roots:
#include "ntt_dilithium_1234_5478_twiddles_barret_mul.s"
.text
    .global ntt_dilithium_1234_5678
    .global _ntt_dilithium_1234_5678
.p2align 4
// modulus here

ntt_dilithium_1234_5678:
_ntt_dilithium_1234_5678:
    push_stack // save regs here

    in      .req x1
    inp     .req x2
    count   .req x3
    r_ptr0  .req x4
    r_ptr1  .req x5
    xtmp    .req x6
    modulus .req x7

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

    // we could also store 2 constants in one reg if necessary
    root1       .req x8     // root1 = \psi^bitinverse(1), root2 = \psi^bitinverse(2) ...
    barretc_1   .req x9     // barretc_1 = floor((root1 << k)\modulus) ...
    root2       .req x10
    barretc_2   .req x11
    root3       .req x12
    barretc_3   .req x13
    root4       .req x14
    barretc_4   .req x15
    root5       .req x16
    barretc_5   .req x17
    root6       .req x18
    barretc_6   .req x19
    root7       .req x20
    barretc_7   .req x21
    root8       .req x22
    barretc_8   .req x23

    tmp0    .req v23  // used by barret_mul
    tmp1    .req v24  // used by ct_butterfly

    .equ L_STRIDE, 64
    .equ S_STRIDE, 16

    // other loads here?
    li count, 4

    load_roots_1234  // macro is yet missing

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

    // Merge 4 layers (interleaved)
    // level 1
    ct_butterfly data0, data8, root1, barretc_1
    ct_butterfly data1, data9, root1, barretc_1
    ct_butterfly data2, data10, root1, barretc_1
    ct_butterfly data3, data11, root1, barretc_1
    ct_butterfly data4, data12, root1, barretc_1
    ct_butterfly data5, data13, root1, barretc_1
    ct_butterfly data6, data14, root1, barretc_1
    ct_butterfly data7, data15, root1, barretc_1

    // level 2
    ct_butterfly data0, data4, root2, barretc_2
    ct_butterfly data1, data5, root2, barretc_2
    ct_butterfly data2, data6, root2, barretc_2
    ct_butterfly data3, data7, root2, barretc_2
    ct_butterfly data8, data12, root3, barretc_3
    ct_butterfly data9, data13, root3, barretc_3
    ct_butterfly data10, data14, root3, barretc_3
    ct_butterfly data11, data15, root3, barretc_3

    // level 3
    ct_butterfly data0, data2, root4, barretc_4
    ct_butterfly data1, data3, root4, barretc_4
    ct_butterfly data4, data6, root5, barretc_5
    ct_butterfly data5, data7, root5, barretc_5
    ct_butterfly data8, data10, root6, barretc_6
    ct_butterfly data9, data11, root6, barretc_6
    ct_butterfly data12, data14, root6, barretc_7
    ct_butterfly data13, data15, root7, barretc_7

    // level 4
    ct_butterfly data0, data1, root8, barretc_8
    ct_butterfly data2, data3, root9, barretc_9
    ct_butterfly data4, data5, root10, barretc_10
    ct_butterfly data6, data7, root11, barretc_11
    ct_butterfly data8, data9, root12, barretc_12
    ct_butterfly data10, data11, root13, barretc_13
    ct_butterfly data12, data13, root14, barretc_14
    ct_butterfly data14, data15, root15, barretc_15

    addi, in, in, -15*L_STRIDE  // decrement pointer to original value

    // make this a loop?
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

    addi in, in, S_STRIDE  // load next coeffient pairs, shifted by 16 bytes
layer1234_end:
    addi count, count, -1
    bnez count, layer1234_start



    //  how to const from memory to regs?
    // store twiddle + barret const interleaved in x registers
    // for layer 7-8 store in vector registers