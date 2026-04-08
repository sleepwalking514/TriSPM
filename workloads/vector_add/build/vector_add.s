	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	vector_add                      # -- Begin function vector_add
	.p2align	2
	.type	vector_add,@function
vector_add:                             # @vector_add
# %bb.0:
	slliw	a3, a3, 6
	vsetvli	a4, zero, e32, m8, ta, mu
	vid.v	v16
	lui	a4, 1
	vmv.v.i	v8, 0
	vor.vx	v16, v16, a3
	slli	a3, a3, 2
	vmslt.vx	v0, v16, a4
	vmv.v.i	v16, 0
	add	a0, a0, a3
	add	a1, a1, a3
	vle32.v	v16, (a0), v0.t
	vle32.v	v8, (a1), v0.t
	vfadd.vv	v8, v16, v8
	add	a2, a2, a3
	vse32.v	v8, (a2), v0.t
	ret
.Lfunc_end0:
	.size	vector_add, .Lfunc_end0-vector_add
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
