	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	layer_norm                      # -- Begin function layer_norm
	.p2align	2
	.type	layer_norm,@function
layer_norm:                             # @layer_norm
# %bb.0:
	sext.w	a6, a4
	mulw	a5, a5, a4
	slli	a5, a5, 2
	add	a0, a0, a5
	blez	a6, .LBB0_6
# %bb.1:                                # %.lr.ph.preheader
	li	a7, 0
	fmv.w.x	fa5, zero
	vsetivli	zero, 8, e32, m1, ta, ma
	vmv.s.x	v9, zero
.LBB0_2:                                # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
	slli	t0, a7, 2
	add	t0, a0, t0
	vl1re32.v	v8, (t0)
	vfredusum.vs	v8, v8, v9
	vfmv.f.s	fa4, v8
	addiw	a7, a7, 8
	fadd.s	fa5, fa5, fa4
	blt	a7, a6, .LBB0_2
# %bb.3:                                # %.lr.ph6.preheader
	li	a7, 0
	fcvt.s.wu	fa4, a4
	fdiv.s	fa5, fa5, fa4
	vfmv.s.f	v10, fa4
	vfmv.v.f	v8, fa5
	vmv.s.x	v11, zero
.LBB0_4:                                # %.lr.ph6
                                        # =>This Inner Loop Header: Depth=1
	slli	a4, a7, 2
	add	a4, a0, a4
	vl1re32.v	v12, (a4)
	vsetivli	zero, 8, e32, m1, ta, ma
	vfsub.vv	v12, v12, v8
	vfmul.vv	v12, v12, v12
	vfredusum.vs	v12, v12, v9
	vsetivli	zero, 1, e32, mf2, ta, ma
	vrgather.vi	v13, v12, 0
	addiw	a7, a7, 8
	vfadd.vv	v11, v11, v13
	blt	a7, a6, .LBB0_4
# %bb.5:                                # %._crit_edge7
	bgtz	a6, .LBB0_7
	j	.LBB0_9
.LBB0_6:                                # %._crit_edge
	fcvt.s.w	fa5, a4
	fmv.w.x	fa4, zero
	fdiv.s	fa4, fa4, fa5
	vsetivli	zero, 8, e32, m1, ta, ma
	vfmv.s.f	v10, fa5
	vfmv.v.f	v8, fa4
	vmv.s.x	v11, zero
	blez	a6, .LBB0_9
.LBB0_7:                                # %.lr.ph10.preheader
	li	a4, 0
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfdiv.vv	v9, v11, v10
	lui	a7, 225916
	addi	a7, a7, 1452
	fmv.w.x	fa5, a7
	vfadd.vf	v9, v9, fa5
	vfsqrt.v	v9, v9
	lui	a7, 260096
	fmv.w.x	fa5, a7
	vfmv.f.s	fa4, v9
	fdiv.s	fa5, fa5, fa4
	add	a3, a3, a5
.LBB0_8:                                # %.lr.ph10
                                        # =>This Inner Loop Header: Depth=1
	slli	a5, a4, 2
	add	a7, a0, a5
	vl1re32.v	v9, (a7)
	add	a7, a1, a5
	vl1re32.v	v10, (a7)
	add	a7, a2, a5
	add	a5, a3, a5
	vl1re32.v	v11, (a7)
	vsetivli	zero, 8, e32, m1, ta, ma
	vfsub.vv	v9, v9, v8
	vfmul.vf	v9, v9, fa5
	vfmul.vv	v9, v10, v9
	vfadd.vv	v9, v11, v9
	addiw	a4, a4, 8
	vs1r.v	v9, (a5)
	blt	a4, a6, .LBB0_8
.LBB0_9:                                # %._crit_edge11
	ret
.Lfunc_end0:
	.size	layer_norm, .Lfunc_end0-layer_norm
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
