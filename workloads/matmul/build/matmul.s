	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	matmul                          # -- Begin function matmul
	.p2align	2
	.type	matmul,@function
matmul:                                 # @matmul
# %bb.0:
	addi	sp, sp, -544
	sd	ra, 536(sp)                     # 8-byte Folded Spill
	sd	s0, 528(sp)                     # 8-byte Folded Spill
	sd	s1, 520(sp)                     # 8-byte Folded Spill
	sd	s2, 512(sp)                     # 8-byte Folded Spill
	sd	s3, 504(sp)                     # 8-byte Folded Spill
	sd	s4, 496(sp)                     # 8-byte Folded Spill
	sd	s5, 488(sp)                     # 8-byte Folded Spill
	sd	s6, 480(sp)                     # 8-byte Folded Spill
	sd	s7, 472(sp)                     # 8-byte Folded Spill
	sd	s8, 464(sp)                     # 8-byte Folded Spill
	sd	s9, 456(sp)                     # 8-byte Folded Spill
	sd	s10, 448(sp)                    # 8-byte Folded Spill
	sd	s11, 440(sp)                    # 8-byte Folded Spill
	fsd	fs0, 432(sp)                    # 8-byte Folded Spill
	fsd	fs1, 424(sp)                    # 8-byte Folded Spill
	fsd	fs2, 416(sp)                    # 8-byte Folded Spill
	fsd	fs3, 408(sp)                    # 8-byte Folded Spill
	fsd	fs4, 400(sp)                    # 8-byte Folded Spill
	fsd	fs5, 392(sp)                    # 8-byte Folded Spill
	fsd	fs6, 384(sp)                    # 8-byte Folded Spill
	fsd	fs7, 376(sp)                    # 8-byte Folded Spill
	fsd	fs8, 368(sp)                    # 8-byte Folded Spill
	fsd	fs9, 360(sp)                    # 8-byte Folded Spill
	fsd	fs10, 352(sp)                   # 8-byte Folded Spill
	fsd	fs11, 344(sp)                   # 8-byte Folded Spill
	addi	sp, sp, -256
	sd	a2, 40(sp)                      # 8-byte Folded Spill
	sraiw	a2, a3, 31
	srliw	a7, a2, 28
	add	a7, a3, a7
	sraiw	a2, a7, 4
	slli	a2, a2, 2
	li	a4, 4
	sub	a6, a4, a2
	blt	a6, a4, .LBB0_2
# %bb.1:
	li	a6, 4
.LBB0_2:
	li	s6, 0
	li	ra, 0
	remw	a4, a3, a6
	andi	a7, a7, -16
	vsetivli	zero, 16, e32, m2, ta, ma
	vid.v	v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.s.x	v8, a0
	vmv.s.x	v12, a1
	li	t4, 256
	li	t3, 512
	li	t2, 768
	li	t1, 1024
	li	t0, 1280
	li	a1, 1536
	li	a0, 1792
	li	s2, 1
	li	s1, 9
	li	s0, 5
	li	t6, 11
	li	t5, 3
	li	s3, 13
	li	s4, 7
	li	s5, 15
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v30, 0
	sub	a3, a3, a7
	slli	s2, s2, 11
	slli	s1, s1, 8
	slli	s0, s0, 9
	divw	s7, a3, a6
	slli	t6, t6, 8
	slli	t5, t5, 10
	slli	a7, s3, 8
	slli	a6, s4, 9
	slli	a3, s5, 8
	addi	a5, sp, 400
	vs2r.v	v30, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 464
	vs2r.v	v30, (a5)                       # vscale x 16-byte Folded Spill
	add	a2, a4, a2
	slli	a4, a2, 4
	slliw	a5, a2, 10
	vor.vx	v14, v10, a4
	slliw	a2, s7, 4
	sd	a5, 24(sp)                      # 8-byte Folded Spill
	vmv.s.x	v10, a5
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v11, v14, 1
	vslidedown.vi	v18, v14, 2
	vslidedown.vi	v19, v14, 3
	vslidedown.vi	v22, v14, 4
	vslidedown.vi	v23, v14, 5
	vslidedown.vi	v24, v14, 6
	vslidedown.vi	v25, v14, 7
	vmv.x.s	a4, v15
	vslidedown.vi	v26, v15, 1
	vslidedown.vi	v27, v15, 2
	vslidedown.vi	v28, v15, 3
	vslidedown.vi	v7, v15, 4
	vslidedown.vi	v6, v15, 5
	vslidedown.vi	v13, v15, 6
	vslidedown.vi	v9, v15, 7
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	vmv.s.x	v14, a2
	vsetivli	zero, 16, e32, m2, ta, ma
	vsll.vi	v16, v10, 2
	vmv.x.s	s5, v11
	vmv.x.s	s7, v18
	vmv.x.s	s8, v19
	vmv2r.v	v20, v30
	vmv.x.s	s9, v22
	vmv.x.s	s10, v23
	vmv.x.s	s4, v24
	vmv.x.s	a2, v25
	slli	s3, a4, 6
	vmv.x.s	a4, v26
	vmv.s.x	v18, s3
	vmv.x.s	s3, v27
	vsll.vi	v14, v14, 2
	vadd.vx	v22, v14, t4
	vmv.x.s	t4, v28
	vwadd.wv	v24, v8, v16
	vadd.vx	v16, v14, t3
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a5, v24
	sd	a5, 224(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v28, v14, t2
	vwadd.wv	v24, v12, v14
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a5, v24
	sd	a5, 216(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v30, v14, t1
	vwadd.wv	v24, v12, v22
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a5, v24
	sd	a5, 208(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v22, v14, t0
	vwadd.wv	v24, v12, v16
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a5, v24
	sd	a5, 200(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v16, v14, a1
	vwadd.wv	v24, v12, v28
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a1, v24
	sd	a1, 192(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v28, v14, a0
	vwadd.wv	v24, v12, v30
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v30, v14, s2
	vwadd.wv	v24, v12, v22
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v22, v14, s1
	vwadd.wv	v24, v12, v16
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v16, v14, s0
	vwadd.wv	v24, v12, v28
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v28, v14, t6
	vwadd.wv	v24, v12, v30
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v4, v14, t5
	vwadd.wv	v24, v12, v22
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vadd.vx	v22, v14, a7
	vmv.x.s	a0, v7
	vadd.vx	v10, v14, a6
	vmv.x.s	a1, v6
	slli	s5, s5, 6
	slli	s7, s7, 6
	slli	s8, s8, 6
	slli	s9, s9, 6
	slli	s10, s10, 6
	vadd.vx	v30, v14, a3
	vwadd.wv	v24, v12, v16
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a3, v24
	sd	a3, 136(sp)                     # 8-byte Folded Spill
	vmv.s.x	v6, s5
	vsetvli	zero, zero, e32, m2, ta, ma
	vwadd.wv	v24, v12, v28
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a3, v24
	sd	a3, 128(sp)                     # 8-byte Folded Spill
	vmv.s.x	v16, s7
	vsetvli	zero, zero, e32, m2, ta, ma
	vwadd.wv	v24, v12, v4
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a3, v24
	sd	a3, 120(sp)                     # 8-byte Folded Spill
	vmv.s.x	v28, s8
	vsetvli	zero, zero, e32, m2, ta, ma
	vwadd.wv	v24, v12, v22
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a3, v24
	sd	a3, 112(sp)                     # 8-byte Folded Spill
	vmv.s.x	v26, s9
	vsetvli	zero, zero, e32, m2, ta, ma
	vwadd.wv	v0, v12, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a3, v0
	sd	a3, 104(sp)                     # 8-byte Folded Spill
	vmv.s.x	v24, s10
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.x.s	a3, v13
	slli	s4, s4, 6
	vmv.s.x	v4, s4
	vmv.x.s	a6, v9
	vmv2r.v	v22, v20
	slli	a2, a2, 6
	slli	a4, a4, 6
	slli	s3, s3, 6
	slli	t4, t4, 6
	slli	a0, a0, 6
	slli	a1, a1, 6
	slli	a3, a3, 6
	slli	a6, a6, 6
	vwadd.wv	v12, v12, v30
	vmv.s.x	v10, a2
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a2, v12
	sd	a2, 96(sp)                      # 8-byte Folded Spill
	vmv.s.x	v12, a4
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v14, v18, 2
	vwadd.wv	v0, v8, v14
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a2, v0
	sd	a2, 88(sp)                      # 8-byte Folded Spill
	vmv.s.x	v14, s3
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v30, v6, 2
	vwadd.wv	v0, v8, v30
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a2, v0
	sd	a2, 80(sp)                      # 8-byte Folded Spill
	vmv.s.x	v18, t4
	addi	a2, sp, 528
	vs2r.v	v18, (a2)                       # vscale x 16-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v16, v16, 2
	vwadd.wv	v0, v8, v16
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a2, v0
	sd	a2, 72(sp)                      # 8-byte Folded Spill
	vmv.s.x	v6, a0
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v16, v28, 2
	vwadd.wv	v0, v8, v16
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v0
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	vmv.s.x	v30, a1
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v26, v26, 2
	vwadd.wv	v0, v8, v26
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v0
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	vmv.s.x	v16, a3
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v28, v24, 2
	vwadd.wv	v24, v8, v28
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a0, v24
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	vmv.s.x	v18, a6
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v28, v4, 2
	vwadd.wv	v24, v8, v28
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s0, v24
	vmv2r.v	v4, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v10, 2
	vwadd.wv	v24, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	t6, v24
	vmv2r.v	v2, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v12, 2
	vwadd.wv	v24, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	t4, v24
	vmv2r.v	v24, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v14, 2
	vwadd.wv	v12, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	t3, v12
	vmv2r.v	v0, v20
	addi	a0, sp, 528
	vl2r.v	v10, (a0)                       # vscale x 16-byte Folded Reload
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v10, 2
	vwadd.wv	v12, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s7, v12
	vmv2r.v	v26, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v6, 2
	vwadd.wv	v12, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s8, v12
	vmv2r.v	v28, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v30, 2
	vwadd.wv	v12, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s9, v12
	vmv2r.v	v30, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v10, v16, 2
	vwadd.wv	v12, v8, v10
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s11, v12
	vmv2r.v	v6, v20
	vmv2r.v	v14, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vsll.vi	v12, v18, 2
	vwadd.wv	v8, v8, v12
	vmv2r.v	v10, v20
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	s10, v8
	vmv2r.v	v12, v20
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	vmv2r.v	v8, v20
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	ld	a6, 216(sp)                     # 8-byte Folded Reload
	add	a6, a6, ra
	add	a3, t3, s6
	add	a2, t4, s6
	ld	a4, 88(sp)                      # 8-byte Folded Reload
	add	a4, a4, s6
	add	a1, t6, s6
	add	a0, s0, s6
	ld	t5, 48(sp)                      # 8-byte Folded Reload
	add	t5, t5, s6
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	add	s1, s1, s6
	ld	s2, 64(sp)                      # 8-byte Folded Reload
	add	s2, s2, s6
	ld	s3, 72(sp)                      # 8-byte Folded Reload
	add	s3, s3, s6
	ld	s4, 80(sp)                      # 8-byte Folded Reload
	add	s4, s4, s6
	ld	s5, 224(sp)                     # 8-byte Folded Reload
	add	s5, s5, s6
	vl2re32.v	v18, (a6)
	flw	fa5, 0(s5)
	flw	fa4, 4(s5)
	fsw	fa4, 324(sp)                    # 4-byte Folded Spill
	flw	fa4, 8(s5)
	fsw	fa4, 316(sp)                    # 4-byte Folded Spill
	flw	fa4, 12(s5)
	fsw	fa4, 320(sp)                    # 4-byte Folded Spill
	flw	fa4, 0(s4)
	flw	fs6, 4(s4)
	flw	ft10, 8(s4)
	flw	fa3, 12(s4)
	fsw	fa3, 312(sp)                    # 4-byte Folded Spill
	flw	fa3, 0(s3)
	flw	fs7, 4(s3)
	flw	fs0, 8(s3)
	flw	fa2, 12(s3)
	fsw	fa2, 304(sp)                    # 4-byte Folded Spill
	flw	fa2, 0(s2)
	flw	fs8, 4(s2)
	flw	fs1, 8(s2)
	flw	fa1, 12(s2)
	fsw	fa1, 292(sp)                    # 4-byte Folded Spill
	flw	fa1, 0(s1)
	flw	fs9, 4(s1)
	flw	fs2, 8(s1)
	flw	fa0, 12(s1)
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	flw	ft0, 0(t5)
	flw	fs10, 4(t5)
	flw	fs3, 8(t5)
	flw	fa0, 12(t5)
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	flw	ft1, 0(a0)
	flw	fs11, 4(a0)
	flw	fs4, 8(a0)
	flw	fa0, 12(a0)
	fsw	fa0, 284(sp)                    # 4-byte Folded Spill
	flw	ft2, 0(a1)
	flw	ft8, 4(a1)
	flw	fs5, 8(a1)
	flw	fa0, 12(a1)
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmacc.vf	v16, fa5, v18
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 400
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fa4, v18
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 464
	vl2r.v	v20, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v20, fa3, v18
	vfmacc.vf	v22, fa2, v18
	flw	fa5, 0(a4)
	flw	ft6, 4(a4)
	flw	ft9, 8(a4)
	flw	fa4, 12(a4)
	fsw	fa4, 276(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v4, fa1, v18
	vfmacc.vf	v2, ft0, v18
	vfmacc.vf	v24, ft1, v18
	vfmacc.vf	v0, ft2, v18
	flw	fa4, 0(a2)
	flw	ft4, 4(a2)
	flw	fa7, 8(a2)
	flw	fa3, 12(a2)
	fsw	fa3, 272(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v26, fa5, v18
	vfmacc.vf	v28, fa4, v18
	flw	fa4, 0(a3)
	flw	ft1, 4(a3)
	flw	fa6, 8(a3)
	flw	fa5, 12(a3)
	fsw	fa5, 268(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v30, fa4, v18
	add	a6, s7, s6
	flw	fa4, 0(a6)
	flw	ft0, 4(a6)
	flw	ft7, 8(a6)
	flw	fa5, 12(a6)
	fsw	fa5, 264(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v6, fa4, v18
	add	a7, s8, s6
	flw	fa4, 0(a7)
	flw	fa1, 4(a7)
	flw	ft5, 8(a7)
	flw	fa5, 12(a7)
	fsw	fa5, 260(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v14, fa4, v18
	add	t0, s9, s6
	flw	fa4, 0(t0)
	flw	fa2, 4(t0)
	flw	ft3, 8(t0)
	flw	fa5, 12(t0)
	fsw	fa5, 256(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v10, fa4, v18
	add	t1, s11, s6
	flw	ft11, 0(t1)
	flw	fa3, 4(t1)
	flw	ft2, 8(t1)
	flw	fa5, 12(t1)
	fsw	fa5, 252(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v12, ft11, v18
	ld	a5, 208(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	add	t2, s10, s6
	vl2re32.v	v16, (a5)
	flw	fa5, 0(t2)
	flw	fa4, 4(t2)
	flw	fa0, 8(t2)
	flw	ft11, 12(t2)
	vfmacc.vf	v8, fa5, v18
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	addi	a0, sp, 528
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 324(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa5, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 400
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v18, fs6, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fs7, v16
	addi	a0, sp, 464
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fs8, v16
	vfmacc.vf	v4, fs9, v16
	vfmacc.vf	v2, fs10, v16
	vfmacc.vf	v24, fs11, v16
	vfmacc.vf	v0, ft8, v16
	vfmacc.vf	v26, ft6, v16
	vfmacc.vf	v28, ft4, v16
	vfmacc.vf	v30, ft1, v16
	vfmacc.vf	v6, ft0, v16
	vfmacc.vf	v14, fa1, v16
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, fa3, v16
	vfmacc.vf	v8, fa4, v16
	ld	a5, 200(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v18, (a5)
	ld	a5, 192(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v16, (a5)
	ld	a5, 184(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	addi	a0, sp, 528
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 316(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 400
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v20, ft10, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v20, fs0, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	vfmacc.vf	v22, fs1, v18
	vfmacc.vf	v4, fs2, v18
	vfmacc.vf	v2, fs3, v18
	vfmacc.vf	v24, fs4, v18
	vfmacc.vf	v0, fs5, v18
	vfmacc.vf	v26, ft9, v18
	vfmacc.vf	v28, fa7, v18
	vfmacc.vf	v30, fa6, v18
	vfmacc.vf	v6, ft7, v18
	vfmacc.vf	v14, ft5, v18
	vfmacc.vf	v10, ft3, v18
	vfmacc.vf	v12, ft2, v18
	vfmacc.vf	v8, fa0, v18
	vl2re32.v	v18, (a5)
	addi	a5, sp, 336
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	fs2, 16(s5)
	flw	fa5, 20(s5)
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
	flw	fa5, 24(s5)
	fsw	fa5, 244(sp)                    # 4-byte Folded Spill
	flw	fa5, 28(s5)
	fsw	fa5, 324(sp)                    # 4-byte Folded Spill
	flw	fa5, 16(s4)
	flw	fs6, 20(s4)
	flw	fa4, 24(s4)
	fsw	fa4, 240(sp)                    # 4-byte Folded Spill
	flw	fa4, 28(s4)
	fsw	fa4, 316(sp)                    # 4-byte Folded Spill
	flw	fa4, 16(s3)
	flw	fs7, 20(s3)
	flw	fs4, 24(s3)
	flw	fa3, 28(s3)
	fsw	fa3, 308(sp)                    # 4-byte Folded Spill
	flw	fa3, 16(s2)
	flw	fs8, 20(s2)
	flw	fs5, 24(s2)
	flw	fa2, 28(s2)
	fsw	fa2, 300(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 320(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 400
	vl2r.v	v20, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 312(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa2, v16
	addi	a5, sp, 464
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 304(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	fa2, 292(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v22, fa2, v16
	flw	fa1, 16(s1)
	flw	fs9, 20(s1)
	flw	fs3, 24(s1)
	flw	fa2, 28(s1)
	fsw	fa2, 292(sp)                    # 4-byte Folded Spill
	flw	fa2, 296(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v4, fa2, v16
	flw	fa2, 288(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v2, fa2, v16
	flw	fa2, 284(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v24, fa2, v16
	flw	fa2, 280(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v0, fa2, v16
	flw	fa0, 16(t5)
	flw	fs10, 20(t5)
	flw	fs1, 24(t5)
	flw	fa2, 28(t5)
	fsw	fa2, 288(sp)                    # 4-byte Folded Spill
	flw	fa2, 276(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v26, fa2, v16
	flw	fa2, 272(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v28, fa2, v16
	flw	fa2, 268(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v30, fa2, v16
	flw	fa2, 264(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v6, fa2, v16
	flw	ft1, 16(a0)
	flw	fs11, 20(a0)
	flw	fs0, 24(a0)
	flw	fa2, 28(a0)
	fsw	fa2, 284(sp)                    # 4-byte Folded Spill
	flw	fa2, 260(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa2, v16
	flw	fa2, 256(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v10, fa2, v16
	flw	fa2, 252(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v12, fa2, v16
	vfmacc.vf	v8, ft11, v16
	flw	ft7, 16(a1)
	flw	fa7, 20(a1)
	flw	ft9, 24(a1)
	flw	fa2, 28(a1)
	fsw	fa2, 280(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	addi	a5, sp, 336
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fs2, v18
	addi	a5, sp, 528
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fa5, v18
	addi	a5, sp, 464
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fa4, v18
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fa3, v18
	flw	fa4, 16(a4)
	flw	ft6, 20(a4)
	flw	fs2, 24(a4)
	flw	fa5, 28(a4)
	fsw	fa5, 276(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v4, fa1, v18
	vfmacc.vf	v2, fa0, v18
	vfmacc.vf	v24, ft1, v18
	vfmacc.vf	v0, ft7, v18
	flw	fa3, 16(a2)
	flw	ft3, 20(a2)
	flw	ft8, 24(a2)
	flw	fa5, 28(a2)
	fsw	fa5, 272(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v26, fa4, v18
	vfmacc.vf	v28, fa3, v18
	flw	fa4, 16(a3)
	flw	ft1, 20(a3)
	flw	fa6, 24(a3)
	flw	fa5, 28(a3)
	fsw	fa5, 268(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v30, fa4, v18
	flw	fa4, 16(a6)
	flw	fa0, 20(a6)
	flw	ft7, 24(a6)
	flw	fa5, 28(a6)
	fsw	fa5, 264(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v6, fa4, v18
	flw	fa4, 16(a7)
	flw	fa1, 20(a7)
	flw	ft5, 24(a7)
	flw	fa5, 28(a7)
	fsw	fa5, 260(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v14, fa4, v18
	flw	ft10, 16(t0)
	flw	fa2, 20(t0)
	flw	ft4, 24(t0)
	flw	fa5, 28(t0)
	fsw	fa5, 256(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v10, ft10, v18
	flw	ft11, 16(t1)
	flw	fa3, 20(t1)
	flw	ft2, 24(t1)
	flw	ft10, 28(t1)
	vfmacc.vf	v12, ft11, v18
	ld	a5, 176(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v16, (a5)
	flw	fa5, 16(t2)
	flw	fa4, 20(t2)
	flw	ft0, 24(t2)
	flw	ft11, 28(t2)
	vfmacc.vf	v8, fa5, v18
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	addi	a0, sp, 528
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa5, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fs6, v16
	addi	a0, sp, 400
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v18, fs7, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fs8, v16
	vfmacc.vf	v4, fs9, v16
	vfmacc.vf	v2, fs10, v16
	vfmacc.vf	v24, fs11, v16
	vfmacc.vf	v0, fa7, v16
	vfmacc.vf	v26, ft6, v16
	vfmacc.vf	v28, ft3, v16
	vfmacc.vf	v30, ft1, v16
	vfmacc.vf	v6, fa0, v16
	vfmacc.vf	v14, fa1, v16
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, fa3, v16
	vfmacc.vf	v8, fa4, v16
	ld	a5, 168(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v18, (a5)
	ld	a5, 160(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v16, (a5)
	ld	a5, 152(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	addi	a0, sp, 528
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 244(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 400
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 240(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v20, fs4, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	vfmacc.vf	v22, fs5, v18
	vfmacc.vf	v4, fs3, v18
	vfmacc.vf	v2, fs1, v18
	vfmacc.vf	v24, fs0, v18
	vfmacc.vf	v0, ft9, v18
	vfmacc.vf	v26, fs2, v18
	vfmacc.vf	v28, ft8, v18
	vfmacc.vf	v30, fa6, v18
	vfmacc.vf	v6, ft7, v18
	vfmacc.vf	v14, ft5, v18
	vfmacc.vf	v10, ft4, v18
	vfmacc.vf	v12, ft2, v18
	vfmacc.vf	v8, ft0, v18
	vl2re32.v	v18, (a5)
	addi	a5, sp, 336
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	fs2, 32(s5)
	flw	fa5, 36(s5)
	fsw	fa5, 252(sp)                    # 4-byte Folded Spill
	flw	fa5, 40(s5)
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
	flw	fa5, 44(s5)
	fsw	fa5, 320(sp)                    # 4-byte Folded Spill
	flw	fa5, 32(s4)
	flw	fs7, 36(s4)
	flw	fa4, 40(s4)
	fsw	fa4, 244(sp)                    # 4-byte Folded Spill
	flw	fa4, 44(s4)
	fsw	fa4, 312(sp)                    # 4-byte Folded Spill
	flw	fa4, 32(s3)
	flw	fs8, 36(s3)
	flw	fs4, 40(s3)
	flw	fa3, 44(s3)
	fsw	fa3, 304(sp)                    # 4-byte Folded Spill
	flw	fa3, 32(s2)
	flw	fs10, 36(s2)
	flw	fs5, 40(s2)
	flw	fa2, 44(s2)
	fsw	fa2, 296(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 324(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 400
	vl2r.v	v20, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 316(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa2, v16
	addi	a5, sp, 464
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 308(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	fa2, 300(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v22, fa2, v16
	flw	fa1, 32(s1)
	flw	fs3, 36(s1)
	flw	fs6, 40(s1)
	flw	fa2, 44(s1)
	fsw	fa2, 324(sp)                    # 4-byte Folded Spill
	flw	fa2, 292(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v4, fa2, v16
	flw	fa2, 288(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v2, fa2, v16
	flw	fa2, 284(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v24, fa2, v16
	flw	fa2, 280(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v0, fa2, v16
	flw	ft3, 32(t5)
	flw	fs0, 36(t5)
	flw	fs9, 40(t5)
	flw	fa2, 44(t5)
	fsw	fa2, 316(sp)                    # 4-byte Folded Spill
	flw	fa2, 276(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v26, fa2, v16
	flw	fa2, 272(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v28, fa2, v16
	flw	fa2, 268(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v30, fa2, v16
	flw	fa2, 264(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v6, fa2, v16
	flw	ft4, 32(a0)
	flw	ft8, 36(a0)
	flw	fs11, 40(a0)
	flw	fa2, 44(a0)
	fsw	fa2, 308(sp)                    # 4-byte Folded Spill
	flw	fa2, 260(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa2, v16
	flw	fa2, 256(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, ft10, v16
	vfmacc.vf	v8, ft11, v16
	flw	ft5, 32(a1)
	flw	ft11, 36(a1)
	flw	fs1, 40(a1)
	flw	fa2, 44(a1)
	fsw	fa2, 300(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	addi	a5, sp, 336
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fs2, v18
	addi	a5, sp, 528
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fa5, v18
	addi	a5, sp, 464
	vl2r.v	v16, (a5)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fa4, v18
	vs2r.v	v16, (a5)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fa3, v18
	flw	fa4, 32(a4)
	flw	ft7, 36(a4)
	flw	ft9, 40(a4)
	flw	fa5, 44(a4)
	fsw	fa5, 292(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v4, fa1, v18
	vfmacc.vf	v2, ft3, v18
	vfmacc.vf	v24, ft4, v18
	vfmacc.vf	v0, ft5, v18
	flw	fa3, 32(a2)
	flw	ft3, 36(a2)
	flw	ft10, 40(a2)
	flw	fa5, 44(a2)
	fsw	fa5, 288(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v26, fa4, v18
	vfmacc.vf	v28, fa3, v18
	flw	fa4, 32(a3)
	flw	ft1, 36(a3)
	flw	fs2, 40(a3)
	flw	fa5, 44(a3)
	fsw	fa5, 284(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v30, fa4, v18
	flw	fa4, 32(a6)
	flw	fa0, 36(a6)
	flw	fa7, 40(a6)
	flw	fa5, 44(a6)
	fsw	fa5, 280(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v6, fa4, v18
	flw	fa4, 32(a7)
	flw	fa1, 36(a7)
	flw	ft6, 40(a7)
	flw	fa5, 44(a7)
	fsw	fa5, 276(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v14, fa4, v18
	flw	fa6, 32(t0)
	flw	fa2, 36(t0)
	flw	ft5, 40(t0)
	flw	fa5, 44(t0)
	fsw	fa5, 272(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v10, fa6, v18
	flw	ft4, 32(t1)
	flw	fa3, 36(t1)
	flw	ft2, 40(t1)
	flw	fa6, 44(t1)
	vfmacc.vf	v12, ft4, v18
	ld	a5, 144(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v16, (a5)
	flw	fa5, 32(t2)
	flw	fa4, 36(t2)
	flw	ft0, 40(t2)
	flw	ft4, 44(t2)
	vfmacc.vf	v8, fa5, v18
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	addi	a0, sp, 528
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 252(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa5, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fs7, v16
	addi	a0, sp, 400
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v18, fs8, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fs10, v16
	vfmacc.vf	v4, fs3, v16
	vfmacc.vf	v2, fs0, v16
	vfmacc.vf	v24, ft8, v16
	vfmacc.vf	v0, ft11, v16
	vfmacc.vf	v26, ft7, v16
	vfmacc.vf	v28, ft3, v16
	vfmacc.vf	v30, ft1, v16
	vfmacc.vf	v6, fa0, v16
	vfmacc.vf	v14, fa1, v16
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, fa3, v16
	vfmacc.vf	v8, fa4, v16
	ld	a5, 136(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v18, (a5)
	ld	a5, 128(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	vl2re32.v	v16, (a5)
	ld	a5, 120(sp)                     # 8-byte Folded Reload
	add	a5, a5, ra
	addi	a0, sp, 528
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 400
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 244(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v20, fs4, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	vfmacc.vf	v22, fs5, v18
	vfmacc.vf	v4, fs6, v18
	vfmacc.vf	v2, fs9, v18
	vfmacc.vf	v24, fs11, v18
	vfmacc.vf	v0, fs1, v18
	vfmacc.vf	v26, ft9, v18
	vfmacc.vf	v28, ft10, v18
	vfmacc.vf	v30, fs2, v18
	vfmacc.vf	v6, fa7, v18
	vfmacc.vf	v14, ft6, v18
	vfmacc.vf	v10, ft5, v18
	vfmacc.vf	v12, ft2, v18
	vfmacc.vf	v8, ft0, v18
	vl2re32.v	v18, (a5)
	addi	a5, sp, 336
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	ft11, 48(s5)
	flw	fa5, 52(s5)
	fsw	fa5, 268(sp)                    # 4-byte Folded Spill
	flw	fa5, 56(s5)
	fsw	fa5, 264(sp)                    # 4-byte Folded Spill
	flw	fa5, 60(s5)
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
	flw	fa5, 48(s4)
	flw	fs6, 52(s4)
	flw	fa4, 56(s4)
	fsw	fa4, 244(sp)                    # 4-byte Folded Spill
	flw	fa4, 60(s4)
	fsw	fa4, 252(sp)                    # 4-byte Folded Spill
	flw	fa4, 48(s3)
	flw	fs7, 52(s3)
	flw	fa3, 56(s3)
	fsw	fa3, 240(sp)                    # 4-byte Folded Spill
	flw	fa3, 60(s3)
	fsw	fa3, 260(sp)                    # 4-byte Folded Spill
	flw	fa3, 48(s2)
	flw	fs8, 52(s2)
	flw	fa2, 56(s2)
	fsw	fa2, 236(sp)                    # 4-byte Folded Spill
	flw	fa2, 60(s2)
	fsw	fa2, 256(sp)                    # 4-byte Folded Spill
	addi	a5, sp, 528
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 320(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	addi	a5, sp, 400
	vl2r.v	v20, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 312(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa2, v16
	addi	a5, sp, 464
	vl2r.v	v18, (a5)                       # vscale x 16-byte Folded Reload
	flw	fa2, 304(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa2, v16
	vs2r.v	v18, (a5)                       # vscale x 16-byte Folded Spill
	flw	fa2, 296(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v22, fa2, v16
	flw	fa1, 48(s1)
	flw	fs9, 52(s1)
	flw	fa2, 56(s1)
	fsw	fa2, 304(sp)                    # 4-byte Folded Spill
	flw	fa2, 60(s1)
	fsw	fa2, 320(sp)                    # 4-byte Folded Spill
	flw	fa2, 324(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v4, fa2, v16
	flw	fa2, 316(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v2, fa2, v16
	flw	fa2, 308(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v24, fa2, v16
	flw	fa2, 300(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v0, fa2, v16
	flw	fa0, 48(t5)
	flw	fs10, 52(t5)
	flw	fa2, 56(t5)
	fsw	fa2, 300(sp)                    # 4-byte Folded Spill
	flw	fa2, 60(t5)
	fsw	fa2, 324(sp)                    # 4-byte Folded Spill
	flw	fa2, 292(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v26, fa2, v16
	flw	fa2, 288(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v28, fa2, v16
	flw	fa2, 284(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v30, fa2, v16
	flw	fa2, 280(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v6, fa2, v16
	flw	ft1, 48(a0)
	flw	ft8, 52(a0)
	flw	fs4, 56(a0)
	flw	fa2, 60(a0)
	fsw	fa2, 316(sp)                    # 4-byte Folded Spill
	flw	fa2, 276(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa2, v16
	flw	fa2, 272(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, fa6, v16
	vfmacc.vf	v8, ft4, v16
	flw	ft4, 48(a1)
	flw	fa6, 52(a1)
	flw	fs2, 56(a1)
	flw	fa2, 60(a1)
	fsw	fa2, 312(sp)                    # 4-byte Folded Spill
	addi	a0, sp, 528
	vl2r.v	v16, (a0)                       # vscale x 16-byte Folded Reload
	addi	a0, sp, 336
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, ft11, v18
	addi	a0, sp, 528
	vs2r.v	v16, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fa5, v18
	addi	a0, sp, 464
	vl2r.v	v16, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v16, fa4, v18
	vs2r.v	v16, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fa3, v18
	flw	fa4, 48(a4)
	flw	ft5, 52(a4)
	flw	ft9, 56(a4)
	flw	fa5, 60(a4)
	fsw	fa5, 308(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v4, fa1, v18
	vfmacc.vf	v2, fa0, v18
	vfmacc.vf	v24, ft1, v18
	vfmacc.vf	v0, ft4, v18
	flw	fa3, 48(a2)
	flw	ft2, 52(a2)
	flw	fs11, 56(a2)
	flw	fa5, 60(a2)
	fsw	fa5, 296(sp)                    # 4-byte Folded Spill
	vfmacc.vf	v26, fa4, v18
	vfmacc.vf	v28, fa3, v18
	flw	fa4, 48(a3)
	flw	ft0, 52(a3)
	flw	fa7, 56(a3)
	flw	fs5, 60(a3)
	vfmacc.vf	v30, fa4, v18
	flw	fa4, 48(a6)
	flw	fa0, 52(a6)
	flw	ft7, 56(a6)
	flw	fs3, 60(a6)
	vfmacc.vf	v6, fa4, v18
	flw	fa4, 48(a7)
	flw	fa1, 52(a7)
	flw	ft6, 56(a7)
	flw	fs1, 60(a7)
	vfmacc.vf	v14, fa4, v18
	flw	fa4, 48(t0)
	flw	fa2, 52(t0)
	flw	ft4, 56(t0)
	flw	fs0, 60(t0)
	vfmacc.vf	v10, fa4, v18
	flw	ft11, 48(t1)
	flw	fa3, 52(t1)
	flw	ft3, 56(t1)
	flw	ft10, 60(t1)
	vfmacc.vf	v12, ft11, v18
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	add	a0, a0, ra
	vl2re32.v	v16, (a0)
	flw	fa5, 48(t2)
	flw	fa4, 52(t2)
	flw	ft1, 56(t2)
	flw	ft11, 60(t2)
	vfmacc.vf	v8, fa5, v18
	addi	a0, sp, 528
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 268(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa5, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v20, fs6, v16
	addi	a0, sp, 400
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	vfmacc.vf	v18, fs7, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v22, fs8, v16
	addi	a0, sp, 336
	vs2r.v	v22, (a0)                       # vscale x 16-byte Folded Spill
	vfmacc.vf	v4, fs9, v16
	vfmacc.vf	v2, fs10, v16
	vfmacc.vf	v24, ft8, v16
	vfmacc.vf	v0, fa6, v16
	vfmacc.vf	v26, ft5, v16
	vfmacc.vf	v28, ft2, v16
	vfmacc.vf	v30, ft0, v16
	vfmacc.vf	v6, fa0, v16
	vfmacc.vf	v14, fa1, v16
	vfmacc.vf	v10, fa2, v16
	vfmacc.vf	v12, fa3, v16
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	add	a0, a0, ra
	vl2re32.v	v18, (a0)
	vfmacc.vf	v8, fa4, v16
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	a0, a0, ra
	vl2re32.v	v16, (a0)
	addi	a0, sp, 528
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 264(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	addi	a0, sp, 464
	vl2r.v	v20, (a0)                       # vscale x 16-byte Folded Reload
	vmv.v.v	v22, v0
	vmv.v.v	v0, v2
	vmv.v.v	v2, v4
	vmv.v.v	v4, v6
	vmv.v.v	v6, v30
	vmv.v.v	v30, v28
	vmv.v.v	v28, v26
	vmv.v.v	v26, v24
	vmv.v.v	v24, v14
	addi	a0, sp, 400
	vl2r.v	v14, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 244(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa5, v18
	vs2r.v	v14, (a0)                       # vscale x 16-byte Folded Spill
	flw	fa5, 240(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v18
	addi	a0, sp, 336
	vl2r.v	v14, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 236(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa5, v18
	flw	fa5, 304(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v2, fa5, v18
	flw	fa5, 300(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v0, fa5, v18
	vfmacc.vf	v26, fs4, v18
	vfmacc.vf	v22, fs2, v18
	vfmacc.vf	v28, ft9, v18
	vfmacc.vf	v30, fs11, v18
	vfmacc.vf	v6, fa7, v18
	vfmacc.vf	v4, ft7, v18
	vfmacc.vf	v24, ft6, v18
	vfmacc.vf	v10, ft4, v18
	vfmacc.vf	v12, ft3, v18
	vfmacc.vf	v8, ft1, v18
	addi	a0, sp, 528
	vl2r.v	v18, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v18, fa5, v16
	vs2r.v	v18, (a0)                       # vscale x 16-byte Folded Spill
	vmv.v.v	v18, v12
	vmv.v.v	v12, v10
	vmv.v.v	v10, v8
	vmv.v.v	v8, v14
	addi	a0, sp, 400
	vl2r.v	v14, (a0)                       # vscale x 16-byte Folded Reload
	flw	fa5, 252(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v14, fa5, v16
	vs2r.v	v14, (a0)                       # vscale x 16-byte Folded Spill
	vmv.v.v	v14, v24
	vmv.v.v	v24, v26
	vmv.v.v	v26, v28
	vmv.v.v	v28, v30
	vmv.v.v	v30, v6
	vmv.v.v	v6, v4
	vmv.v.v	v4, v2
	vmv.v.v	v2, v0
	vmv.v.v	v0, v22
	vmv.v.v	v22, v8
	vmv.v.v	v8, v10
	vmv.v.v	v10, v12
	vmv.v.v	v12, v18
	flw	fa5, 260(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v20, fa5, v16
	addi	a0, sp, 464
	vs2r.v	v20, (a0)                       # vscale x 16-byte Folded Spill
	flw	fa5, 256(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v22, fa5, v16
	flw	fa5, 320(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v4, fa5, v16
	flw	fa5, 324(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v2, fa5, v16
	flw	fa5, 316(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v24, fa5, v16
	flw	fa5, 312(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v0, fa5, v16
	flw	fa5, 308(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v26, fa5, v16
	flw	fa5, 296(sp)                    # 4-byte Folded Reload
	vfmacc.vf	v28, fa5, v16
	vfmacc.vf	v30, fs5, v16
	vfmacc.vf	v6, fs3, v16
	vfmacc.vf	v14, fs1, v16
	vfmacc.vf	v10, fs0, v16
	vfmacc.vf	v12, ft10, v16
	vfmacc.vf	v8, ft11, v16
	lui	a0, 1
	add	ra, ra, a0
	addi	s6, s6, 64
	lui	a0, 4
	bne	ra, a0, .LBB0_3
# %bb.4:
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	slli	a1, a1, 2
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	add	a0, a2, a0
	add	a0, a0, a1
	addi	a1, sp, 528
	vl2r.v	v16, (a1)                       # vscale x 16-byte Folded Reload
	vs2r.v	v16, (a0)
	addi	a1, a0, 256
	addi	a2, a0, 512
	addi	a3, sp, 400
	vl2r.v	v16, (a3)                       # vscale x 16-byte Folded Reload
	vs2r.v	v16, (a1)
	addi	a1, a0, 768
	addi	a3, sp, 464
	vl2r.v	v16, (a3)                       # vscale x 16-byte Folded Reload
	vs2r.v	v16, (a2)
	addi	a2, a0, 1024
	vs2r.v	v22, (a1)
	addi	a1, a0, 1280
	vs2r.v	v4, (a2)
	addi	a2, a0, 1536
	vs2r.v	v2, (a1)
	addi	a1, a0, 1792
	addi	a0, a0, 2047
	vs2r.v	v24, (a2)
	vs2r.v	v0, (a1)
	addi	a1, a0, 1
	addi	a2, a0, 257
	vs2r.v	v26, (a1)
	addi	a1, a0, 513
	vs2r.v	v28, (a2)
	addi	a2, a0, 769
	vs2r.v	v30, (a1)
	addi	a1, a0, 1025
	vs2r.v	v6, (a2)
	addi	a2, a0, 1281
	vs2r.v	v14, (a1)
	addi	a1, a0, 1537
	addi	a0, a0, 1793
	vs2r.v	v10, (a2)
	vs2r.v	v12, (a1)
	vs2r.v	v8, (a0)
	addi	sp, sp, 256
	ld	ra, 536(sp)                     # 8-byte Folded Reload
	ld	s0, 528(sp)                     # 8-byte Folded Reload
	ld	s1, 520(sp)                     # 8-byte Folded Reload
	ld	s2, 512(sp)                     # 8-byte Folded Reload
	ld	s3, 504(sp)                     # 8-byte Folded Reload
	ld	s4, 496(sp)                     # 8-byte Folded Reload
	ld	s5, 488(sp)                     # 8-byte Folded Reload
	ld	s6, 480(sp)                     # 8-byte Folded Reload
	ld	s7, 472(sp)                     # 8-byte Folded Reload
	ld	s8, 464(sp)                     # 8-byte Folded Reload
	ld	s9, 456(sp)                     # 8-byte Folded Reload
	ld	s10, 448(sp)                    # 8-byte Folded Reload
	ld	s11, 440(sp)                    # 8-byte Folded Reload
	fld	fs0, 432(sp)                    # 8-byte Folded Reload
	fld	fs1, 424(sp)                    # 8-byte Folded Reload
	fld	fs2, 416(sp)                    # 8-byte Folded Reload
	fld	fs3, 408(sp)                    # 8-byte Folded Reload
	fld	fs4, 400(sp)                    # 8-byte Folded Reload
	fld	fs5, 392(sp)                    # 8-byte Folded Reload
	fld	fs6, 384(sp)                    # 8-byte Folded Reload
	fld	fs7, 376(sp)                    # 8-byte Folded Reload
	fld	fs8, 368(sp)                    # 8-byte Folded Reload
	fld	fs9, 360(sp)                    # 8-byte Folded Reload
	fld	fs10, 352(sp)                   # 8-byte Folded Reload
	fld	fs11, 344(sp)                   # 8-byte Folded Reload
	addi	sp, sp, 544
	ret
.Lfunc_end0:
	.size	matmul, .Lfunc_end0-matmul
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
