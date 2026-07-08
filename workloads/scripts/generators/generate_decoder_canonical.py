#!/usr/bin/env python3
"""Generate canonical decoder graph fixtures.

The default graph models one causal transformer decoder block at:

  SEQ=512, D_MODEL=512, HEADS=8, HEAD_DIM=64, FFN_DIM=2048

Attention is the canonical materialized form per head:

  QK^T -> causal Softmax -> PV

FlashAttention-style nodes are intentionally not part of this graph.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
WORKLOADS_DIR = SCRIPTS_DIR.parents[1]
GRAPHS_DIR = WORKLOADS_DIR / "graphs"

SEQ = 512
HEADS = 8
D_MODEL = 512
HEAD_DIM = 64
FFN_DIM = 2048
GRAPH_NAME = "decoder_canonical_mh8"
OUT_DIR = GRAPHS_DIR / GRAPH_NAME

SCALED_CASES = {
    "small": {
        "name": "decoder_canonical_small_mh4",
        "seq": 256,
        "heads": 4,
        "d_model": 256,
        "head_dim": 64,
        "ffn_dim": 1024,
        "fixed_policy": True,
    },
    "base": {
        "name": "decoder_canonical_mh8",
        "seq": 512,
        "heads": 8,
        "d_model": 512,
        "head_dim": 64,
        "ffn_dim": 2048,
        "fixed_policy": False,
        "checked_in_mh8_layout": True,
    },
    "large": {
        "name": "decoder_canonical_large_mh16",
        "seq": 1024,
        "heads": 16,
        "d_model": 1024,
        "head_dim": 64,
        "ffn_dim": 4096,
        "fixed_policy": True,
    },
}

TRISPM_MATMUL_PARAMS = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
}
TRISPM_MATMUL_ENV = {
    "TRITON_MICRO_M": "8",
    "TRITON_SPM_WINDOW_K": "4",
    "TRITON_SPM_PROMOTION_REPORT": "1",
}
BASE_MATMUL_ROLE_OVERRIDES = {
    "qkv": {
        "cache": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 2,
        },
    },
    "qk": {
        "cache": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
        },
    },
    "pv": {
        "cache": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 2,
        },
    },
    "o_proj": {
        "cache": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
        },
    },
    "ffn_up": {
        "cache": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
        },
    },
    "ffn_down": {
        "cache": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        },
        "spm": {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
        },
    },
}

PAPER_CACHE_GEMM_PARAMS = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
}
PAPER_SPM_GEMM_PARAMS = dict(TRISPM_MATMUL_PARAMS)
SCALE_MATMUL_ROLE_OVERRIDES = {
    "qkv": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
    "qk": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
    "pv": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
    "o_proj": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
    "ffn_up": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
    "ffn_down": {
        "cache": dict(PAPER_CACHE_GEMM_PARAMS),
        "spm": dict(PAPER_SPM_GEMM_PARAMS),
    },
}
MATMUL_ROLE_OVERRIDES = BASE_MATMUL_ROLE_OVERRIDES
CURRENT_FIXED_POLICY = False
CURRENT_CHECKED_IN_MH8_LAYOUT = False
TRISPM_ROW_BLOCK_PARAMS = {
    "SPM_ROW_BLOCK": 4,
    "SPM_ROW_GROUP_BLOCKS": 8,
    "SPM_INTERNAL_ROW_BLOCK": 1,
}
CACHE_SOFTMAX_BLOCK_N = 512
TRISPM_SOFTMAX_BLOCK_N = 32
TRISPM_SOFTMAX_ROW_BLOCK_PARAMS = {
    "SPM_ROW_BLOCK": 2,
    "SPM_ROW_GROUP_BLOCKS": 8,
    "SPM_INTERNAL_ROW_BLOCK": 1,
}
CACHE_ROW_BLOCK_PARAMS = {
    "SPM_ROW_BLOCK": 1,
    "SPM_ROW_GROUP_BLOCKS": 1,
    "SPM_INTERNAL_ROW_BLOCK": 0,
}


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def write_table(lines: list[str], table: str, values: dict[str, Any]) -> None:
    lines.append("")
    lines.append(f"[{table}]")
    for key, value in values.items():
        lines.append(f"{key} = {toml_value(value)}")


def set_shape(
    seq: int,
    heads: int,
    d_model: int,
    head_dim: int,
    ffn_dim: int,
    graph_name: str,
    fixed_policy: bool = False,
    checked_in_mh8_layout: bool = False,
) -> None:
    global SEQ, HEADS, D_MODEL, HEAD_DIM, FFN_DIM, GRAPH_NAME, OUT_DIR
    global MATMUL_ROLE_OVERRIDES, CURRENT_FIXED_POLICY
    global CURRENT_CHECKED_IN_MH8_LAYOUT
    if heads <= 0 or heads % 2 != 0:
        raise ValueError("HEADS must be a positive even number")
    if d_model != heads * head_dim:
        raise ValueError("D_MODEL must equal HEADS * HEAD_DIM")
    SEQ = seq
    HEADS = heads
    D_MODEL = d_model
    HEAD_DIM = head_dim
    FFN_DIM = ffn_dim
    GRAPH_NAME = graph_name
    OUT_DIR = GRAPHS_DIR / graph_name
    CURRENT_CHECKED_IN_MH8_LAYOUT = (
        checked_in_mh8_layout
        and graph_name == "decoder_canonical_mh8"
        and heads == 8
    )
    MATMUL_ROLE_OVERRIDES = {
        role: {
            "cache": dict(cfg["cache"]),
            "spm": dict(cfg["spm"]),
        }
        for role, cfg in (
            SCALE_MATMUL_ROLE_OVERRIDES
            if fixed_policy
            else BASE_MATMUL_ROLE_OVERRIDES
        ).items()
    }
    CURRENT_FIXED_POLICY = fixed_policy


def head_pair_name(left: int, right: int) -> str:
    if CURRENT_CHECKED_IN_MH8_LAYOUT:
        if left == 0 and right == HEADS - 1:
            return "attn_head_sum_all"
        head_range = "".join(str(head) for head in range(left, right + 1))
        return f"attn_head_sum_{head_range}"
    return f"attn_head_sum_{left}_{right}"


def head_pair_tensor_name(left: int, right: int) -> str:
    if CURRENT_CHECKED_IN_MH8_LAYOUT:
        head_range = "".join(str(head) for head in range(left, right + 1))
        return f"attn_sum_{head_range}"
    return f"attn_sum_{left}_{right}"


def head_sum_plan() -> list[dict[str, Any]]:
    """Build a pairwise head-sum tree, carrying odd nodes to the next level."""
    current = [
        {"tensor": f"attn_proj_h{h}", "start": h, "end": h}
        for h in range(HEADS)
    ]
    plan: list[dict[str, Any]] = []

    while len(current) > 1:
        next_level: list[dict[str, Any]] = []
        idx = 0
        while idx < len(current):
            if idx + 1 == len(current):
                next_level.append(current[idx])
                idx += 1
                continue
            left = current[idx]
            right = current[idx + 1]
            start = int(left["start"])
            end = int(right["end"])
            node_name = head_pair_name(start, end)
            output = head_pair_tensor_name(start, end)
            record = {
                "name": node_name,
                "left": str(left["tensor"]),
                "right": str(right["tensor"]),
                "output": output,
                "start": start,
                "end": end,
            }
            plan.append(record)
            next_level.append({"tensor": output, "start": start, "end": end})
            idx += 2
        current = next_level

    final_output = str(current[0]["tensor"])
    for record in reversed(plan):
        if record["output"] == final_output:
            record["output"] = "attn_sum"
            break
    return plan


def head_sum_consumers(plan: list[dict[str, Any]]) -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for record in plan:
        consumers.setdefault(str(record["left"]), []).append(str(record["name"]))
        consumers.setdefault(str(record["right"]), []).append(str(record["name"]))
    consumers.setdefault("attn_sum", []).append("attn_residual_add")
    return consumers


def ref_expr(tensor: str) -> str:
    if tensor.startswith("attn_proj_h"):
        return f"attn_proj_ref[{int(tensor.removeprefix('attn_proj_h'))}]"
    if CURRENT_CHECKED_IN_MH8_LAYOUT and tensor.startswith("attn_sum_"):
        return f"sum{tensor.removeprefix('attn_sum_')}_ref"
    return f"{tensor}_ref"


def sum_ref_name(tensor: str) -> str:
    if CURRENT_CHECKED_IN_MH8_LAYOUT and tensor.startswith("attn_sum_"):
        return f"sum{tensor.removeprefix('attn_sum_')}_ref"
    return f"{tensor}_ref"


def c_expr(tensor: str) -> str:
    if tensor.startswith("attn_proj_h"):
        return f"attn_proj[{int(tensor.removeprefix('attn_proj_h'))}]"
    return tensor


def matmul_params(m: int, n: int, k: int, role: str) -> dict[str, int]:
    return {
        "M": m,
        "N": n,
        "K": k,
        **MATMUL_ROLE_OVERRIDES[role]["spm"],
        "CHECK_RESULT": 0,
    }


def layer_norm_params() -> dict[str, int]:
    params = {
        "M": SEQ,
        "N": D_MODEL,
    }
    if not CURRENT_CHECKED_IN_MH8_LAYOUT:
        params["BLOCK_N"] = 8
    params.update({
        "CHECK_RESULT": 0,
        "LAYERNORM_FLUSH_BEFORE_ROI": 1,
        **TRISPM_ROW_BLOCK_PARAMS,
    })
    return params


def add_matmul_overrides(node: dict[str, Any], role: str) -> dict[str, Any]:
    role_cfg = MATMUL_ROLE_OVERRIDES[role]
    node["cache"] = {"params": dict(role_cfg["cache"]), "env": {}}
    node["spm"] = {
        "params": dict(role_cfg["spm"]),
        "env": dict(TRISPM_MATMUL_ENV),
    }
    return node


def add_row_block_overrides(node: dict[str, Any]) -> dict[str, Any]:
    if CURRENT_CHECKED_IN_MH8_LAYOUT:
        cache_params = dict(CACHE_ROW_BLOCK_PARAMS)
        spm_params = dict(TRISPM_ROW_BLOCK_PARAMS)
    else:
        cache_params = {"BLOCK_N": D_MODEL, **CACHE_ROW_BLOCK_PARAMS}
        spm_params = {"BLOCK_N": 8, **TRISPM_ROW_BLOCK_PARAMS}
    node["cache"] = {"params": cache_params}
    node["spm"] = {"params": spm_params}
    return node


def add_softmax_overrides(node: dict[str, Any]) -> dict[str, Any]:
    node["cache"] = {
        "params": {
            "BLOCK_N": SEQ,
            **CACHE_ROW_BLOCK_PARAMS,
        }
    }
    node["spm"] = {
        "params": {
            "BLOCK_N": TRISPM_SOFTMAX_BLOCK_N,
            **TRISPM_SOFTMAX_ROW_BLOCK_PARAMS,
        }
    }
    return node


def render_graph_toml() -> str:
    sum_plan = head_sum_plan()
    sum_consumers = head_sum_consumers(sum_plan)
    tensors: dict[str, dict[str, Any]] = {
        "x": {"kind": "external_input", "read_only": True},
        "gamma": {"kind": "external_weight", "read_only": True},
        "beta": {"kind": "external_weight", "read_only": True},
        "ln_out": {
            "kind": "intermediate",
            "producer": "layer_norm",
            "consumers": [
                *(f"q_proj_h{h}" for h in range(HEADS)),
                *(f"k_proj_h{h}" for h in range(HEADS)),
                *(f"v_proj_h{h}" for h in range(HEADS)),
            ],
        },
    }

    for h in range(HEADS):
        tensors.update({
            f"wq_h{h}": {"kind": "external_weight", "read_only": True},
            f"wk_h{h}": {"kind": "external_weight", "read_only": True},
            f"wv_h{h}": {"kind": "external_weight", "read_only": True},
            f"q_h{h}": {
                "kind": "intermediate",
                "producer": f"q_proj_h{h}",
                "consumers": [f"qk_h{h}"],
            },
            f"k_h{h}": {
                "kind": "intermediate",
                "producer": f"k_proj_h{h}",
                "consumers": [f"k_transpose_h{h}"],
            },
            f"k_t_h{h}": {
                "kind": "intermediate",
                "producer": f"k_transpose_h{h}",
                "consumers": [f"qk_h{h}"],
            },
            f"v_h{h}": {
                "kind": "intermediate",
                "producer": f"v_proj_h{h}",
                "consumers": [f"pv_h{h}"],
            },
            f"scores_h{h}": {
                "kind": "intermediate",
                "producer": f"qk_h{h}",
                "consumers": [f"softmax_h{h}"],
            },
            f"probs_h{h}": {
                "kind": "intermediate",
                "producer": f"softmax_h{h}",
                "consumers": [f"pv_h{h}"],
            },
            f"attn_h{h}": {
                "kind": "intermediate",
                "producer": f"pv_h{h}",
                "consumers": [f"o_proj_h{h}"],
            },
            f"wo_h{h}": {"kind": "external_weight", "read_only": True},
            f"attn_proj_h{h}": {
                "kind": "intermediate",
                "producer": f"o_proj_h{h}",
                "consumers": sum_consumers.get(f"attn_proj_h{h}", []),
            },
        })

    tensors.update({
        str(record["output"]): {
            "kind": "intermediate",
            "producer": str(record["name"]),
            "consumers": sum_consumers.get(str(record["output"]), []),
        }
        for record in sum_plan
    })
    tensors.update({
        "residual": {"kind": "external_input", "read_only": True},
        "resid_out": {
            "kind": "intermediate",
            "producer": "attn_residual_add",
            "consumers": ["ln2", "final_residual_add"],
        },
        "gamma2": {"kind": "external_weight", "read_only": True},
        "beta2": {"kind": "external_weight", "read_only": True},
        "ln2_out": {
            "kind": "intermediate",
            "producer": "ln2",
            "consumers": ["ffn_up"],
        },
        "w_up": {"kind": "external_weight", "read_only": True},
        "ffn_hidden": {
            "kind": "intermediate",
            "producer": "ffn_up",
            "consumers": ["ffn_activation"],
        },
        "ffn_act": {
            "kind": "intermediate",
            "producer": "ffn_activation",
            "consumers": ["ffn_down"],
        },
        "w_down": {"kind": "external_weight", "read_only": True},
        "ffn_out": {
            "kind": "intermediate",
            "producer": "ffn_down",
            "consumers": ["final_residual_add"],
        },
        "block_out": {
            "kind": "graph_output",
            "producer": "final_residual_add",
        },
    })

    nodes: dict[str, dict[str, Any]] = {
        "layer_norm": add_row_block_overrides({
            "kernel": "layer_norm",
            "args": ["x", "gamma", "beta", "ln_out"],
            "params": layer_norm_params(),
        }),
    }

    for h in range(HEADS):
        nodes.update({
            f"q_proj_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": ["ln_out", f"wq_h{h}", f"q_h{h}"],
                "params": matmul_params(SEQ, HEAD_DIM, D_MODEL, "qkv"),
            }, "qkv"),
            f"k_proj_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": ["ln_out", f"wk_h{h}", f"k_h{h}"],
                "params": matmul_params(SEQ, HEAD_DIM, D_MODEL, "qkv"),
            }, "qkv"),
            f"v_proj_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": ["ln_out", f"wv_h{h}", f"v_h{h}"],
                "params": matmul_params(SEQ, HEAD_DIM, D_MODEL, "qkv"),
            }, "qkv"),
            f"k_transpose_h{h}": {
                "kernel": "transpose",
                "args": [f"k_h{h}", f"k_t_h{h}"],
                "params": {
                    "M": SEQ,
                    "N": HEAD_DIM,
                    "BLOCK_M": 16,
                    "BLOCK_N": 16,
                    "CHECK_RESULT": 0,
                },
            },
            f"qk_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": [f"q_h{h}", f"k_t_h{h}", f"scores_h{h}"],
                "params": matmul_params(SEQ, SEQ, HEAD_DIM, "qk"),
            }, "qk"),
            f"softmax_h{h}": add_softmax_overrides({
                "kernel": "softmax",
                "args": [f"scores_h{h}", f"probs_h{h}"],
                "params": {
                    "M": SEQ,
                    "N": SEQ,
                    "BLOCK_N": TRISPM_SOFTMAX_BLOCK_N,
                    "CAUSAL": 1,
                    **TRISPM_SOFTMAX_ROW_BLOCK_PARAMS,
                    "CHECK_RESULT": 0,
                },
            }),
            f"pv_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": [f"probs_h{h}", f"v_h{h}", f"attn_h{h}"],
                "params": matmul_params(SEQ, HEAD_DIM, SEQ, "pv"),
            }, "pv"),
            f"o_proj_h{h}": add_matmul_overrides({
                "kernel": "matmul",
                "args": [f"attn_h{h}", f"wo_h{h}", f"attn_proj_h{h}"],
                "params": matmul_params(SEQ, D_MODEL, HEAD_DIM, "o_proj"),
            }, "o_proj"),
        })

    for record in sum_plan:
        nodes[str(record["name"])] = {
            "kernel": "residual_add",
            "args": [
                str(record["left"]),
                str(record["right"]),
                str(record["output"]),
            ],
            "params": {
                "SIZE": SEQ * D_MODEL,
                "BLOCK_SIZE": 64,
                "CHECK_RESULT": 0,
            },
        }

    nodes.update({
        "attn_residual_add": {
            "kernel": "residual_add",
            "args": ["attn_sum", "residual", "resid_out"],
            "params": {"SIZE": SEQ * D_MODEL, "BLOCK_SIZE": 64, "CHECK_RESULT": 0},
        },
        "ln2": add_row_block_overrides({
            "kernel": "layer_norm",
            "args": ["resid_out", "gamma2", "beta2", "ln2_out"],
            "params": layer_norm_params(),
        }),
        "ffn_up": add_matmul_overrides({
            "kernel": "matmul",
            "args": ["ln2_out", "w_up", "ffn_hidden"],
            "params": matmul_params(SEQ, FFN_DIM, D_MODEL, "ffn_up"),
        }, "ffn_up"),
        "ffn_activation": {
            "kernel": "activation",
            "args": ["ffn_hidden", "ffn_act"],
            "params": {"SIZE": SEQ * FFN_DIM, "BLOCK_SIZE": 64, "CHECK_RESULT": 0},
        },
        "ffn_down": add_matmul_overrides({
            "kernel": "matmul",
            "args": ["ffn_act", "w_down", "ffn_out"],
            "params": matmul_params(SEQ, D_MODEL, FFN_DIM, "ffn_down"),
        }, "ffn_down"),
        "final_residual_add": {
            "kernel": "residual_add",
            "args": ["ffn_out", "resid_out", "block_out"],
            "params": {"SIZE": SEQ * D_MODEL, "BLOCK_SIZE": 64, "CHECK_RESULT": 0},
        },
    })

    harness_params = {
        "SEQ": SEQ,
        "HEADS": HEADS,
        "D_MODEL": D_MODEL,
        "HEAD_DIM": HEAD_DIM,
        "FFN_DIM": FFN_DIM,
        "BLOCK": 64,
        "QKV_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "QKV_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "QK_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "QK_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "PV_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "PV_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "O_PROJ_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "O_PROJ_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "FFN_UP_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "FFN_UP_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "FFN_DOWN_BLOCK_M": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"],
        "FFN_DOWN_BLOCK_N": TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"],
        "K_TRANSPOSE_BLOCK_M": 16,
        "K_TRANSPOSE_BLOCK_N": 16,
        "LAYERNORM_ROW_BLOCK": TRISPM_ROW_BLOCK_PARAMS["SPM_ROW_BLOCK"],
        "LAYERNORM_ROW_GROUP_BLOCKS": TRISPM_ROW_BLOCK_PARAMS["SPM_ROW_GROUP_BLOCKS"],
        "LAYERNORM_INTERNAL_ROW_BLOCK": TRISPM_ROW_BLOCK_PARAMS["SPM_INTERNAL_ROW_BLOCK"],
        "SOFTMAX_ROW_BLOCK": TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_ROW_BLOCK"],
        "SOFTMAX_ROW_GROUP_BLOCKS": TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_ROW_GROUP_BLOCKS"],
        "SOFTMAX_INTERNAL_ROW_BLOCK": TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_INTERNAL_ROW_BLOCK"],
        "CAUSAL": 1,
        "FOLD_ATTENTION_SCALE_IN_Q": 1,
        "CHECK_RESULT": 0,
        "CHECK_INTERMEDIATES": 0,
        "FLUSH_BEFORE_ROI": 1,
        "TRACE_PROGRESS": 0,
        "DUMP_KERNEL_STATS": 0,
    }
    cache_harness_params = {
        "QKV_BLOCK_M": MATMUL_ROLE_OVERRIDES["qkv"]["cache"]["BLOCK_SIZE_M"],
        "QKV_BLOCK_N": MATMUL_ROLE_OVERRIDES["qkv"]["cache"]["BLOCK_SIZE_N"],
        "QK_BLOCK_M": MATMUL_ROLE_OVERRIDES["qk"]["cache"]["BLOCK_SIZE_M"],
        "QK_BLOCK_N": MATMUL_ROLE_OVERRIDES["qk"]["cache"]["BLOCK_SIZE_N"],
        "PV_BLOCK_M": MATMUL_ROLE_OVERRIDES["pv"]["cache"]["BLOCK_SIZE_M"],
        "PV_BLOCK_N": MATMUL_ROLE_OVERRIDES["pv"]["cache"]["BLOCK_SIZE_N"],
        "O_PROJ_BLOCK_M": MATMUL_ROLE_OVERRIDES["o_proj"]["cache"]["BLOCK_SIZE_M"],
        "O_PROJ_BLOCK_N": MATMUL_ROLE_OVERRIDES["o_proj"]["cache"]["BLOCK_SIZE_N"],
        "FFN_UP_BLOCK_M": MATMUL_ROLE_OVERRIDES["ffn_up"]["cache"]["BLOCK_SIZE_M"],
        "FFN_UP_BLOCK_N": MATMUL_ROLE_OVERRIDES["ffn_up"]["cache"]["BLOCK_SIZE_N"],
        "FFN_DOWN_BLOCK_M": MATMUL_ROLE_OVERRIDES["ffn_down"]["cache"]["BLOCK_SIZE_M"],
        "FFN_DOWN_BLOCK_N": MATMUL_ROLE_OVERRIDES["ffn_down"]["cache"]["BLOCK_SIZE_N"],
        "LAYERNORM_ROW_BLOCK": CACHE_ROW_BLOCK_PARAMS["SPM_ROW_BLOCK"],
        "LAYERNORM_ROW_GROUP_BLOCKS": CACHE_ROW_BLOCK_PARAMS["SPM_ROW_GROUP_BLOCKS"],
        "LAYERNORM_INTERNAL_ROW_BLOCK": CACHE_ROW_BLOCK_PARAMS["SPM_INTERNAL_ROW_BLOCK"],
        "SOFTMAX_ROW_BLOCK": CACHE_ROW_BLOCK_PARAMS["SPM_ROW_BLOCK"],
        "SOFTMAX_ROW_GROUP_BLOCKS": CACHE_ROW_BLOCK_PARAMS["SPM_ROW_GROUP_BLOCKS"],
        "SOFTMAX_INTERNAL_ROW_BLOCK": CACHE_ROW_BLOCK_PARAMS["SPM_INTERNAL_ROW_BLOCK"],
    }
    c_macros = [
        "GRAPH_SEQ={SEQ}",
        "GRAPH_HEADS={HEADS}",
        "GRAPH_D_MODEL={D_MODEL}",
        "GRAPH_HEAD_DIM={HEAD_DIM}",
        "GRAPH_FFN_DIM={FFN_DIM}",
        "GRAPH_BLOCK={BLOCK}",
        "GRAPH_QKV_BLOCK_M={QKV_BLOCK_M}",
        "GRAPH_QKV_BLOCK_N={QKV_BLOCK_N}",
        "GRAPH_QK_BLOCK_M={QK_BLOCK_M}",
        "GRAPH_QK_BLOCK_N={QK_BLOCK_N}",
        "GRAPH_PV_BLOCK_M={PV_BLOCK_M}",
        "GRAPH_PV_BLOCK_N={PV_BLOCK_N}",
        "GRAPH_O_PROJ_BLOCK_M={O_PROJ_BLOCK_M}",
        "GRAPH_O_PROJ_BLOCK_N={O_PROJ_BLOCK_N}",
        "GRAPH_FFN_UP_BLOCK_M={FFN_UP_BLOCK_M}",
        "GRAPH_FFN_UP_BLOCK_N={FFN_UP_BLOCK_N}",
        "GRAPH_FFN_DOWN_BLOCK_M={FFN_DOWN_BLOCK_M}",
        "GRAPH_FFN_DOWN_BLOCK_N={FFN_DOWN_BLOCK_N}",
        "GRAPH_K_TRANSPOSE_BLOCK_M={K_TRANSPOSE_BLOCK_M}",
        "GRAPH_K_TRANSPOSE_BLOCK_N={K_TRANSPOSE_BLOCK_N}",
        "GRAPH_LAYERNORM_ROW_BLOCK={LAYERNORM_ROW_BLOCK}",
        "GRAPH_LAYERNORM_ROW_GROUP_BLOCKS={LAYERNORM_ROW_GROUP_BLOCKS}",
        "GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK={LAYERNORM_INTERNAL_ROW_BLOCK}",
        "GRAPH_SOFTMAX_ROW_BLOCK={SOFTMAX_ROW_BLOCK}",
        "GRAPH_SOFTMAX_ROW_GROUP_BLOCKS={SOFTMAX_ROW_GROUP_BLOCKS}",
        "GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK={SOFTMAX_INTERNAL_ROW_BLOCK}",
        "GRAPH_CAUSAL={CAUSAL}",
        "GRAPH_FOLD_ATTENTION_SCALE_IN_Q={FOLD_ATTENTION_SCALE_IN_Q}",
        "GRAPH_CHECK_RESULT={CHECK_RESULT}",
        "GRAPH_CHECK_INTERMEDIATES={CHECK_INTERMEDIATES}",
        "GRAPH_FLUSH_BEFORE_ROI={FLUSH_BEFORE_ROI}",
        "GRAPH_TRACE_PROGRESS={TRACE_PROGRESS}",
        "GRAPH_DUMP_KERNEL_STATS={DUMP_KERNEL_STATS}",
    ]

    lines: list[str] = [
        "[graph]",
        f'name = "{GRAPH_NAME}"',
        (
            f'description = "{"Eight-head" if CURRENT_CHECKED_IN_MH8_LAYOUT else f"{HEADS}-head"} causal decoder-block graph using canonical '
            'attention per head: layer_norm -> per-head q/k/v -> k_transpose -> '
            'qk -> causal softmax -> pv -> per-head o_proj -> head-sum -> '
            'residual_add -> layer_norm -> ffn_up -> activation -> ffn_down -> '
            'residual_add. The attention scale is folded into each Q projection '
            'weight."'
        ),
        "",
        "[harness]",
        'source = "harness.c"',
        "",
        "[harness.params]",
    ]
    for key, value in harness_params.items():
        lines.append(f"{key} = {toml_value(value)}")
    write_table(lines, "harness.build", {"c_macros": c_macros})
    write_table(lines, "harness.cache.params", cache_harness_params)
    write_table(lines, "harness.spm.params", {
        key: harness_params[key]
        for key in (
            "QKV_BLOCK_M",
            "QKV_BLOCK_N",
            "QK_BLOCK_M",
            "QK_BLOCK_N",
            "PV_BLOCK_M",
            "PV_BLOCK_N",
            "O_PROJ_BLOCK_M",
            "O_PROJ_BLOCK_N",
            "FFN_UP_BLOCK_M",
            "FFN_UP_BLOCK_N",
            "FFN_DOWN_BLOCK_M",
            "FFN_DOWN_BLOCK_N",
            "LAYERNORM_ROW_BLOCK",
            "LAYERNORM_ROW_GROUP_BLOCKS",
            "LAYERNORM_INTERNAL_ROW_BLOCK",
            "SOFTMAX_ROW_BLOCK",
            "SOFTMAX_ROW_GROUP_BLOCKS",
            "SOFTMAX_INTERNAL_ROW_BLOCK",
        )
    })

    for tensor, values in tensors.items():
        write_table(lines, f"tensors.{tensor}", values)

    for node, values in nodes.items():
        write_table(lines, f"nodes.{node}", {
            k: v for k, v in values.items()
            if k not in {"params", "env", "cache", "spm"}
        })
        if "env" in values:
            write_table(lines, f"nodes.{node}.env", values["env"])
        write_table(lines, f"nodes.{node}.params", values["params"])
        for mode in ("cache", "spm"):
            mode_cfg = values.get(mode)
            if not isinstance(mode_cfg, dict):
                continue
            mode_values = {
                k: v for k, v in mode_cfg.items()
                if k not in {"params", "env"}
            }
            if mode_values:
                write_table(lines, f"nodes.{node}.{mode}", mode_values)
            if mode_cfg.get("params"):
                write_table(lines, f"nodes.{node}.{mode}.params", mode_cfg["params"])
            if mode_cfg.get("env"):
                write_table(lines, f"nodes.{node}.{mode}.env", mode_cfg["env"])

    if CURRENT_CHECKED_IN_MH8_LAYOUT:
        large_description = (
            "Large eight-head canonical decoder block at "
            f"s{SEQ} d{D_MODEL} heads{HEADS} head_dim{HEAD_DIM} ffn{FFN_DIM}."
        )
        profile_description = (
            "Profiling preset for the large eight-head canonical decoder block. "
            "The harness dumps and resets gem5 stats after each kernel launch."
        )
    else:
        large_description = (
            f"Canonical decoder block at s{SEQ} d{D_MODEL} heads{HEADS} "
            f"head_dim{HEAD_DIM} ffn{FFN_DIM}."
        )
        profile_description = (
            f"Profiling preset for the {HEADS}-head canonical decoder block. "
            "The harness dumps and resets gem5 stats after each kernel launch."
        )
    write_table(lines, "presets.large.graph", {"description": large_description})
    write_table(
        lines,
        "presets.large_profile.graph",
        {"description": profile_description},
    )
    write_table(lines, "presets.large_profile.harness.params", {
        "DUMP_KERNEL_STATS": 1,
        "CHECK_RESULT": 0,
    })
    return "\n".join(lines) + "\n"


def symbol_list(prefix: str, suffix: str) -> str:
    return ", ".join(f"{prefix}_h{h}_{suffix}" for h in range(HEADS))


def render_harness_c() -> str:
    sum_plan = head_sum_plan()
    sum_allocs = "\n    ".join(
        f'float *{record["output"]} = (float *){record["name"]}_alloc(2, model_bytes);'
        for record in sum_plan
    )
    sum_failed_checks = "\n        ".join(
        f'|| !{record["output"]}' for record in sum_plan
    )
    sum_zeroes = "\n    ".join(
        f'memset({record["output"]}, 0, model_bytes);'
        for record in sum_plan
    )
    sum_frees = "\n    ".join(
        f'{record["name"]}_free_all();'
        for record in sum_plan
    )
    sum_launches = "\n    ".join(
        f'{record["name"]}_launch(MODEL_GRID_X, 1, 1, {c_expr(str(record["left"]))}, {c_expr(str(record["right"]))}, {record["output"]});\n'
        f'    KERNEL_DONE("{record["name"]}");'
        for record in sum_plan
    )
    sum_ref_decls = "\n        ".join(
        f'float *{sum_ref_name(str(record["output"]))} = (float *)malloc(model_bytes);'
        for record in sum_plan
    )
    sum_ref_failed_checks = "\n            ".join(
        f'|| !{sum_ref_name(str(record["output"]))}' for record in sum_plan
    )
    sum_ref_frees = "\n            ".join(
        f'free({sum_ref_name(str(record["output"]))});'
        for record in sum_plan
    )
    sum_ref_compute = "\n        ".join(
        f'reference_residual_add({ref_expr(str(record["left"]))}, {ref_expr(str(record["right"]))}, {sum_ref_name(str(record["output"]))}, MODEL_ELEMS);'
        for record in sum_plan
    )
    q_allocs = symbol_list("q_proj", "alloc")
    k_allocs = symbol_list("k_proj", "alloc")
    v_allocs = symbol_list("v_proj", "alloc")
    kt_allocs = symbol_list("k_transpose", "alloc")
    qk_allocs = symbol_list("qk", "alloc")
    sm_allocs = symbol_list("softmax", "alloc")
    pv_allocs = symbol_list("pv", "alloc")
    o_allocs = symbol_list("o_proj", "alloc")
    q_launches = symbol_list("q_proj", "launch")
    k_launches = symbol_list("k_proj", "launch")
    v_launches = symbol_list("v_proj", "launch")
    kt_launches = symbol_list("k_transpose", "launch")
    qk_launches = symbol_list("qk", "launch")
    sm_launches = symbol_list("softmax", "launch")
    pv_launches = symbol_list("pv", "launch")
    o_launches = symbol_list("o_proj", "launch")
    q_frees = symbol_list("q_proj", "free_all")
    k_frees = symbol_list("k_proj", "free_all")
    v_frees = symbol_list("v_proj", "free_all")
    kt_frees = symbol_list("k_transpose", "free_all")
    qk_frees = symbol_list("qk", "free_all")
    sm_frees = symbol_list("softmax", "free_all")
    pv_frees = symbol_list("pv", "free_all")
    o_frees = symbol_list("o_proj", "free_all")
    head_guard = (
        f"""#if GRAPH_HEADS != {HEADS}
#error "{GRAPH_NAME} harness expects GRAPH_HEADS == {HEADS}"
#endif
"""
        if CURRENT_CHECKED_IN_MH8_LAYOUT
        else ""
    )

    return f"""#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_nodes.h"
#include "libspm.h"

#ifndef GRAPH_SEQ
#define GRAPH_SEQ {SEQ}
#endif
#ifndef GRAPH_HEADS
#define GRAPH_HEADS {HEADS}
#endif
#ifndef GRAPH_D_MODEL
#define GRAPH_D_MODEL {D_MODEL}
#endif
#ifndef GRAPH_HEAD_DIM
#define GRAPH_HEAD_DIM {HEAD_DIM}
#endif
#ifndef GRAPH_FFN_DIM
#define GRAPH_FFN_DIM {FFN_DIM}
#endif
#ifndef GRAPH_BLOCK
#define GRAPH_BLOCK 64
#endif
#ifndef GRAPH_QKV_BLOCK_M
#define GRAPH_QKV_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_QKV_BLOCK_N
#define GRAPH_QKV_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_QK_BLOCK_M
#define GRAPH_QK_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_QK_BLOCK_N
#define GRAPH_QK_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_PV_BLOCK_M
#define GRAPH_PV_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_PV_BLOCK_N
#define GRAPH_PV_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_O_PROJ_BLOCK_M
#define GRAPH_O_PROJ_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_O_PROJ_BLOCK_N
#define GRAPH_O_PROJ_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_FFN_UP_BLOCK_M
#define GRAPH_FFN_UP_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_FFN_UP_BLOCK_N
#define GRAPH_FFN_UP_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_FFN_DOWN_BLOCK_M
#define GRAPH_FFN_DOWN_BLOCK_M {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_M"]}
#endif
#ifndef GRAPH_FFN_DOWN_BLOCK_N
#define GRAPH_FFN_DOWN_BLOCK_N {TRISPM_MATMUL_PARAMS["BLOCK_SIZE_N"]}
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_M
#define GRAPH_K_TRANSPOSE_BLOCK_M 16
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_N
#define GRAPH_K_TRANSPOSE_BLOCK_N 16
#endif
#ifndef GRAPH_LAYERNORM_ROW_BLOCK
#define GRAPH_LAYERNORM_ROW_BLOCK 1
#endif
#ifndef GRAPH_LAYERNORM_ROW_GROUP_BLOCKS
#define GRAPH_LAYERNORM_ROW_GROUP_BLOCKS 1
#endif
#ifndef GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#define GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK 0
#endif
#ifndef GRAPH_SOFTMAX_ROW_BLOCK
#define GRAPH_SOFTMAX_ROW_BLOCK {TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_ROW_BLOCK"]}
#endif
#ifndef GRAPH_SOFTMAX_ROW_GROUP_BLOCKS
#define GRAPH_SOFTMAX_ROW_GROUP_BLOCKS {TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_ROW_GROUP_BLOCKS"]}
#endif
#ifndef GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK {TRISPM_SOFTMAX_ROW_BLOCK_PARAMS["SPM_INTERNAL_ROW_BLOCK"]}
#endif
#ifndef GRAPH_CAUSAL
#define GRAPH_CAUSAL 1
#endif
#ifndef GRAPH_FOLD_ATTENTION_SCALE_IN_Q
#define GRAPH_FOLD_ATTENTION_SCALE_IN_Q 1
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 0
#endif
#ifndef GRAPH_CHECK_INTERMEDIATES
#define GRAPH_CHECK_INTERMEDIATES 0
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif
#ifndef GRAPH_TRACE_PROGRESS
#define GRAPH_TRACE_PROGRESS 0
#endif
#ifndef GRAPH_DUMP_KERNEL_STATS
#define GRAPH_DUMP_KERNEL_STATS 0
#endif

#if GRAPH_TRACE_PROGRESS
static void trace_step(const char *label)
{{
    void *probe;
    printf("TRACE %s\\n", label);
    fflush(stdout);
    probe = malloc(16);
    if (probe)
        free(probe);
    printf("TRACE_HEAP_OK %s\\n", label);
    fflush(stdout);
}}
#define TRACE_STEP(label) trace_step(label)
#else
#define TRACE_STEP(label) do {{ }} while (0)
#endif

#if GRAPH_DUMP_KERNEL_STATS
static void dump_kernel_stats(const char *label)
{{
    printf("KERNEL_STATS %s\\n", label);
    fflush(stdout);
    m5_dump_reset_stats(0, 0);
}}
#define KERNEL_DONE(label) do {{ TRACE_STEP(label); dump_kernel_stats(label); }} while (0)
#else
#define KERNEL_DONE(label) TRACE_STEP(label)
#endif

{head_guard}#if GRAPH_D_MODEL != (GRAPH_HEADS * GRAPH_HEAD_DIM)
#error "{GRAPH_NAME} expects GRAPH_D_MODEL == GRAPH_HEADS * GRAPH_HEAD_DIM"
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MODEL_ELEMS (GRAPH_SEQ * GRAPH_D_MODEL)
#define HEAD_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define FFN_ELEMS (GRAPH_SEQ * GRAPH_FFN_DIM)
#define MODEL_GRID_X CEIL_DIV(MODEL_ELEMS, GRAPH_BLOCK)
#define FFN_GRID_X CEIL_DIV(FFN_ELEMS, GRAPH_BLOCK)
#define MATMUL_GRID(m, n, bm, bn) (CEIL_DIV((m), (bm)) * CEIL_DIV((n), (bn)))
#if GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#define LAYERNORM_GRID_X (GRAPH_SEQ / (GRAPH_LAYERNORM_ROW_BLOCK * GRAPH_LAYERNORM_ROW_GROUP_BLOCKS))
#else
#define LAYERNORM_GRID_X GRAPH_SEQ
#endif
#if GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define SOFTMAX_GRID_X (GRAPH_SEQ / (GRAPH_SOFTMAX_ROW_BLOCK * GRAPH_SOFTMAX_ROW_GROUP_BLOCKS))
#else
#define SOFTMAX_GRID_X GRAPH_SEQ
#endif

#if GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#if GRAPH_LAYERNORM_ROW_BLOCK <= 1
#error "SPM internal row-block layer_norm requires GRAPH_LAYERNORM_ROW_BLOCK > 1"
#endif
#if GRAPH_LAYERNORM_ROW_GROUP_BLOCKS <= 0
#error "SPM internal row-block layer_norm requires GRAPH_LAYERNORM_ROW_GROUP_BLOCKS > 0"
#endif
#if (GRAPH_SEQ % (GRAPH_LAYERNORM_ROW_BLOCK * GRAPH_LAYERNORM_ROW_GROUP_BLOCKS)) != 0
#error "SPM internal row-block layer_norm requires GRAPH_SEQ divisible by row-block * row-group-blocks"
#endif
#endif

#if GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#if GRAPH_SOFTMAX_ROW_BLOCK <= 1
#error "SPM internal row-block softmax requires GRAPH_SOFTMAX_ROW_BLOCK > 1"
#endif
#if GRAPH_SOFTMAX_ROW_GROUP_BLOCKS <= 0
#error "SPM internal row-block softmax requires GRAPH_SOFTMAX_ROW_GROUP_BLOCKS > 0"
#endif
#if (GRAPH_SEQ % (GRAPH_SOFTMAX_ROW_BLOCK * GRAPH_SOFTMAX_ROW_GROUP_BLOCKS)) != 0
#error "SPM internal row-block softmax requires GRAPH_SEQ divisible by row-block * row-group-blocks"
#endif
#endif

typedef void *(*alloc_fn)(int, size_t);
typedef void (*free_fn)(void);
typedef void (*binary_launch_fn)(int32_t, int32_t, int32_t, void *, void *, void *);
typedef void (*unary_launch_fn)(int32_t, int32_t, int32_t, void *, void *);

static alloc_fn q_proj_allocs[GRAPH_HEADS] = {{ {q_allocs} }};
static alloc_fn k_proj_allocs[GRAPH_HEADS] = {{ {k_allocs} }};
static alloc_fn v_proj_allocs[GRAPH_HEADS] = {{ {v_allocs} }};
static alloc_fn k_transpose_allocs[GRAPH_HEADS] = {{ {kt_allocs} }};
static alloc_fn qk_allocs[GRAPH_HEADS] = {{ {qk_allocs} }};
static alloc_fn softmax_allocs[GRAPH_HEADS] = {{ {sm_allocs} }};
static alloc_fn pv_allocs[GRAPH_HEADS] = {{ {pv_allocs} }};
static alloc_fn o_proj_allocs[GRAPH_HEADS] = {{ {o_allocs} }};

static binary_launch_fn q_proj_launches[GRAPH_HEADS] = {{ {q_launches} }};
static binary_launch_fn k_proj_launches[GRAPH_HEADS] = {{ {k_launches} }};
static binary_launch_fn v_proj_launches[GRAPH_HEADS] = {{ {v_launches} }};
static unary_launch_fn k_transpose_launches[GRAPH_HEADS] = {{ {kt_launches} }};
static binary_launch_fn qk_launches[GRAPH_HEADS] = {{ {qk_launches} }};
static unary_launch_fn softmax_launches[GRAPH_HEADS] = {{ {sm_launches} }};
static binary_launch_fn pv_launches[GRAPH_HEADS] = {{ {pv_launches} }};
static binary_launch_fn o_proj_launches[GRAPH_HEADS] = {{ {o_launches} }};

static free_fn q_proj_frees[GRAPH_HEADS] = {{ {q_frees} }};
static free_fn k_proj_frees[GRAPH_HEADS] = {{ {k_frees} }};
static free_fn v_proj_frees[GRAPH_HEADS] = {{ {v_frees} }};
static free_fn k_transpose_frees[GRAPH_HEADS] = {{ {kt_frees} }};
static free_fn qk_frees[GRAPH_HEADS] = {{ {qk_frees} }};
static free_fn softmax_frees[GRAPH_HEADS] = {{ {sm_frees} }};
static free_fn pv_frees[GRAPH_HEADS] = {{ {pv_frees} }};
static free_fn o_proj_frees[GRAPH_HEADS] = {{ {o_frees} }};

static void init_matrix(float *x, int rows, int cols, int salt)
{{
    for (int i = 0; i < rows * cols; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}}

static void scale_matrix(float *x, int elems, float scale)
{{
    for (int i = 0; i < elems; i++)
        x[i] *= scale;
}}

static void init_layer_norm_params(float *gamma, float *beta, int cols, int salt)
{{
    for (int j = 0; j < cols; j++) {{
        gamma[j] = 0.75f + (float)((j + salt) % 7) * 0.05f;
        beta[j] = (float)(((j + salt) % 5) - 2) * 0.025f;
    }}
}}

static void reference_layer_norm(
    const float *x, const float *gamma, const float *beta, float *out)
{{
    for (int i = 0; i < GRAPH_SEQ; i++) {{
        float mean = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++)
            mean += x[i * GRAPH_D_MODEL + j];
        mean /= GRAPH_D_MODEL;

        float var = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++) {{
            float d = x[i * GRAPH_D_MODEL + j] - mean;
            var += d * d;
        }}
        var /= GRAPH_D_MODEL;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < GRAPH_D_MODEL; j++) {{
            out[i * GRAPH_D_MODEL + j] =
                (x[i * GRAPH_D_MODEL + j] - mean) * inv_std * gamma[j] + beta[j];
        }}
    }}
}}

static void reference_matmul(
    const float *a, const float *b, float *c, int m, int n, int k)
{{
    for (int i = 0; i < m; i++) {{
        for (int j = 0; j < n; j++) {{
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++)
                sum += a[i * k + kk] * b[kk * n + j];
            c[i * n + j] = sum;
        }}
    }}
}}

static void reference_transpose(const float *x, float *out, int rows, int cols)
{{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j * rows + i] = x[i * cols + j];
}}

static void reference_softmax(const float *x, float *out)
{{
    for (int i = 0; i < GRAPH_SEQ; i++) {{
        float max_v = -3.4028234663852886e38f;
        for (int j = 0; j < GRAPH_SEQ; j++) {{
            if (GRAPH_CAUSAL && j > i)
                continue;
            float v = x[i * GRAPH_SEQ + j];
            if (v > max_v)
                max_v = v;
        }}

        float denom = 0.0f;
        for (int j = 0; j < GRAPH_SEQ; j++) {{
            float e = 0.0f;
            if (!GRAPH_CAUSAL || j <= i)
                e = expf(x[i * GRAPH_SEQ + j] - max_v);
            out[i * GRAPH_SEQ + j] = e;
            denom += e;
        }}
        for (int j = 0; j < GRAPH_SEQ; j++)
            out[i * GRAPH_SEQ + j] /= denom;
    }}
}}

static void reference_residual_add(
    const float *x, const float *residual, float *out, int elems)
{{
    for (int i = 0; i < elems; i++)
        out[i] = x[i] + residual[i];
}}

static void reference_activation(const float *x, float *out, int elems)
{{
    for (int i = 0; i < elems; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}}

static int check_tensor(const char *name, const float *got, const float *ref,
                        int elems, float tolerance)
{{
    int errors = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < elems; i++) {{
        float abs_err = fabsf(got[i] - ref[i]);
        float rel_err = abs_err / fmaxf(fabsf(ref[i]), 1e-6f);
        if (abs_err > max_abs)
            max_abs = abs_err;
        if (rel_err > max_rel)
            max_rel = rel_err;
        if (abs_err > tolerance) {{
            if (errors < 10) {{
                printf("MISMATCH %s[%d]: got %.6f, expected %.6f\\n",
                       name, i, got[i], ref[i]);
            }}
            errors++;
        }}
    }}
    if (errors == 0)
        printf("PASS %s: max_abs=%.6g max_rel=%.6g\\n",
               name, (double)max_abs, (double)max_rel);
    else
        printf("FAIL %s: %d / %d mismatches max_abs=%.6g max_rel=%.6g\\n",
               name, errors, elems, (double)max_abs, (double)max_rel);
    return errors;
}}

static void free_head_arrays(float *a[GRAPH_HEADS])
{{
    for (int h = 0; h < GRAPH_HEADS; h++)
        free(a[h]);
}}

static void free_all_nodes(void)
{{
    layer_norm_free_all();
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        q_proj_frees[h]();
        k_proj_frees[h]();
        v_proj_frees[h]();
        k_transpose_frees[h]();
        qk_frees[h]();
        softmax_frees[h]();
        pv_frees[h]();
        o_proj_frees[h]();
    }}
    {sum_frees}
    attn_residual_add_free_all();
    ln2_free_all();
    ffn_up_free_all();
    ffn_activation_free_all();
    ffn_down_free_all();
    final_residual_add_free_all();
}}

static void label_head(char *buf, size_t n, const char *prefix, int h)
{{
    snprintf(buf, n, "%s_h%d", prefix, h);
}}

int main(void)
{{
    printf("graph {GRAPH_NAME}: SEQ=%d HEADS=%d D_MODEL=%d HEAD_DIM=%d FFN_DIM=%d canonical_attention=1 causal=%d fold_scale=%d check=%d flush=%d kernel_stats=%d\\n",
           GRAPH_SEQ, GRAPH_HEADS, GRAPH_D_MODEL, GRAPH_HEAD_DIM, GRAPH_FFN_DIM,
           GRAPH_CAUSAL, GRAPH_FOLD_ATTENTION_SCALE_IN_Q, GRAPH_CHECK_RESULT,
           GRAPH_FLUSH_BEFORE_ROI, GRAPH_DUMP_KERNEL_STATS);

    const float attention_scale = 1.0f / sqrtf((float)GRAPH_HEAD_DIM);
    const size_t model_bytes = (size_t)MODEL_ELEMS * sizeof(float);
    const size_t head_bytes = (size_t)HEAD_ELEMS * sizeof(float);
    const size_t score_bytes = (size_t)SCORE_ELEMS * sizeof(float);
    const size_t ffn_bytes = (size_t)FFN_ELEMS * sizeof(float);
    const size_t ln_param_bytes = (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t qkv_weight_bytes =
        (size_t)GRAPH_D_MODEL * (size_t)GRAPH_HEAD_DIM * sizeof(float);
    const size_t o_weight_bytes =
        (size_t)GRAPH_HEAD_DIM * (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t ffn_up_weight_bytes =
        (size_t)GRAPH_D_MODEL * (size_t)GRAPH_FFN_DIM * sizeof(float);
    const size_t ffn_down_weight_bytes =
        (size_t)GRAPH_FFN_DIM * (size_t)GRAPH_D_MODEL * sizeof(float);

    float *x_shadow = (float *)malloc(model_bytes);
    float *gamma_shadow = (float *)malloc(ln_param_bytes);
    float *beta_shadow = (float *)malloc(ln_param_bytes);
    float *wq_shadow[GRAPH_HEADS] = {{0}};
    float *wk_shadow[GRAPH_HEADS] = {{0}};
    float *wv_shadow[GRAPH_HEADS] = {{0}};
    float *wo_shadow[GRAPH_HEADS] = {{0}};
    float *residual_shadow = (float *)malloc(model_bytes);
    float *gamma2_shadow = (float *)malloc(ln_param_bytes);
    float *beta2_shadow = (float *)malloc(ln_param_bytes);
    float *w_up_shadow = (float *)malloc(ffn_up_weight_bytes);
    float *w_down_shadow = (float *)malloc(ffn_down_weight_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        wq_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wk_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wv_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wo_shadow[h] = (float *)malloc(o_weight_bytes);
    }}
    TRACE_STEP("shadow_alloc");

    float *x = (float *)layer_norm_alloc(0, model_bytes);
    float *gamma = (float *)layer_norm_alloc(1, ln_param_bytes);
    float *beta = (float *)layer_norm_alloc(2, ln_param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, model_bytes);

    float *wq[GRAPH_HEADS];
    float *wk[GRAPH_HEADS];
    float *wv[GRAPH_HEADS];
    float *wo[GRAPH_HEADS];
    float *q[GRAPH_HEADS];
    float *k[GRAPH_HEADS];
    float *v[GRAPH_HEADS];
    float *k_t[GRAPH_HEADS];
    float *scores[GRAPH_HEADS];
    float *probs[GRAPH_HEADS];
    float *attn[GRAPH_HEADS];
    float *attn_proj[GRAPH_HEADS];
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        wq[h] = (float *)q_proj_allocs[h](1, qkv_weight_bytes);
        wk[h] = (float *)k_proj_allocs[h](1, qkv_weight_bytes);
        wv[h] = (float *)v_proj_allocs[h](1, qkv_weight_bytes);
        q[h] = (float *)q_proj_allocs[h](2, head_bytes);
        k[h] = (float *)k_proj_allocs[h](2, head_bytes);
        v[h] = (float *)v_proj_allocs[h](2, head_bytes);
        k_t[h] = (float *)k_transpose_allocs[h](1, head_bytes);
        scores[h] = (float *)qk_allocs[h](2, score_bytes);
        probs[h] = (float *)softmax_allocs[h](1, score_bytes);
        attn[h] = (float *)pv_allocs[h](2, head_bytes);
        wo[h] = (float *)o_proj_allocs[h](1, o_weight_bytes);
        attn_proj[h] = (float *)o_proj_allocs[h](2, model_bytes);
    }}

    {sum_allocs}
    float *residual = (float *)attn_residual_add_alloc(1, model_bytes);
    float *resid_out = (float *)attn_residual_add_alloc(2, model_bytes);
    float *gamma2 = (float *)ln2_alloc(1, ln_param_bytes);
    float *beta2 = (float *)ln2_alloc(2, ln_param_bytes);
    float *ln2_out = (float *)ln2_alloc(3, model_bytes);
    float *w_up = (float *)ffn_up_alloc(1, ffn_up_weight_bytes);
    float *ffn_hidden = (float *)ffn_up_alloc(2, ffn_bytes);
    float *ffn_act = (float *)ffn_activation_alloc(1, ffn_bytes);
    float *w_down = (float *)ffn_down_alloc(1, ffn_down_weight_bytes);
    float *ffn_out = (float *)ffn_down_alloc(2, model_bytes);
    float *block_out = (float *)final_residual_add_alloc(2, model_bytes);
    TRACE_STEP("node_alloc");

    int malloc_failed =
        !x_shadow || !gamma_shadow || !beta_shadow || !residual_shadow ||
        !gamma2_shadow || !beta2_shadow || !w_up_shadow || !w_down_shadow ||
        !x || !gamma || !beta || !ln_out
        {sum_failed_checks}
        || !residual || !resid_out || !gamma2 || !beta2 ||
        !ln2_out || !w_up || !ffn_hidden || !ffn_act || !w_down ||
        !ffn_out || !block_out;
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        malloc_failed = malloc_failed || !wq_shadow[h] || !wk_shadow[h] ||
                        !wv_shadow[h] || !wo_shadow[h] || !wq[h] || !wk[h] ||
                        !wv[h] || !wo[h] || !q[h] || !k[h] || !v[h] ||
                        !k_t[h] || !scores[h] || !probs[h] || !attn[h] ||
                        !attn_proj[h];
    }}
    if (malloc_failed) {{
        fprintf(stderr, "malloc failed\\n");
        free_all_nodes();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free_head_arrays(wq_shadow);
        free_head_arrays(wk_shadow);
        free_head_arrays(wv_shadow);
        free_head_arrays(wo_shadow);
        free(residual_shadow);
        free(gamma2_shadow);
        free(beta2_shadow);
        free(w_up_shadow);
        free(w_down_shadow);
        return 1;
    }}

    init_matrix(x_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 1);
    init_layer_norm_params(gamma_shadow, beta_shadow, GRAPH_D_MODEL, 0);
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        init_matrix(wq_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 2 + h * 10);
        if (GRAPH_FOLD_ATTENTION_SCALE_IN_Q)
            scale_matrix(wq_shadow[h], GRAPH_D_MODEL * GRAPH_HEAD_DIM, attention_scale);
        init_matrix(wk_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 3 + h * 10);
        init_matrix(wv_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 4 + h * 10);
        init_matrix(wo_shadow[h], GRAPH_HEAD_DIM, GRAPH_D_MODEL, 5 + h * 10);
    }}
    init_matrix(residual_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 6);
    init_layer_norm_params(gamma2_shadow, beta2_shadow, GRAPH_D_MODEL, 3);
    init_matrix(w_up_shadow, GRAPH_D_MODEL, GRAPH_FFN_DIM, 7);
    init_matrix(w_down_shadow, GRAPH_FFN_DIM, GRAPH_D_MODEL, 8);
    TRACE_STEP("init_inputs");

    flush_caches();
    publish_input(x, x_shadow, model_bytes);
    publish_input(gamma, gamma_shadow, ln_param_bytes);
    publish_input(beta, beta_shadow, ln_param_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        publish_input(wq[h], wq_shadow[h], qkv_weight_bytes);
        publish_input(wk[h], wk_shadow[h], qkv_weight_bytes);
        publish_input(wv[h], wv_shadow[h], qkv_weight_bytes);
        publish_input(wo[h], wo_shadow[h], o_weight_bytes);
    }}
    publish_input(residual, residual_shadow, model_bytes);
    publish_input(gamma2, gamma2_shadow, ln_param_bytes);
    publish_input(beta2, beta2_shadow, ln_param_bytes);
    publish_input(w_up, w_up_shadow, ffn_up_weight_bytes);
    publish_input(w_down, w_down_shadow, ffn_down_weight_bytes);
    TRACE_STEP("publish_inputs");

    memset(ln_out, 0, model_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        memset(q[h], 0, head_bytes);
        memset(k[h], 0, head_bytes);
        memset(v[h], 0, head_bytes);
        memset(k_t[h], 0, head_bytes);
        memset(scores[h], 0, score_bytes);
        memset(probs[h], 0, score_bytes);
        memset(attn[h], 0, head_bytes);
        memset(attn_proj[h], 0, model_bytes);
    }}
    {sum_zeroes}
    memset(resid_out, 0, model_bytes);
    memset(ln2_out, 0, model_bytes);
    memset(ffn_hidden, 0, ffn_bytes);
    memset(ffn_act, 0, ffn_bytes);
    memset(ffn_out, 0, model_bytes);
    memset(block_out, 0, model_bytes);
    TRACE_STEP("zero_outputs");

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    char label[64];
    m5_reset_stats(0, 0);

    layer_norm_launch(LAYERNORM_GRID_X, 1, 1, x, gamma, beta, ln_out);
    KERNEL_DONE("layer_norm");
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        label_head(label, sizeof(label), "q_proj", h);
        q_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wq[h], q[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "k_proj", h);
        k_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wk[h], k[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "v_proj", h);
        v_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wv[h], v[h]);
        KERNEL_DONE(label);
    }}
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        label_head(label, sizeof(label), "k_transpose", h);
        k_transpose_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_K_TRANSPOSE_BLOCK_M, GRAPH_K_TRANSPOSE_BLOCK_N), 1, 1, k[h], k_t[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "qk", h);
        qk_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N), 1, 1, q[h], k_t[h], scores[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "softmax", h);
        softmax_launches[h](SOFTMAX_GRID_X, 1, 1, scores[h], probs[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "pv", h);
        pv_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N), 1, 1, probs[h], v[h], attn[h]);
        KERNEL_DONE(label);
    }}
    for (int h = 0; h < GRAPH_HEADS; h++) {{
        label_head(label, sizeof(label), "o_proj", h);
        o_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_O_PROJ_BLOCK_M, GRAPH_O_PROJ_BLOCK_N), 1, 1, attn[h], wo[h], attn_proj[h]);
        KERNEL_DONE(label);
    }}
    {sum_launches}
    attn_residual_add_launch(MODEL_GRID_X, 1, 1, attn_sum, residual, resid_out);
    KERNEL_DONE("attn_residual_add");
    ln2_launch(LAYERNORM_GRID_X, 1, 1, resid_out, gamma2, beta2, ln2_out);
    KERNEL_DONE("ln2");
    ffn_up_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_FFN_UP_BLOCK_M, GRAPH_FFN_UP_BLOCK_N), 1, 1, ln2_out, w_up, ffn_hidden);
    KERNEL_DONE("ffn_up");
    ffn_activation_launch(FFN_GRID_X, 1, 1, ffn_hidden, ffn_act);
    KERNEL_DONE("ffn_activation");
    ffn_down_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DOWN_BLOCK_M, GRAPH_FFN_DOWN_BLOCK_N), 1, 1, ffn_act, w_down, ffn_out);
    KERNEL_DONE("ffn_down");
    final_residual_add_launch(MODEL_GRID_X, 1, 1, ffn_out, resid_out, block_out);
    KERNEL_DONE("final_residual_add");

#if !GRAPH_DUMP_KERNEL_STATS
    m5_dump_stats(0, 0);
    TRACE_STEP("after_dump_stats");
#else
    TRACE_STEP("after_kernel_stats");
#endif

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {{
        float *ln_ref = (float *)malloc(model_bytes);
        float *q_ref[GRAPH_HEADS] = {{0}};
        float *k_ref[GRAPH_HEADS] = {{0}};
        float *v_ref[GRAPH_HEADS] = {{0}};
        float *k_t_ref[GRAPH_HEADS] = {{0}};
        float *scores_ref[GRAPH_HEADS] = {{0}};
        float *probs_ref[GRAPH_HEADS] = {{0}};
        float *attn_ref[GRAPH_HEADS] = {{0}};
        float *attn_proj_ref[GRAPH_HEADS] = {{0}};
        {sum_ref_decls}
        float *resid_ref = (float *)malloc(model_bytes);
        float *ln2_ref = (float *)malloc(model_bytes);
        float *ffn_hidden_ref = (float *)malloc(ffn_bytes);
        float *ffn_act_ref = (float *)malloc(ffn_bytes);
        float *ffn_out_ref = (float *)malloc(model_bytes);
        float *block_ref = (float *)malloc(model_bytes);
        for (int h = 0; h < GRAPH_HEADS; h++) {{
            q_ref[h] = (float *)malloc(head_bytes);
            k_ref[h] = (float *)malloc(head_bytes);
            v_ref[h] = (float *)malloc(head_bytes);
            k_t_ref[h] = (float *)malloc(head_bytes);
            scores_ref[h] = (float *)malloc(score_bytes);
            probs_ref[h] = (float *)malloc(score_bytes);
            attn_ref[h] = (float *)malloc(head_bytes);
            attn_proj_ref[h] = (float *)malloc(model_bytes);
        }}

        int ref_malloc_failed =
            !ln_ref
            {sum_ref_failed_checks}
            || !resid_ref ||
            !ln2_ref || !ffn_hidden_ref || !ffn_act_ref || !ffn_out_ref ||
            !block_ref;
        for (int h = 0; h < GRAPH_HEADS; h++) {{
            ref_malloc_failed = ref_malloc_failed || !q_ref[h] || !k_ref[h] ||
                                !v_ref[h] || !k_t_ref[h] || !scores_ref[h] ||
                                !probs_ref[h] || !attn_ref[h] ||
                                !attn_proj_ref[h];
        }}
        if (ref_malloc_failed) {{
            fprintf(stderr, "malloc failed\\n");
            free(ln_ref);
            free_head_arrays(q_ref);
            free_head_arrays(k_ref);
            free_head_arrays(v_ref);
            free_head_arrays(k_t_ref);
            free_head_arrays(scores_ref);
            free_head_arrays(probs_ref);
            free_head_arrays(attn_ref);
            free_head_arrays(attn_proj_ref);
            {sum_ref_frees}
            free(resid_ref);
            free(ln2_ref);
            free(ffn_hidden_ref);
            free(ffn_act_ref);
            free(ffn_out_ref);
            free(block_ref);
            free_all_nodes();
            return 1;
        }}

        reference_layer_norm(x_shadow, gamma_shadow, beta_shadow, ln_ref);
        for (int h = 0; h < GRAPH_HEADS; h++) {{
            reference_matmul(ln_ref, wq_shadow[h], q_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            reference_matmul(ln_ref, wk_shadow[h], k_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            reference_matmul(ln_ref, wv_shadow[h], v_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            reference_transpose(k_ref[h], k_t_ref[h], GRAPH_SEQ, GRAPH_HEAD_DIM);
            reference_matmul(q_ref[h], k_t_ref[h], scores_ref[h],
                             GRAPH_SEQ, GRAPH_SEQ, GRAPH_HEAD_DIM);
            reference_softmax(scores_ref[h], probs_ref[h]);
            reference_matmul(probs_ref[h], v_ref[h], attn_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_SEQ);
            reference_matmul(attn_ref[h], wo_shadow[h], attn_proj_ref[h],
                             GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM);
        }}
        {sum_ref_compute}
        reference_residual_add(attn_sum_ref, residual_shadow, resid_ref, MODEL_ELEMS);
        reference_layer_norm(resid_ref, gamma2_shadow, beta2_shadow, ln2_ref);
        reference_matmul(ln2_ref, w_up_shadow, ffn_hidden_ref,
                         GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_D_MODEL);
        reference_activation(ffn_hidden_ref, ffn_act_ref, FFN_ELEMS);
        reference_matmul(ffn_act_ref, w_down_shadow, ffn_out_ref,
                         GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DIM);
        reference_residual_add(ffn_out_ref, resid_ref, block_ref, MODEL_ELEMS);

        errors += check_tensor("block_out", block_out, block_ref, MODEL_ELEMS, 1e-1f);
        if (errors == 0)
            printf("PASS: graph outputs correct\\n");
        else
            printf("FAIL: graph has %d mismatches\\n", errors);

        free(ln_ref);
        free_head_arrays(q_ref);
        free_head_arrays(k_ref);
        free_head_arrays(v_ref);
        free_head_arrays(k_t_ref);
        free_head_arrays(scores_ref);
        free_head_arrays(probs_ref);
        free_head_arrays(attn_ref);
        free_head_arrays(attn_proj_ref);
        {sum_ref_frees}
        free(resid_ref);
        free(ln2_ref);
        free(ffn_hidden_ref);
        free(ffn_act_ref);
        free(ffn_out_ref);
        free(block_ref);
    }} else {{
        printf("SKIP: graph result check disabled\\n");
    }}

    free_all_nodes();
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);
    free_head_arrays(wq_shadow);
    free_head_arrays(wk_shadow);
    free_head_arrays(wv_shadow);
    free_head_arrays(wo_shadow);
    free(residual_shadow);
    free(gamma2_shadow);
    free(beta2_shadow);
    free(w_up_shadow);
    free(w_down_shadow);

    return (errors > 0) ? 1 : 0;
}}
"""


def generate_case(cfg: dict[str, Any]) -> None:
    set_shape(
        int(cfg["seq"]),
        int(cfg["heads"]),
        int(cfg["d_model"]),
        int(cfg["head_dim"]),
        int(cfg["ffn_dim"]),
        str(cfg["name"]),
        bool(cfg["fixed_policy"]),
        bool(cfg.get("checked_in_mh8_layout", False)),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "graph.toml").write_text(render_graph_toml())
    (OUT_DIR / "harness.c").write_text(render_harness_c())
    print(f"generated {OUT_DIR / 'graph.toml'}")
    print(f"generated {OUT_DIR / 'harness.c'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(SCALED_CASES),
        default="base",
        help="predefined decoder fixture to generate",
    )
    parser.add_argument("--name", default=None, help="override graph directory/name")
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(SCALED_CASES[args.case])
    if args.name is not None:
        cfg["name"] = args.name
    if args.seq is not None:
        cfg["seq"] = args.seq
    if args.heads is not None:
        cfg["heads"] = args.heads
    if args.d_model is not None:
        cfg["d_model"] = args.d_model
    if args.head_dim is not None:
        cfg["head_dim"] = args.head_dim
    if args.ffn_dim is not None:
        cfg["ffn_dim"] = args.ffn_dim

    generate_case(cfg)


if __name__ == "__main__":
    main()
