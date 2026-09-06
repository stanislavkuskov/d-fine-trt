#!/usr/bin/env python3
"""
D-FINE ONNX Graph Preparation for TensorRT (deployment_v2)

This script sanitizes and optimizes an exported D-FINE ONNX model for TensorRT (including TRT 8.5+).
It replaces non-standard or problematic operators (LayerNormalization, Gelu, GatherElements)
with standard, well-supported primitive operations while preserving numerical precision.
"""

import argparse
import os
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper


def replace_layer_norm(graph: gs.Graph) -> int:
    """
    Decomposes LayerNormalization nodes into primitive ops:
    LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta along axis [-1]
    """
    count = 0
    for n in list(graph.nodes):
        if n.op == "LayerNormalization":
            x, gamma, beta = n.inputs
            eps_val = float(n.attrs.get("epsilon", 1e-5))
            eps = gs.Constant(f"{n.name}_eps", np.array([eps_val], dtype=np.float32))
            axes = [-1]

            mean = gs.Variable(f"{n.name}_mean", dtype=x.dtype)
            cent = gs.Variable(f"{n.name}_cent", dtype=x.dtype)
            sq = gs.Variable(f"{n.name}_sq", dtype=x.dtype)
            var = gs.Variable(f"{n.name}_var", dtype=x.dtype)
            vare = gs.Variable(f"{n.name}_vare", dtype=x.dtype)
            std = gs.Variable(f"{n.name}_std", dtype=x.dtype)
            norm = gs.Variable(f"{n.name}_norm", dtype=x.dtype)
            scal = gs.Variable(f"{n.name}_scal", dtype=x.dtype)

            graph.nodes.extend([
                gs.Node(op="ReduceMean", inputs=[x], outputs=[mean], attrs={"axes": axes, "keepdims": 1}),
                gs.Node(op="Sub", inputs=[x, mean], outputs=[cent]),
                gs.Node(op="Mul", inputs=[cent, cent], outputs=[sq]),
                gs.Node(op="ReduceMean", inputs=[sq], outputs=[var], attrs={"axes": axes, "keepdims": 1}),
                gs.Node(op="Add", inputs=[var, eps], outputs=[vare]),
                gs.Node(op="Sqrt", inputs=[vare], outputs=[std]),
                gs.Node(op="Div", inputs=[cent, std], outputs=[norm]),
                gs.Node(op="Mul", inputs=[norm, gamma], outputs=[scal]),
                gs.Node(op="Add", inputs=[scal, beta], outputs=n.outputs),
            ])
            n.outputs = []
            graph.nodes.remove(n)
            count += 1
    return count


def replace_gelu(graph: gs.Graph) -> int:
    """
    Replaces Gelu nodes with the standard polynomial tanh approximation:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    count = 0
    for n in list(graph.nodes):
        if n.op == "Gelu":
            x = n.inputs[0]
            a = gs.Constant(f"{n.name}_a", np.array([0.7978845608], dtype=np.float32))  # sqrt(2/pi)
            b = gs.Constant(f"{n.name}_b", np.array([0.044715], dtype=np.float32))
            half = gs.Constant(f"{n.name}_half", np.array([0.5], dtype=np.float32))
            one = gs.Constant(f"{n.name}_one", np.array([1.0], dtype=np.float32))

            x2 = gs.Variable(f"{n.name}_x2", dtype=x.dtype)
            x3 = gs.Variable(f"{n.name}_x3", dtype=x.dtype)
            t0 = gs.Variable(f"{n.name}_t0", dtype=x.dtype)
            t1 = gs.Variable(f"{n.name}_t1", dtype=x.dtype)
            th = gs.Variable(f"{n.name}_th", dtype=x.dtype)
            tanh_out = gs.Variable(f"{n.name}_tanh", dtype=x.dtype)
            add1 = gs.Variable(f"{n.name}_add1", dtype=x.dtype)
            mul1 = gs.Variable(f"{n.name}_mul1", dtype=x.dtype)

            graph.nodes.extend([
                gs.Node(op="Mul", inputs=[x, x], outputs=[x2]),
                gs.Node(op="Mul", inputs=[x2, x], outputs=[x3]),
                gs.Node(op="Mul", inputs=[x3, b], outputs=[t0]),
                gs.Node(op="Add", inputs=[x, t0], outputs=[t1]),
                gs.Node(op="Mul", inputs=[t1, a], outputs=[th]),
                gs.Node(op="Tanh", inputs=[th], outputs=[tanh_out]),
                gs.Node(op="Add", inputs=[one, tanh_out], outputs=[add1]),
                gs.Node(op="Mul", inputs=[x, add1], outputs=[mul1]),
                gs.Node(op="Mul", inputs=[mul1, half], outputs=n.outputs),
            ])
            n.outputs = []
            graph.nodes.remove(n)
            count += 1
    return count


def fix_gather_elements(graph: gs.Graph) -> int:
    """
    Stabilizes GatherElements nodes for TensorRT:
    1. Clips index ranges dynamically to prevent out-of-bounds indexing in TRT optimizer.
    2. Enforces axis=-1 for decoder / postprocessor gather nodes.
    3. Flattens dynamic batch indexing for static anchor tensors in the decoder.
    """
    count = 0

    # 1. Decoder anchor gather: flatten batch dim of indices to keep anchor tensor static
    for n in list(graph.nodes):
        if n.op == "GatherElements" and n.name == "/net/decoder/GatherElements":
            data, idx = n.inputs
            if data.name == "net.decoder.anchors":
                orig_output = n.outputs[0]
                idx_shape = gs.Variable(f"{n.name}_idx_shape", dtype=np.int64)
                idx_flat = gs.Variable(f"{n.name}_idx_flat", dtype=idx.dtype)
                gather_flat = gs.Variable(f"{n.name}_flat_out", dtype=orig_output.dtype)
                flat_shape = gs.Constant(f"{n.name}_flat_shape", np.array([1, -1, 4], dtype=np.int64))

                graph.nodes.extend([
                    gs.Node(op="Shape", inputs=[idx], outputs=[idx_shape]),
                    gs.Node(op="Reshape", inputs=[idx, flat_shape], outputs=[idx_flat]),
                    gs.Node(op="Reshape", inputs=[gather_flat, idx_shape], outputs=[orig_output]),
                ])
                n.inputs[1] = idx_flat
                n.outputs = [gather_flat]
                count += 1

    # 2. Targeted fix for decoder and postprocessor GatherElements: force axis=-1 & clip
    for n in list(graph.nodes):
        if n.op == "GatherElements":
            is_decoder = (n.name == "/model/decoder/GatherElements")
            is_postproc = ("/postprocessor/" in (n.name or ""))

            if is_decoder or is_postproc:
                data, idx = n.inputs
                shape = gs.Variable(f"{n.name}_shape_last", dtype=np.int64)
                lastd = gs.Variable(f"{n.name}_lastd", dtype=np.int64)
                maxidx = gs.Variable(f"{n.name}_max_last", dtype=np.int64)
                neg1 = gs.Constant(f"{n.name}_neg1", np.array([-1], dtype=np.int64))
                zero = gs.Constant(f"{n.name}_zero", np.array([0], dtype=np.int64))
                one = gs.Constant(f"{n.name}_one", np.array([1], dtype=np.int64))
                idx_c = gs.Variable(f"{n.name}_idxc_last", dtype=idx.dtype)

                graph.nodes.extend([
                    gs.Node(op="Shape", inputs=[data], outputs=[shape]),
                    gs.Node(op="Gather", inputs=[shape, neg1], outputs=[lastd], attrs={"axis": 0}),
                    gs.Node(op="Sub", inputs=[lastd, one], outputs=[maxidx]),
                    gs.Node(op="Clip", inputs=[idx, zero, maxidx], outputs=[idx_c]),
                ])
                n.inputs[1] = idx_c
                n.attrs["axis"] = -1
                count += 1
            else:
                # Generic clipping along declared axis
                data, idx = n.inputs
                axis = n.attrs.get("axis", 0)
                shape = gs.Variable(f"{n.name}_shape", dtype=np.int64)
                dim = gs.Variable(f"{n.name}_dim", dtype=np.int64)
                maxidx = gs.Variable(f"{n.name}_max", dtype=np.int64)
                zero = gs.Constant(f"{n.name}_zero", np.array([0], dtype=np.int64))
                axc = gs.Constant(f"{n.name}_axis", np.array([axis], dtype=np.int64))
                one = gs.Constant(f"{n.name}_one", np.array([1], dtype=np.int64))
                idx_cl = gs.Variable(f"{n.name}_idxc", dtype=idx.dtype)

                graph.nodes.extend([
                    gs.Node(op="Shape", inputs=[data], outputs=[shape]),
                    gs.Node(op="Gather", inputs=[shape, axc], outputs=[dim], attrs={"axis": 0}),
                    gs.Node(op="Sub", inputs=[dim, one], outputs=[maxidx]),
                    gs.Node(op="Clip", inputs=[idx, zero, maxidx], outputs=[idx_cl]),
                ])
                n.inputs[1] = idx_cl
                count += 1

    return count


def prepare_dfine_onnx(input_path: str, output_path: str, target_opset: int = 17):
    print(f"[prepare_onnx] Loading input ONNX: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input ONNX not found: {input_path}")

    orig_model = onnx.load(input_path)
    graph = gs.import_onnx(orig_model)

    print("[prepare_onnx] Transforming graph...")
    ln_count = replace_layer_norm(graph)
    gelu_count = replace_gelu(graph)
    ge_count = fix_gather_elements(graph)

    print(f"  - Replaced {ln_count} LayerNormalization nodes with primitive arithmetic.")
    print(f"  - Replaced {gelu_count} Gelu nodes with polynomial Tanh approximation.")
    print(f"  - Stabilized {ge_count} GatherElements operations.")

    # Cleanup unused nodes and toposort
    graph.cleanup().toposort()

    # Export to ONNX
    prepared_model = gs.export_onnx(graph)
    prepared_model.ir_version = getattr(orig_model, "ir_version", 11)

    # Rebuild opset imports targeting requested opset (default <= 17)
    del prepared_model.opset_import[:]
    for imp in orig_model.opset_import:
        dom = imp.domain or ""
        ver = imp.version
        if dom in ("", "ai.onnx") and ver > target_opset:
            ver = target_opset
        prepared_model.opset_import.append(helper.make_operatorsetid(dom, ver))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(prepared_model, output_path)
    print(f"[prepare_onnx] Successfully saved sanitized ONNX to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize D-FINE ONNX for TensorRT deployment.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input raw D-FINE ONNX")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output prepared ONNX")
    parser.add_argument("--opset", type=int, default=17, help="Target max ONNX opset version (default: 17)")

    args = parser.parse_args()
    prepare_dfine_onnx(args.input, args.output, args.opset)
