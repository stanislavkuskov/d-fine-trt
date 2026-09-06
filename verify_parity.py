#!/usr/bin/env python3
"""
Parity and Benchmark Verification Tool (deployment_v2)

Validates numerical parity (max absolute diff, MSE, cosine similarity) and measures
latency/FPS between the baseline ONNX model and the compiled TensorRT Engine (or prepared ONNX).
"""

import argparse
import os
import sys
import time
import numpy as np
import tensorrt as trt

try:
    import torch
except ImportError:
    torch = None


def run_onnx_inference(onnx_path: str, input_tensor: np.ndarray):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    output_names = [o.name for o in session.get_outputs()]
    return dict(zip(output_names, outputs))


def run_trt_inference(engine_path: str, input_tensor: np.ndarray, warmup: int = 5, reps: int = 30):

    if torch is None:
        raise ImportError("PyTorch is required for TRT GPU buffer allocation.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine '{engine_path}'. "
            f"Note: TensorRT engines are tied to the exact TRT version and GPU architecture they were compiled on. "
            f"(Current runtime TRT version: {trt.__version__})."
        )

    context = engine.create_execution_context()
    batch_size = input_tensor.shape[0]

    # Handle TRT 10+ vs TRT 8.x API
    is_trt10 = hasattr(engine, "num_io_tensors")
    
    tensors = {}
    outputs = {}
    
    torch_dtype_map = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    if is_trt10:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            t_dtype = torch_dtype_map.get(np_dtype, torch.float32)

            if mode == trt.TensorIOMode.INPUT:
                shape = tuple(input_tensor.shape)
                context.set_input_shape(name, shape)
                t_in = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np_dtype)).to(device)
                tensors[name] = t_in
                context.set_tensor_address(name, int(t_in.data_ptr()))
            else:
                shape = tuple(context.get_tensor_shape(name))
                shape = tuple(batch_size if d == -1 else d for d in shape)
                t_out = torch.empty(shape, dtype=t_dtype, device=device)
                tensors[name] = t_out
                outputs[name] = t_out
                context.set_tensor_address(name, int(t_out.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        for _ in range(warmup):
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(reps):
            t0 = time.perf_counter()
            context.execute_async_v3(stream_handle=stream)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    else:
        bindings = [0] * engine.num_bindings
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            np_dtype = trt.nptype(engine.get_binding_dtype(i))
            t_dtype = torch_dtype_map.get(np_dtype, torch.float32)

            if engine.binding_is_input(i):
                context.set_binding_shape(i, tuple(input_tensor.shape))
                t_in = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np_dtype)).to(device)
                bindings[i] = int(t_in.data_ptr())
                tensors[name] = t_in
            else:
                shape = tuple(context.get_binding_shape(i))
                shape = tuple(batch_size if d == -1 else d for d in shape)
                t_out = torch.empty(shape, dtype=t_dtype, device=device)
                bindings[i] = int(t_out.data_ptr())
                tensors[name] = t_out
                outputs[name] = t_out

        stream = torch.cuda.current_stream().cuda_stream
        for _ in range(warmup):
            context.execute_async_v2(bindings=bindings, stream_handle=stream)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(reps):
            t0 = time.perf_counter()
            context.execute_async_v2(bindings=bindings, stream_handle=stream)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    fps = (1000.0 / mean_latency) * batch_size

    host_outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
    return host_outputs, mean_latency, fps


def compute_metrics(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()

    max_diff = float(np.max(np.abs(a - b)))
    mean_diff = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 1e-12 and norm_b > 1e-12:
        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    else:
        cosine_sim = 1.0

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mse": mse,
        "cosine_sim": cosine_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify parity and performance between ONNX and TRT.")
    parser.add_argument("--baseline", "-b", type=str, required=True, help="Baseline ONNX file")
    parser.add_argument("--target", "-t", type=str, required=True, help="Target file (.engine or .onnx)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image for real detection comparison")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection score threshold (default: 0.3)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--reps", type=int, default=30, help="Benchmark repetitions")

    args = parser.parse_args()

    print("=" * 60)
    print("D-FINE Parity & Performance Verification")
    print("=" * 60)
    print(f"Baseline : {args.baseline}")
    print(f"Target   : {args.target}")

    if args.image and os.path.exists(args.image):
        print(f"Loading real test image: {args.image}")
        from PIL import Image
        im = Image.open(args.image).convert("RGB").resize((640, 640))
        arr = np.array(im, dtype=np.float32) / 255.0
        # HWC -> CHW -> NCHW
        arr = np.transpose(arr, (2, 0, 1))
        dummy_input = np.ascontiguousarray(np.expand_dims(arr, axis=0))
    else:
        # Generate synthetic input (B, 3, 640, 640)
        np.random.seed(42)
        dummy_input = np.random.uniform(0.0, 1.0, (args.batch_size, 3, 640, 640)).astype(np.float32)

    print("\n1. Running Baseline ONNX inference...")
    baseline_out = run_onnx_inference(args.baseline, dummy_input)

    target_is_engine = args.target.endswith(".engine") or args.target.endswith(".plan")

    if target_is_engine:
        print("2. Running TensorRT Engine benchmark & inference...")
        target_out, mean_lat, fps = run_trt_inference(
            args.target, dummy_input, warmup=args.warmup, reps=args.reps
        )
        print(f"\n[Performance] Average Latency: {mean_lat:.2f} ms | Throughput: {fps:.1f} FPS")
    else:
        print("2. Running Target ONNX inference...")
        target_out = run_onnx_inference(args.target, dummy_input)

    print("\n" + "=" * 60)
    print("Numerical Parity Comparison")
    print("=" * 60)

    common_keys = set(baseline_out.keys()).intersection(set(target_out.keys()))
    if not common_keys:
        b_keys = sorted(list(baseline_out.keys()))
        t_keys = sorted(list(target_out.keys()))
        paired = list(zip(b_keys, t_keys))
    else:
        paired = [(k, k) for k in sorted(list(common_keys))]

    for b_key, t_key in paired:
        t_base = baseline_out[b_key]
        t_targ = target_out[t_key]
        metrics = compute_metrics(t_base, t_targ)

        status = "PASSED" if metrics["cosine_sim"] > 0.98 else "CHECK"
        print(f"\nOutput: '{b_key}' vs '{t_key}' [{status}]")
        print(f"  - Shape            : {t_base.shape}")
        print(f"  - Max Absolute Diff: {metrics['max_diff']:.6e}")
        print(f"  - Mean Diff (MAE)  : {metrics['mean_diff']:.6e}")
        print(f"  - MSE              : {metrics['mse']:.6e}")
        print(f"  - Cosine Similarity: {metrics['cosine_sim']:.8f}")

    # Detection-level comparison (sigmoid on logits, filter by threshold)
    logit_key = next((k for k in baseline_out if "logit" in k.lower()), None)
    box_key = next((k for k in baseline_out if "box" in k.lower()), None)

    if logit_key and box_key:
        print("\n" + "=" * 60)
        print(f"Detection Parity (Confidence Threshold >= {args.threshold})")
        print("=" * 60)
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

        b_scores = sigmoid(baseline_out[logit_key][0])  # [300, 80]
        t_scores = sigmoid(target_out[logit_key][0])

        b_max_score = np.max(b_scores, axis=-1)
        b_labels = np.argmax(b_scores, axis=-1)
        b_mask = b_max_score >= args.threshold

        t_max_score = np.max(t_scores, axis=-1)
        t_labels = np.argmax(t_scores, axis=-1)
        t_mask = t_max_score >= args.threshold

        print(f"Baseline ONNX Detections count: {int(np.sum(b_mask))}")
        print(f"Target TRT/ONNX Detections count: {int(np.sum(t_mask))}")

        top_indices = np.where(b_mask)[0]
        if len(top_indices) > 0:
            print("\nTop Confident Detections Comparison:")
            for idx in top_indices[:5]:
                b_cls = b_labels[idx]
                t_cls = t_labels[idx]
                b_box = baseline_out[box_key][0, idx]
                t_box = target_out[box_key][0, idx]
                box_iou_diff = np.max(np.abs(b_box - t_box))
                print(f"  Query #{idx}:")
                print(f"    Baseline: class={b_cls} score={b_max_score[idx]:.3f} box={np.round(b_box, 3)}")
                print(f"    Target  : class={t_cls} score={t_max_score[idx]:.3f} box={np.round(t_box, 3)} (box diff={box_iou_diff:.4f})")
        else:
            print("No detections above threshold on this input (expected for non-scene inputs).")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
