"""Compare two ONNX models for equivalence.

Usage:
    python scripts/compare_onnx.py model_a.onnx model_b.onnx
    python scripts/compare_onnx.py model_a.onnx model_b.onnx --obs-dim 316
    python scripts/compare_onnx.py model_a.onnx model_b.onnx --num-samples 100
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_structure(a: onnx.ModelProto, b: onnx.ModelProto) -> list[str]:
    diffs = []
    ga, gb = a.graph, b.graph

    # Inputs
    inputs_a = {(i.name, tuple(d.dim_value for d in i.type.tensor_type.shape.dim)) for i in ga.input}
    inputs_b = {(i.name, tuple(d.dim_value for d in i.type.tensor_type.shape.dim)) for i in gb.input}
    if inputs_a != inputs_b:
        diffs.append(f"Inputs differ: {inputs_a} vs {inputs_b}")

    # Outputs
    outputs_a = {(o.name, tuple(d.dim_value for d in o.type.tensor_type.shape.dim)) for o in ga.output}
    outputs_b = {(o.name, tuple(d.dim_value for d in o.type.tensor_type.shape.dim)) for o in gb.output}
    if outputs_a != outputs_b:
        diffs.append(f"Outputs differ: {outputs_a} vs {outputs_b}")

    # Nodes
    nodes_a = [(n.op_type, n.name, tuple(n.input), tuple(n.output)) for n in ga.node]
    nodes_b = [(n.op_type, n.name, tuple(n.input), tuple(n.output)) for n in gb.node]
    if len(nodes_a) != len(nodes_b):
        diffs.append(f"Node count: {len(nodes_a)} vs {len(nodes_b)}")
    for i, (na, nb) in enumerate(zip(nodes_a, nodes_b)):
        if na != nb:
            diffs.append(f"Node {i}: {na} vs {nb}")
            if len(diffs) > 10:
                diffs.append("... (truncated)")
                break

    # Initializers (weights)
    inits_a = {i.name: onnx.numpy_helper.to_array(i) for i in ga.initializer}
    inits_b = {i.name: onnx.numpy_helper.to_array(i) for i in gb.initializer}
    if set(inits_a) != set(inits_b):
        only_a = set(inits_a) - set(inits_b)
        only_b = set(inits_b) - set(inits_a)
        if only_a:
            diffs.append(f"Initializers only in A: {only_a}")
        if only_b:
            diffs.append(f"Initializers only in B: {only_b}")
    for name in set(inits_a) & set(inits_b):
        if inits_a[name].shape != inits_b[name].shape:
            diffs.append(f"Initializer '{name}' shape: {inits_a[name].shape} vs {inits_b[name].shape}")
        elif not np.array_equal(inits_a[name], inits_b[name]):
            max_diff = np.max(np.abs(inits_a[name] - inits_b[name]))
            diffs.append(f"Initializer '{name}' values differ (max diff: {max_diff:.2e})")

    return diffs


def compare_metadata(a: onnx.ModelProto, b: onnx.ModelProto) -> list[str]:
    meta_a = {p.key: p.value for p in a.metadata_props}
    meta_b = {p.key: p.value for p in b.metadata_props}
    all_keys = set(meta_a) | set(meta_b)
    diffs = []
    for k in sorted(all_keys):
        va, vb = meta_a.get(k), meta_b.get(k)
        if va != vb:
            diffs.append(f"Metadata '{k}': {va!r} vs {vb!r}")
    return diffs


def compare_inference(path_a: str, path_b: str, num_samples: int, obs_dim: int | None) -> dict:
    sess_a = ort.InferenceSession(path_a, providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(path_b, providers=["CPUExecutionProvider"])

    input_name = sess_a.get_inputs()[0].name
    if obs_dim is None:
        obs_dim = sess_a.get_inputs()[0].shape[-1]

    max_diffs = []
    for _ in range(num_samples):
        x = np.random.randn(1, obs_dim).astype(np.float32)
        out_a = sess_a.run(None, {input_name: x})[0]
        out_b = sess_b.run(None, {input_name: x})[0]
        max_diffs.append(np.max(np.abs(out_a - out_b)))

    max_diffs = np.array(max_diffs)
    return {
        "num_samples": num_samples,
        "max_diff_mean": float(max_diffs.mean()),
        "max_diff_worst": float(max_diffs.max()),
        "allclose_1e-5": bool(np.all(max_diffs < 1e-5)),
        "allclose_1e-6": bool(np.all(max_diffs < 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two ONNX models")
    parser.add_argument("model_a", help="Path to first ONNX model")
    parser.add_argument("model_b", help="Path to second ONNX model")
    parser.add_argument("--obs-dim", type=int, default=None, help="Observation dimension (auto-detected if omitted)")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of random inference samples (default: 50)")
    args = parser.parse_args()

    print(f"=== Comparing {args.model_a} vs {args.model_b} ===\n")

    # 1. File hash
    hash_a = file_hash(args.model_a)
    hash_b = file_hash(args.model_b)
    if hash_a == hash_b:
        print("[PASS] Files are byte-identical (SHA-256 match)")
        return
    print("[INFO] Files differ at byte level\n")

    # 2. Structure
    model_a = onnx.load(args.model_a)
    model_b = onnx.load(args.model_b)

    struct_diffs = compare_structure(model_a, model_b)
    if struct_diffs:
        print(f"[FAIL] Structure differences ({len(struct_diffs)}):")
        for d in struct_diffs:
            print(f"  - {d}")
    else:
        print("[PASS] Structure identical (nodes, inputs, outputs, weights)")
    print()

    # 3. Metadata
    meta_diffs = compare_metadata(model_a, model_b)
    if meta_diffs:
        print(f"[INFO] Metadata differences ({len(meta_diffs)}):")
        for d in meta_diffs:
            print(f"  - {d}")
    else:
        print("[PASS] Metadata identical")
    print()

    # 4. Inference
    print(f"Running {args.num_samples} inference comparisons...")
    result = compare_inference(args.model_a, args.model_b, args.num_samples, args.obs_dim)
    print(f"  Max diff (mean): {result['max_diff_mean']:.2e}")
    print(f"  Max diff (worst): {result['max_diff_worst']:.2e}")
    print(f"  Allclose (1e-5): {result['allclose_1e-5']}")
    print(f"  Allclose (1e-6): {result['allclose_1e-6']}")

    if result["allclose_1e-6"]:
        print("\n[PASS] Models produce numerically identical outputs")
    elif result["allclose_1e-5"]:
        print("\n[PASS] Models produce equivalent outputs (within 1e-5)")
    else:
        print("\n[FAIL] Models produce different outputs")


if __name__ == "__main__":
    main()
