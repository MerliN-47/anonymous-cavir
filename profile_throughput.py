#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Systems Throughput, Latency & VRAM Profiling Suite
Double-Blind Compliant Modular Architecture
Profiles Token, Latent (ARM), and Output (Vincentization) injection across backbones,
batch sizes (B), context lengths (L), and retrieval budgets (k).
Identifies crossover points, latency scaling, and VRAM footprints.
"""

import os
import sys
import time
import json
import argparse
import traceback
from typing import Dict, List, Any, Optional

import torch
import numpy as np

from models.backbone_interface import get_backbone_adapter
from models.injection_heads import UnifiedRetrievalInjector


def save_json_safe(data: Any, primary_path: str):
    """
    Attempts to write profiling data to primary path or fallback scratch directories.
    """
    targets = [
        primary_path,
        "./scratch/systems_scaling_grid.json",
        "./results/systems_scaling_grid.json"
    ]
    written = []
    for tgt in targets:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(tgt)), exist_ok=True)
            with open(tgt, "w") as f:
                json.dump(data, f, indent=2)
            written.append(tgt)
        except Exception:
            pass

    if written:
        print(f"[profile_throughput] Successfully saved scaling grid to: {written[0]}")
    else:
        print(f"[profile_throughput] WARNING: Failed to write profiling grid to {primary_path}")


def profile_systems_throughput(
    backbones: List[str],
    batch_sizes: List[int],
    context_lengths: List[int],
    k_values: List[int],
    fusion_types: List[str],
    pred_len: int = 64,
    num_warmup: int = 1,
    num_runs: int = 3,
    out_file: str = "./scratch/systems_scaling_grid.json"
) -> Dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== [Anonymous] Systems Throughput, Latency & VRAM Profiling Sweep ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbones: {backbones}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Context Lengths: {context_lengths}")
    print(f"Retrieval Budgets (k): {k_values}")
    print(f"Fusion Types: {fusion_types}")
    print("=" * 70)

    results_grid = []
    summary_stats = {
        "total_configs": len(backbones) * len(batch_sizes) * len(context_lengths) * len(k_values) * len(fusion_types),
        "successful_runs": 0,
        "oom_runs": 0,
        "crossover_points": []
    }

    pretrained_path = "./checkpoints/base"

    for backbone_name in backbones:
        for L in context_lengths:
            print(f"\n>>> Loading {backbone_name} (L={L}, H={pred_len}) on {device}...")
            try:
                backbone = get_backbone_adapter(
                    backbone_name=backbone_name,
                    pretrained_model_path=pretrained_path,
                    context_length=L,
                    prediction_length=pred_len,
                    device=device
                )
                backbone.eval()
            except Exception as e:
                print(f"ERROR: Failed to load backbone {backbone_name}: {e}")
                continue

            for fusion in fusion_types:
                try:
                    injector = UnifiedRetrievalInjector(
                        backbone=backbone,
                        injection_point=fusion,
                        d_model=backbone.d_model,
                        prediction_length=pred_len
                    ).to(device)
                    injector.eval()
                except Exception as e:
                    print(f"ERROR: Failed to initialize injector {fusion}: {e}")
                    continue

                for k in k_values:
                    for B in batch_sizes:
                        config_desc = f"{backbone_name} | {fusion} | L={L} | k={k} | B={B}"
                        entry = {
                            "backbone": backbone_name,
                            "fusion": fusion,
                            "context_length": L,
                            "k": k,
                            "batch_size": B,
                            "prediction_length": pred_len,
                            "status": "SUCCESS",
                            "oom": False,
                            "mean_latency_ms": None,
                            "std_latency_ms": None,
                            "throughput_samples_sec": None,
                            "peak_vram_mb": None,
                            "active_vram_mb": None,
                            "error": None
                        }

                        # Construct dummy input tensors
                        x_q = torch.randn(B, L, device=device)
                        ret_past = torch.randn(B, k, L, device=device)
                        ret_future = torch.randn(B, k, pred_len, device=device)

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(device)

                        try:
                            # Warmup passes
                            with torch.no_grad():
                                for _ in range(num_warmup):
                                    _ = injector(x_q, ret_past, ret_future)

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                            # Timed benchmark iterations
                            durations = []
                            with torch.no_grad():
                                for _ in range(num_runs):
                                    t0 = time.perf_counter()
                                    _ = injector(x_q, ret_past, ret_future)
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    t1 = time.perf_counter()
                                    durations.append((t1 - t0) * 1000.0)  # ms

                            mean_lat = float(np.mean(durations))
                            std_lat = float(np.std(durations))
                            throughput = float((B / (mean_lat / 1000.0))) if mean_lat > 0 else 0.0

                            entry["mean_latency_ms"] = round(mean_lat, 2)
                            entry["std_latency_ms"] = round(std_lat, 2)
                            entry["throughput_samples_sec"] = round(throughput, 2)

                            if torch.cuda.is_available():
                                peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                entry["peak_vram_mb"] = round(peak_vram, 2)

                            summary_stats["successful_runs"] += 1
                            vram_str = f" | Peak VRAM: {entry['peak_vram_mb']:.1f} MB" if entry['peak_vram_mb'] else ""
                            print(f"  [OK] {config_desc} => {mean_lat:.2f} ms ({throughput:.1f} seq/s){vram_str}")

                        except torch.cuda.OutOfMemoryError as e:
                            print(f"  [OOM] {config_desc} => CUDA Out of Memory!")
                            entry["status"] = "OOM"
                            entry["oom"] = True
                            entry["error"] = str(e)
                            summary_stats["oom_runs"] += 1
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"  [ERROR] {config_desc} => {e}")
                            entry["status"] = "ERROR"
                            entry["error"] = str(e)

                        results_grid.append(entry)

    full_output = {
        "metadata": {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": summary_stats,
        "grid": results_grid
    }

    save_json_safe(full_output, out_file)
    print("\n" + "=" * 70)
    print(f"Profiling Sweep Complete! Runs: {summary_stats['successful_runs']} OK, {summary_stats['oom_runs']} OOM")
    print("=" * 70)
    return full_output


def main():
    parser = argparse.ArgumentParser(description="[Anonymous] TS-RAG / CAVIR Systems Profiling Runner")
    parser.add_argument("--backbones", nargs="+", default=["chronos-bolt", "moirai-2.0"], help="Backbone models")
    parser.add_argument("--fusions", nargs="+", default=["latent", "token", "output"], help="Fusion strategies")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32, 128], help="Batch sizes")
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[512, 1024, 2048], help="Context lengths")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 10, 25, 50], help="Retrieval budgets k")
    parser.add_argument("--pred_len", type=int, default=64, help="Prediction horizon H")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark iterations")
    parser.add_argument("--out_file", type=str, default="./scratch/systems_scaling_grid.json", help="Output JSON path")
    args = parser.parse_args()

    profile_systems_throughput(
        backbones=args.backbones,
        batch_sizes=args.batch_sizes,
        context_lengths=args.context_lengths,
        k_values=args.k_values,
        fusion_types=args.fusions,
        pred_len=args.pred_len,
        num_warmup=args.warmup,
        num_runs=args.runs,
        out_file=args.out_file
    )


if __name__ == "__main__":
    main()
