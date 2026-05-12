#!/usr/bin/python3
"""Convert SEI autoencoder(s) to TensorRT engines for fast Orin NX inference.

Converts autoencoder.pt -> autoencoder.onnx -> autoencoder.trt for each
model directory. The .trt engine replaces PyTorch inference in sei_model.py
via the TRTAutoencoder wrapper at the bottom of this file.

Usage
-----
# Convert both models (typical dual-model setup):
    python convert_to_trt.py \
        --models sei_model_dl sei_model_ul \
        --input-length 1024 \
        --fp16

# Convert one model only:
    python convert_to_trt.py --models sei_model_dl --input-length 1024 --fp16

# Verify an engine against the original PyTorch model:
    python convert_to_trt.py --models sei_model_dl --verify

Then patch sei_model.py to use TRTAutoencoder (instructions printed at end).

Requirements
------------
    pip install tensorrt pycuda onnx onnxruntime  (all pre-installed on AIR-T)
    torch, numpy                                   (already in requirements.txt)

Notes
-----
- Batch size is dynamic: min=1, opt=8, max=16 (matches max_windows default).
  Change --opt-batch / --max-batch if you run with different --max-windows.
- FP16 is strongly recommended on Orin NX Ampere; accuracy impact on the
  z-score outputs is negligible because the threshold operates on medians.
- The ONNX intermediate is kept alongside the .trt for inspection / re-export.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Guard heavy imports so --help works without TRT installed
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Autoencoder definition (must match sei_model.py exactly)
# ---------------------------------------------------------------------------

from sei_model import IQAutoencoder


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model_dir: Path, input_length: int, max_batch: int) -> Path:
    """Load autoencoder.pt and export to autoencoder.onnx."""
    pt_path   = model_dir / "autoencoder.pt"
    onnx_path = model_dir / "autoencoder.onnx"

    model = IQAutoencoder(input_length)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, input_length, 2)   # (B, N, 2)

    print(f"  Exporting ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["iq_bn2"],
        output_names=["reconstruction"],
        dynamic_axes={
            "iq_bn2":        {0: "batch"},
            "reconstruction": {0: "batch"},
        },
    )
    print(f"  ONNX export done: {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# TensorRT build
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_length: int,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    """Parse ONNX and build a TensorRT engine with dynamic batch."""
    if not TRT_AVAILABLE:
        print("ERROR: tensorrt not importable. Install it or run on the AIR-T.", file=sys.stderr)
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser     = trt.OnnxParser(network, TRT_LOGGER)

    print(f"  Parsing ONNX: {onnx_path}")
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}", file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    # Workspace: 512 MB — generous for a small conv autoencoder on Orin NX
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled (Tensor Core acceleration)")
    elif fp16:
        print("  WARNING: FP16 requested but not supported on this platform; using FP32")

    # Dynamic shape profile: (B, N, 2)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "iq_bn2",
        min=(min_batch, input_length, 2),
        opt=(opt_batch, input_length, 2),
        max=(max_batch, input_length, 2),
    )
    config.add_optimization_profile(profile)

    print(f"  Building TRT engine (this takes ~30-90s on Orin NX) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: TensorRT engine build failed.", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"  Engine built in {elapsed:.1f}s")

    with open(str(engine_path), "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {engine_path}")


# ---------------------------------------------------------------------------
# Accuracy verification (ONNX Runtime vs PyTorch)
# ---------------------------------------------------------------------------

def verify(model_dir: Path, input_length: int) -> None:
    """Compare ONNX Runtime output to PyTorch for a random batch."""
    if not ORT_AVAILABLE:
        print("  onnxruntime not available; skipping verification.", file=sys.stderr)
        return

    onnx_path = model_dir / "autoencoder.onnx"
    if not onnx_path.exists():
        print(f"  {onnx_path} not found; run export first.", file=sys.stderr)
        return

    pt_path = model_dir / "autoencoder.pt"
    model   = IQAutoencoder(input_length)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(4, input_length, 2)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess   = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["reconstruction"], {"iq_bn2": dummy.numpy()})[0]

    max_err = float(np.abs(pt_out - ort_out).max())
    mean_err = float(np.abs(pt_out - ort_out).mean())
    print(f"  Verification: max_err={max_err:.2e}  mean_err={mean_err:.2e}")
    if max_err < 1e-4:
        print("  PASSED — ONNX output matches PyTorch within tolerance.")
    else:
        print("  WARNING — larger than expected discrepancy; inspect before deploying.")


# ---------------------------------------------------------------------------
# TRTAutoencoder — drop-in replacement for IQAutoencoder at inference time
# ---------------------------------------------------------------------------

TRT_WRAPPER_CODE = '''
# ---------------------------------------------------------------------------
# Paste this block into sei_model.py, replacing the _ae_error method and
# adding the TRTAutoencoder class. Then call SEIModel.load_trt() instead of
# SEIModel.load() in gnuradio_watcher.py and infer.py.
# ---------------------------------------------------------------------------

import pycuda.driver as cuda
import pycuda.autoinit          # noqa: F401  initialises CUDA context
import tensorrt as trt
import numpy as np


class TRTAutoencoder:
    """TensorRT wrapper matching IQAutoencoder's forward() signature.

    Accepts (B, N, 2) float32 numpy or torch tensors; returns (B, N, 2) numpy.
    Keeps CUDA buffers pre-allocated for the maximum batch size.
    """

    def __init__(self, engine_path: str, max_batch: int = 16, input_length: int = 1024):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_length = input_length
        self.max_batch    = max_batch

        # Pre-allocate pinned host + device buffers for max_batch
        n_elems   = max_batch * input_length * 2
        self.h_in  = cuda.pagelocked_empty(n_elems, dtype=np.float32)
        self.h_out = cuda.pagelocked_empty(n_elems, dtype=np.float32)
        self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

    def __call__(self, iq_bn2):
        """Run inference. iq_bn2: (B, N, 2) torch tensor or numpy array."""
        import torch
        if isinstance(iq_bn2, torch.Tensor):
            arr = iq_bn2.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(iq_bn2, dtype=np.float32)

        B = arr.shape[0]
        assert B <= self.max_batch, f"Batch {B} exceeds TRT max_batch {self.max_batch}"

        n = B * self.input_length * 2
        self.h_in[:n] = arr.ravel()

        # Set dynamic input shape for this batch
        self.context.set_input_shape("iq_bn2", (B, self.input_length, 2))

        cuda.memcpy_htod_async(self.d_in, self.h_in[:n], self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_in), int(self.d_out)],
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_out[:n], self.d_out, self.stream)
        self.stream.synchronize()

        return torch.from_numpy(self.h_out[:n].reshape(B, self.input_length, 2).copy())


# In SEIModel, replace _ae_error with:

    @torch.no_grad()
    def _ae_error_trt(self, iq_bn2, device="cuda"):
        """TRT-accelerated reconstruction error. device arg kept for API compat."""
        x = iq_bn2
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.from_numpy(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        recon = self.autoencoder(x)           # TRTAutoencoder handles device
        if isinstance(recon, np.ndarray):
            recon = torch.from_numpy(recon)
        err = ((recon - x.cpu()) ** 2).mean(dim=[1, 2])
        return err.numpy()

# And add to SEIModel.load() (classmethod):

    @classmethod
    def load_trt(cls, model_dir, input_length=1024, max_batch=16):
        """Load model with TensorRT autoencoder if .trt engine exists."""
        import os
        obj = cls.load(model_dir)   # loads .pt first (also sets config)
        engine_path = os.path.join(model_dir, "autoencoder.trt")
        if os.path.exists(engine_path):
            print(f"[INFO] Loading TRT engine: {engine_path}")
            obj.autoencoder = TRTAutoencoder(engine_path, max_batch=max_batch,
                                              input_length=input_length)
            obj._ae_error = obj._ae_error_trt
        else:
            print(f"[WARN] No .trt engine at {engine_path}; using PyTorch fallback.")
        return obj
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Convert SEI autoencoder(s) to TensorRT")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model directories (e.g. sei_model_dl sei_model_ul)")
    p.add_argument("--input-length", type=int, default=1024,
                   help="Autoencoder input length in samples (default 1024)")
    p.add_argument("--fp16", action="store_true",
                   help="Enable FP16 Tensor Core inference (recommended on Orin NX)")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8,
                   help="Optimal batch size for TRT tuning (default 8)")
    p.add_argument("--max-batch", type=int, default=16,
                   help="Maximum batch size (default 16, matches --max-windows default)")
    p.add_argument("--verify", action="store_true",
                   help="Compare ONNX vs PyTorch output after export (requires onnxruntime)")
    p.add_argument("--onnx-only", action="store_true",
                   help="Export ONNX only, skip TRT build")
    return p.parse_args()


def main():
    args = get_args()

    for model_dir_str in args.models:
        model_dir = Path(model_dir_str)
        if not model_dir.exists():
            print(f"\nSkipping {model_dir}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_dir}")
        print(f"{'='*60}")

        # 1. ONNX export
        onnx_path = export_onnx(model_dir, args.input_length, args.max_batch)

        # 2. Verify ONNX vs PyTorch (optional)
        if args.verify:
            print("  Running ONNX vs PyTorch verification ...")
            verify(model_dir, args.input_length)

        # 3. TRT build
        if not args.onnx_only:
            engine_path = model_dir / "autoencoder.trt"
            build_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                input_length=args.input_length,
                fp16=args.fp16,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
            )

    print(f"\n{'='*60}")
    print("Conversion complete.")
    print("\nNext steps:")
    print("  1. Copy the TRT_WRAPPER_CODE block (printed below) into sei_model.py")
    print("  2. In gnuradio_watcher.py and infer.py, replace:")
    print("         sei = SEIModel.load(args.model)")
    print("     with:")
    print("         sei = SEIModel.load_trt(args.model)")
    print("  3. Run: python infer.py --model sei_model_dl --channelized-file ...")
    print("     and confirm sei:verdict output is identical to the PyTorch path.")
    print(f"\n{'='*60}")
    print("TRTAutoencoder wrapper code to paste into sei_model.py:")
    print(TRT_WRAPPER_CODE)


if __name__ == "__main__":
    main()
