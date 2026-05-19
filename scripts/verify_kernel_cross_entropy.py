"""Standalone verification for the NKI cross_entropy wrapper.

Compares forward loss + backward grad_logits between PyTorch's
F.cross_entropy and our scripts.kernels.nki_cross_entropy on identical
inputs. Runs entirely on the requested device — defaults to neuron, falls
back to cpu (where the wrapper itself isn't usable, but the test stays
useful as a quick PyTorch baseline).

Usage on trn2:
    NEURON_RT_NUM_CORES=1 NEURON_RT_VISIBLE_CORES=0 \\
    NEURON_CC_FLAGS="--model-type=transformer" \\
    /opt/torch-neuronx/.venv/bin/python scripts/verify_kernel_cross_entropy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEVICE = os.environ.get("DEEPSEEK_DEVICE", "neuron").lower()
if DEVICE == "neuron":
    import torch_neuronx  # noqa: F401  registers 'neuron' device

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.kernels import nki_cross_entropy  # noqa: E402


# Match training config: B*S = 1*512, but slice off the last position to
# mimic the modeling.py "shift labels" path => B*(S-1) = 511 positions.
SEED = 42
N = 511                    # num positions (B*(S-1) for our debug config)
V = 129280                 # vocab_size from main_100m.yaml
ATOL_F, RTOL_F = 1e-4, 1e-4   # forward (scalar, low dynamic range)
ATOL_G, RTOL_G = 1e-3, 1e-3   # grad_logits (large tensor, more accumulated noise)


def build_inputs():
    g = torch.Generator().manual_seed(SEED)
    logits_cpu = torch.randn(N, V, generator=g, dtype=torch.float32)
    targets_cpu = torch.randint(0, V, (N,), generator=g, dtype=torch.long)
    return logits_cpu, targets_cpu


def run_pytorch(logits_cpu, targets_cpu):
    """Reference path on CPU."""
    logits = logits_cpu.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(logits, targets_cpu, reduction="mean")
    loss.backward()
    return loss.detach(), logits.grad.detach()


def run_nki(logits_cpu, targets_cpu):
    """NKI path on the target device."""
    logits = logits_cpu.clone().to(DEVICE).detach().requires_grad_(True)
    targets = targets_cpu.to(DEVICE)
    loss = nki_cross_entropy(logits, targets)
    loss.backward()
    return loss.detach().to("cpu"), logits.grad.detach().to("cpu")


def report(name, ref, got, atol, rtol):
    ref = ref.float()
    got = got.float()
    diff = (ref - got).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<20s}  shape={tuple(ref.shape)!s:<14s}"
          f"  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  max_rel={max_rel:.3e}")
    return ok


def main():
    print(f"[verify-ce] device={DEVICE}  N={N}  V={V}  seed={SEED}")
    print(f"[verify-ce] tolerances: forward atol={ATOL_F} rtol={RTOL_F},"
          f"  grad atol={ATOL_G} rtol={RTOL_G}")

    logits_cpu, targets_cpu = build_inputs()

    print("[verify-ce] running PyTorch reference on CPU...")
    ref_loss, ref_grad = run_pytorch(logits_cpu, targets_cpu)
    print(f"  ref loss = {ref_loss.item():.6f}")

    print(f"[verify-ce] running NKI path on {DEVICE}...")
    got_loss, got_grad = run_nki(logits_cpu, targets_cpu)
    print(f"  nki loss = {got_loss.item():.6f}")

    n_pass = n_fail = 0
    for name, ref, got, atol, rtol in [
        ("loss",        ref_loss, got_loss, ATOL_F, RTOL_F),
        ("grad_logits", ref_grad, got_grad, ATOL_G, RTOL_G),
    ]:
        if report(name, ref, got, atol, rtol):
            n_pass += 1
        else:
            n_fail += 1

    print(f"[verify-ce] summary: pass={n_pass} fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
