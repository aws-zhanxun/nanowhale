"""parity_check.py — minimal CPU vs Neuron parity test.

Two phases:
  1. dump   : run a single fp32 training step on the chosen device and save
              {init_state_dict, input_ids, loss, logits, grads,
               params_after_step, [router_topk_indices]}.
  2. compare: load the cpu and neuron dumps, print PASS/FAIL per tensor.

Workflow:
    # CPU baseline (also produces the shared init weights + input_ids)
    python scripts/parity_check.py --mode dump \\
        --device cpu --out /tmp/parity_cpu.pt --dump-router

    # Neuron run (loads init + inputs from the CPU dump, ensuring identity)
    NEURON_RT_NUM_CORES=1 NEURON_RT_VISIBLE_CORES=0 \\
    NEURON_CC_FLAGS="--model-type=transformer" \\
    /opt/torch-neuronx/.venv/bin/python scripts/parity_check.py --mode dump \\
        --device neuron --init /tmp/parity_cpu.pt \\
        --out /tmp/parity_neuron.pt --dump-router

    # Compare (router indices auto-compared if both dumps contain them)
    python scripts/parity_check.py --mode compare \\
        --cpu /tmp/parity_cpu.pt --neuron /tmp/parity_neuron.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Register the 'neuron' device BEFORE `import torch` if --device neuron was
# requested. Mirrors the gating in scripts/train_pretrain_neuron.py.
if "--device" in sys.argv:
    _i = sys.argv.index("--device")
    if _i + 1 < len(sys.argv) and sys.argv[_i + 1] == "neuron":
        import torch_neuronx  # noqa: F401  registers 'neuron'

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from modeling_deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Gate  # noqa: E402


# Cross-run determinism is not needed for parity (neuron loads cpu's dump),
# but keeping a fixed seed means rerunning the cpu baseline produces the same
# loss number, which makes ad-hoc debugging less confusing.
SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_input_ids(vocab_size: int, seq_len: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.randint(0, vocab_size, (1, seq_len), generator=g, dtype=torch.long)


def attach_router_hooks(model):
    """Stash top-k indices from every DeepseekV4Gate.forward call.

    Gates are keyed by encounter order (0, 1, ...) during forward. The
    module graph is fixed across devices, so the same key refers to the
    same gate on cpu and neuron.
    """
    captured: dict[int, torch.Tensor] = {}
    counter = {"n": 0}

    def hook(module, args, output):
        # output of DeepseekV4Gate.forward is (weights, indices)
        captured[counter["n"]] = output[1].detach().to("cpu").clone()
        counter["n"] += 1

    handles = []
    for m in model.modules():
        if isinstance(m, DeepseekV4Gate):
            handles.append(m.register_forward_hook(hook))
    return captured, handles


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def cmd_dump(args):
    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    vocab_size = model_cfg["vocab_size"]
    seq_len = train_cfg["max_seq_length"]

    # init weights / input_ids: load from prior dump if given, else fresh
    if args.init:
        ref = torch.load(args.init, map_location="cpu", weights_only=False)
        init_state_dict = ref["init_state_dict"]
        input_ids = ref["input_ids"]
        seq_len = input_ids.shape[1]
        print(f"[parity-dump] loaded init + input_ids from {args.init}")
    else:
        init_state_dict = None
        input_ids = build_input_ids(vocab_size, seq_len)
        print(f"[parity-dump] freshly initialized (seq_len={seq_len})")

    # Build on CPU, force fp32. README warns bf16 NaNs at this scale; fp32
    # also removes a source of non-bitwise noise from the comparison.
    torch.manual_seed(SEED)
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config).float()
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=True)

    # Capture init AFTER potential load, so neuron and cpu dumps embed the
    # same init bytes — useful for an extra sanity diff.
    init_to_save = {
        k: v.detach().to("cpu").float().clone() for k, v in model.state_dict().items()
    }

    print(f"[parity-dump] device={args.device}  seq_len={seq_len}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    model = model.to(args.device)
    model.train()

    captured_router = None
    handles = []
    if args.dump_router:
        captured_router, handles = attach_router_hooks(model)

    # forward + backward
    input_ids_dev = input_ids.to(args.device)
    outputs = model(input_ids=input_ids_dev, labels=input_ids_dev)
    loss = outputs.loss
    logits = outputs.logits
    loss.backward()

    grads = {
        name: p.grad.detach().to("cpu").float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    # optimizer step
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 6e-4),
        betas=(train_cfg.get("adam_beta1", 0.9), train_cfg.get("adam_beta2", 0.95)),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )
    optim.step()

    params_after = {
        name: p.detach().to("cpu").float().clone()
        for name, p in model.named_parameters()
    }

    for h in handles:
        h.remove()

    payload = {
        "device": args.device,
        "config_path": str(args.config),
        "input_ids": input_ids,
        "init_state_dict": init_to_save,
        "loss": loss.detach().to("cpu").float().clone(),
        "logits": logits.detach().to("cpu").float().clone(),
        "grads": grads,
        "params_after_step": params_after,
    }
    if captured_router is not None:
        payload["router_topk_indices"] = captured_router

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[parity-dump] wrote {args.out}  loss={payload['loss'].item():.6f}  "
          f"keys={list(payload.keys())}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (a.abs() + 1e-8)).max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    return ok, max_abs, mean_abs, max_rel


def _row(name, shape, ok, max_abs, mean_abs, max_rel):
    tag = "PASS" if ok else "FAIL"
    return (f"  [{tag}] {name:<60s} shape={str(tuple(shape)):<22s}"
            f" max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} max_rel={max_rel:.3e}")


def cmd_compare(args):
    cpu = torch.load(args.cpu, map_location="cpu", weights_only=False)
    neu = torch.load(args.neuron, map_location="cpu", weights_only=False)

    # Tighter for scalars/post-step params (just AdamW arithmetic on top of
    # grads). Looser for logits/grads (bigger dynamic range).
    ATOL_T, RTOL_T = 1e-4, 1e-4
    ATOL_L, RTOL_L = 1e-3, 1e-3

    print("=" * 88)
    print(f"compare  cpu={args.cpu}  neuron={args.neuron}")
    print("=" * 88)

    n_pass = n_fail = 0

    print("[loss]")
    ok, ma, mn, mr = _stats(cpu["loss"], neu["loss"], ATOL_T, RTOL_T)
    print(_row("loss", cpu["loss"].shape, ok, ma, mn, mr))
    n_pass += int(ok); n_fail += int(not ok)

    print("[logits]")
    ok, ma, mn, mr = _stats(cpu["logits"], neu["logits"], ATOL_L, RTOL_L)
    print(_row("logits", cpu["logits"].shape, ok, ma, mn, mr))
    n_pass += int(ok); n_fail += int(not ok)

    print("[grads]")
    for name in cpu["grads"]:
        if name not in neu["grads"]:
            print(f"  [MISS] {name} not in neuron dump")
            n_fail += 1
            continue
        ok, ma, mn, mr = _stats(cpu["grads"][name], neu["grads"][name],
                                ATOL_L, RTOL_L)
        print(_row(name, cpu["grads"][name].shape, ok, ma, mn, mr))
        n_pass += int(ok); n_fail += int(not ok)

    print("[params_after_step]")
    for name in cpu["params_after_step"]:
        if name not in neu["params_after_step"]:
            print(f"  [MISS] {name} not in neuron dump")
            n_fail += 1
            continue
        ok, ma, mn, mr = _stats(cpu["params_after_step"][name],
                                neu["params_after_step"][name],
                                ATOL_T, RTOL_T)
        print(_row(name, cpu["params_after_step"][name].shape, ok, ma, mn, mr))
        n_pass += int(ok); n_fail += int(not ok)

    cpu_has = "router_topk_indices" in cpu
    neu_has = "router_topk_indices" in neu
    if cpu_has != neu_has:
        # User mistake: one dump produced with --dump-router, the other not.
        # Abort instead of silently skipping — the asymmetry usually means the
        # user intended to compare it, and a half-result would mislead.
        which = "cpu" if cpu_has else "neuron"
        missing = "neuron" if cpu_has else "cpu"
        print(f"\n[ERROR] router_topk_indices present in {which} dump but missing "
              f"from {missing} dump. Re-dump with --dump-router on both sides "
              f"(or neither) and retry.")
        sys.exit(2)
    if cpu_has and neu_has:
        print("[router_topk_indices]")
        for k in sorted(cpu["router_topk_indices"]):
            a = cpu["router_topk_indices"][k]
            b = neu["router_topk_indices"].get(k)
            if b is None:
                print(f"  [MISS] gate#{k} not in neuron dump")
                n_fail += 1
                continue
            ok = a.shape == b.shape and torch.equal(a, b)
            tag = "PASS" if ok else "FAIL"
            mismatch = (a != b).sum().item() if a.shape == b.shape else -1
            print(f"  [{tag}] gate#{k:<3d} shape={tuple(a.shape)}  "
                  f"mismatched_entries={mismatch}")
            n_pass += int(ok); n_fail += int(not ok)
    else:
        print("[WARN] router_topk_indices skipped — neither dump contains it. "
              "Re-run both dumps with --dump-router if you want to diagnose "
              "MoE routing divergence.")

    print("-" * 88)
    print(f"summary  pass={n_pass}  fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dump", "compare"], required=True)

    # dump
    p.add_argument("--device", choices=["cpu", "neuron"])
    p.add_argument("--config", default="configs/main_100m.yaml")
    p.add_argument("--out", help="output path (dump mode)")
    p.add_argument("--init", default=None,
                   help="path to a prior dump; load init_state_dict + input_ids from it")

    # compare
    p.add_argument("--cpu", help="path to cpu dump (compare mode)")
    p.add_argument("--neuron", help="path to neuron dump (compare mode)")

    # dump-only
    p.add_argument("--dump-router", action="store_true",
                   help="(dump mode) include MoE gate top-k indices in the dump. "
                        "Compare mode auto-detects them in the dump files.")

    args = p.parse_args()

    if args.mode == "dump":
        if not args.device or not args.out:
            p.error("--device and --out are required for --mode dump")
        cmd_dump(args)
    else:
        if not args.cpu or not args.neuron:
            p.error("--cpu and --neuron are required for --mode compare")
        cmd_compare(args)


if __name__ == "__main__":
    main()
