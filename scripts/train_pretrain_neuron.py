"""Pretrain DeepSeek-V4 on FineWeb-Edu, targeting AWS Neuron (Trainium2).

This is a **Neuron-first** copy of [scripts/train_pretrain.py](train_pretrain.py).
It deliberately bypasses HuggingFace Trainer / TRL SFTTrainer because their
device dispatch + AMP paths assume CUDA/CPU and do not know about
`torch.device("neuron")`. A small hand-written training loop matches the
shape of [DeepSeek-onboarding/src/run_inference.py](../../DeepSeek-onboarding/src/run_inference.py)
(eager torch_neuronx, no torch.compile, no FSDP/TP) and is the smallest
thing that exercises forward + backward on a single Logical NeuronCore.

Usage on trn2.48xlarge (single LNC, no sharding):
    NEURON_RT_NUM_CORES=1 \\
    NEURON_RT_VISIBLE_CORES=0 \\
    NEURON_CC_FLAGS="--model-type=transformer" \\
    /opt/torch-neuronx/.venv/bin/python scripts/train_pretrain_neuron.py \\
        --config configs/main_100m.yaml --debug

Local CPU smoke check (no Neuron runtime needed):
    DEEPSEEK_DEVICE=cpu python scripts/train_pretrain_neuron.py \\
        --config configs/main_100m.yaml --debug
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import yaml

# Resolve repo root so the modeling files import cleanly regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Pick the device BEFORE importing torch_neuronx — only import it when we
# actually want the 'neuron' device, otherwise CPU smoke tests work on
# machines without the Neuron SDK.
DEVICE = os.environ.get("DEEPSEEK_DEVICE", "neuron").lower()
if DEVICE == "neuron":
    import torch_neuronx  # noqa: F401  registers the 'neuron' device

import torch  # noqa: E402  (must come after the optional torch_neuronx import)
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from modeling_deepseek_v4 import DeepseekV4ForCausalLM  # noqa: E402


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Build a DeepSeek-V4 model from scratch (random init).

    Construction stays on CPU — the model's `post_init()` calls hit
    `tensor.normal_()` etc., which are CPU_FALLBACK ops on Neuron and would
    serialize back to CPU anyway. We move to the target device below.
    """
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    return model


def build_tokenizer(tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = str(REPO_ROOT / "tokenizer")
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class PackedTextDataset(IterableDataset):
    """Tokenize a streamed text corpus and emit fixed-length token blocks.

    Replaces SFTTrainer's `packing=True` data path. We tokenize each example,
    drop EOS-terminated sequences into a rolling buffer, and yield
    contiguous `seq_len`-sized tensors. This is a deliberate simplification:
    no document boundary handling, no attention-mask manipulation — the
    point is to feed the trainer with valid token ids deterministically.
    """

    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buf: list[int] = []
        eos = self.tokenizer.eos_token_id
        for row in self.hf_dataset:
            ids = self.tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            if eos is not None:
                buf.append(eos)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def cosine_lr(step, max_steps, base_lr, warmup_ratio):
    """Linear warmup → cosine decay to 10% of base_lr.

    Mirrors HF Trainer's `lr_scheduler_type=cosine` + `warmup_ratio` so the
    Neuron loss curve is comparable to the CPU/CUDA Trainer run.
    """
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main_100m.yaml")
    parser.add_argument("--debug", action="store_true", help="Use tiny streamed subset")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override training.max_steps from the yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # ------------------------------------------------------------------
    # Model / tokenizer
    # ------------------------------------------------------------------
    torch.manual_seed(train_cfg.get("seed", 42))
    model = build_model(model_cfg)
    tokenizer = build_tokenizer()

    # Move to target device. randn/normal_/randint are CPU_FALLBACK on
    # Neuron, so anything that creates a fresh tensor inline must either
    # stay on CPU or use a precomputed buffer — we already followed that
    # rule for init (post_init runs at construction time, on CPU).
    print(f"[neuron-train] target device = {DEVICE}")
    model = model.to(DEVICE)
    model.train()

    # See README "Known Issues": Hyper-Connections overflow bf16 at 100M
    # scale. The CPU/CUDA path also runs in fp32 here (the Trainer gates
    # bf16 on CUDA availability), so for parity we stay in fp32 on Neuron.

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.debug:
        # Tiny streamed slice — same shape as the CPU/CUDA --debug path.
        raw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        raw = raw.take(200)
        max_steps = args.max_steps or train_cfg.get("max_steps", 10)
        logging_steps = 1
    else:
        raw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        max_steps = args.max_steps or train_cfg.get("max_steps", 20000)
        logging_steps = train_cfg.get("logging_steps", 10)

    seq_len = train_cfg.get("max_seq_length", 2048)
    bsz = train_cfg.get("per_device_train_batch_size", 1)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)

    packed = PackedTextDataset(raw, tokenizer, seq_len)
    loader = DataLoader(
        packed,
        batch_size=bsz,
        # num_workers=0 to keep all DataLoader work in-process. Neuron
        # runtime has its own per-process state, and DataLoader workers
        # would each try to attach — keep it boring on the data side.
        num_workers=0,
    )
    batch_iter = iter(loader)

    # ------------------------------------------------------------------
    # Optimizer + LR schedule
    # ------------------------------------------------------------------
    base_lr = train_cfg.get("learning_rate", 6e-4)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(train_cfg.get("adam_beta1", 0.9), train_cfg.get("adam_beta2", 0.95)),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        # No `fused=True`: the CUDA-fused AdamW kernel is unavailable on Neuron.
    )
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.03)

    # ------------------------------------------------------------------
    # Output dir / banner
    # ------------------------------------------------------------------
    output_dir = args.output_dir or str(
        REPO_ROOT / f"checkpoints/pretrain_neuron_{Path(args.config).stem}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pretraining DeepSeek-V4 (Neuron path)")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {bsz} x {grad_accum} GA")
    print(f"Seq length: {seq_len}")
    print(f"LR: {base_lr}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    optim.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        lr = cosine_lr(step - 1, max_steps, base_lr, warmup_ratio)
        for pg in optim.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
            # Build inputs on CPU then move — randint-style ops are
            # CPU_FALLBACK on Neuron (DeepSeek-onboarding doc, "randn /
            # normal_ are CPU_FALLBACK"). Tokenization above already
            # happened on CPU so this is just a host→device copy.
            input_ids = batch.to(DEVICE)
            labels = input_ids
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.detach().to("cpu").float().item()

        # Gradient clip + step. clip_grad_norm_ on Neuron may CPU_FALLBACK
        # via the host-side reduction; fine for smoke-testing.
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % logging_steps == 0 or step == 1:
            print(f"step {step:>5}/{max_steps}  loss={accum_loss:.4f}  lr={lr:.3e}")

    # ------------------------------------------------------------------
    # Save final state — back to CPU so safetensors can serialize.
    # ------------------------------------------------------------------
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    cpu_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(cpu_state, os.path.join(final_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_dir)
    model.config.save_pretrained(final_dir)
    print(f"\nFinal model saved to {final_dir}")


if __name__ == "__main__":
    main()
