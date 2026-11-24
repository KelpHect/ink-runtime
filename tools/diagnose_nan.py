"""Diagnostic: find where NaN first appears in the Inkling2Loop forward pass.

Loads the model the same way as dump_golden.py, hooks every decoder layer's
input and output plus selected submodules, and runs one prefill forward.
"""

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "Inkling-2-Mini")
OFFLOAD_DIR = os.path.join(REPO, "data", "hf_offload")

PROMPT = "What is the capital of France? Answer briefly."


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop import Inkling2LoopForCausalLM
    from accelerate import init_empty_weights

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )["input_ids"]
    log(f"prompt: {ids.shape}")

    text_cfg = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()
    with init_empty_weights():
        skeleton = Inkling2LoopForCausalLM(text_cfg)
    device_map = {n: ("disk" if ".mlp.experts." in n else "cpu")
                  for n, _ in skeleton.named_parameters()}
    device_map.update({n: "cpu" for n, _ in skeleton.named_buffers()})
    del skeleton

    t0 = time.time()
    model = Inkling2LoopForCausalLM.from_pretrained(
        MODEL_DIR, config=text_cfg, dtype=torch.bfloat16,
        device_map=device_map, offload_folder=OFFLOAD_DIR,
        attn_implementation="eager",
    )
    model.eval()
    log(f"loaded in {time.time()-t0:.0f}s")

    tm = model.model
    visit = {"i": -1}
    records = []

    def fin(x):
        return bool(torch.isfinite(x).all())

    def stat(x):
        x = x.float()
        return f"absmax={x.abs().max().item():.4g} mean={x.mean().item():.4g}"

    # hook every decoder layer: pre-hook records input, fwd-hook records output
    for i, lyr in enumerate(tm.layers):
        def pre(module, args, i=i):
            visit["i"] += 1
            v = visit["i"]
            x = args[0]
            records.append(("in", v, i, fin(x), stat(x) if fin(x) else "NaN"))

        def fwd(module, args, output, i=i):
            o = output[0] if isinstance(output, tuple) else output
            v = visit["i"]
            records.append(("out", v, i, fin(o), stat(o) if fin(o) else "NaN"))

        lyr.register_forward_pre_hook(pre)
        lyr.register_forward_hook(fwd)

    # submodule detail for a few suspect layers (full layers + loop layers)
    subs = {}
    for i in (5, 6, 11, 12, 17, 18):
        lyr = tm.layers[i]
        for name in ("attention_input", "self_attn", "post_attention_convolution",
                     "mlp_input", "mlp", "post_mlp_convolution",
                     "q_message_gate", "mlp_message_gate"):
            mod = getattr(lyr, name, None)
            if mod is None:
                continue
            def mk(i=i, name=name):
                def f(module, args, output):
                    o = output[0] if isinstance(output, tuple) else output
                    if isinstance(o, torch.Tensor):
                        subs.setdefault((visit["i"], i, name), fin(o) and stat(o) or "NaN")
                return f
            mod.register_forward_hook(mk())

    log("running prefill...")
    out = model(input_ids=ids, use_cache=True)
    lg = out.logits[0, -1].float()
    log(f"logits finite: {fin(lg)}")

    first_nan = None
    for kind, v, i, ok, s in records:
        if not ok and first_nan is None:
            first_nan = v
        print(f"visit {v:2d} layer {i:2d} {kind:3s} finite={ok} {s}", flush=True)
    print("first NaN visit:", first_nan, flush=True)
    for (v, i, name), s in sorted(subs.items()):
        print(f"  visit {v:2d} layer {i:2d} {name:28s} {s}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
