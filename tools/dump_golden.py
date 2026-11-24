"""Dump golden reference outputs for the ink-runtime correctness harness.

Runs Inkling-2-Mini (arch `inkling2_loop`, HF `thinkingmachines/Inkling-2-Mini`)
on CPU through HF transformers (git main) and writes to testdata/golden/:

  meta.json                      prompt ids, generated ids/text, dtype, seed,
                                 capture documentation, versions
  logits_step{0..7}.npy          fp32 [200058] post-softcap logits of the last
                                 position for each forward step (step 0 = prefill,
                                 steps 1..7 = one-token decode)
  hidden_visit{5,23,41}_step*.npy  fp32 [T,2304] decoder-layer INPUT (the attn-res
                                 "partial") captured by a forward pre-hook at the
                                 virtual-visit taps used by the MTP/DSpark head
                                 (research/07 section 9): visit 5 = input of
                                 layer 5 (prelude), visit 23 = input of layer 11
                                 on pass 1, visit 41 = input of layer 17 on
                                 pass 2. Step 0 holds all T=prompt positions,
                                 later steps hold T=1.
  hidden_final_step*.npy         fp32 [T,2304] output of `model.norm` (the final
                                 RMSNorm, tap id 48), same T convention.

Choices (documented, not silently assumed):
  * Model class: Inkling2LoopForCausalLM (model_type inkling2_loop_text). The
    checkpoint dir holds a composite `inkling2_loop` config; the text config is
    extracted via `AutoConfig.get_text_config()`. This loads ONLY the language
    model (the `model.llm.*` checkpoint tensors); `model.visual.*` source keys
    are unused. Vision is out of scope for this harness.
  * dtype: torch.bfloat16 - the native checkpoint dtype and the reference
    semantics (fp32 is used internally by the modeling code where the spec
    requires it: router, attn_res mixes, convs, softmax, unembed output).
  * attn_implementation="eager" - the semantic ground truth per research/07
    (fp32 softmax), chosen over sdpa for spec fidelity.
  * Checkpoint experts are NVFP4/MXFP8; transformers dequantizes them to bf16
    at load (Inkling2LoopDequantizeExperts). Dequantized experts total ~49 GB,
    exceeding the 48 GB RAM, so `model.layers.*.mlp.experts.*` parameters are
    mapped to "disk" (accelerate disk offload -> data/hf_offload/, ~49 GB,
    safe to delete after the run). They are paged in per forward call; the
    page cache makes repeated tokens fast. All other weights stay resident.
  * Greedy decoding is done MANUALLY (argmax of out.logits[:, -1], no
    generation_config sampling) for full control of per-step captures.
    seed=0 is fixed anyway.
  * Prompt: chat template with add_generation_prompt=True. NOTE: this
    template's `{% generation %}` block wraps whole assistant turns, so the
    rendered prefix ends after the user message's `<|end_message|>` and the
    first generated token is expected to be `<|message_model|>` (200001).
    Exact ids are recorded in meta.json.

Runs unattended: prints progress to stdout (use `python -u`).
"""

import faulthandler
import json
import os
import sys
import time

import numpy as np
import torch

faulthandler.enable()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "Inkling-2-Mini")
OUT_DIR = os.path.join(REPO, "testdata", "golden")
OFFLOAD_DIR = os.path.join(REPO, "data", "hf_offload")

NEW_TOKENS = 8
TAP_VISITS = (5, 23, 41)
USER_PROMPT = "What is the capital of France? Answer briefly."
SEED = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop import Inkling2LoopForCausalLM

    log(f"transformers {transformers.__version__}, torch {torch.__version__}")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = [{"role": "user", "content": USER_PROMPT}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    prompt_ids = enc["input_ids"]
    prompt_text = tok.decode(prompt_ids)
    log(f"prompt tokens: {len(prompt_ids)} -> {prompt_text!r}")

    # ------------------------------------------------------------------
    # Config + device map (experts -> disk, everything else cpu)
    # ------------------------------------------------------------------
    full_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    text_cfg = full_cfg.get_text_config()
    log(f"text config: {type(text_cfg).__name__}, layers={text_cfg.num_hidden_layers}, "
        f"visits={text_cfg.total_num_hidden_layers}")

    # device_map notes:
    #  * Parameter-level keys do NOT work: accelerate's dispatch_model matches
    #    keys to *submodules* only, silently leaving offloaded params on meta
    #    (F.linear on a meta weight then returns garbage without erroring).
    #  * A root '' -> 'cpu' entry also does NOT work: it attaches a
    #    place_submodules hook at the root that eagerly moves the offloaded
    #    (meta) expert params to cpu and crashes.
    #  * 'auto' emits leaf-module-level keys only (decoder layers are
    #    _no_split_modules, so each ~2 GB layer maps atomically); modules that
    #    do not fit max_memory land on 'disk' and page in per forward call.
    device_map = "auto"
    max_memory = {"cpu": "6GiB"}
    log("device_map: auto, max_memory cpu=6GiB (whole decoder layers -> disk)")

    # ------------------------------------------------------------------
    # Load (experts dequantize bf16 -> written to OFFLOAD_DIR during load)
    # ------------------------------------------------------------------
    t0 = time.time()
    model = Inkling2LoopForCausalLM.from_pretrained(
        MODEL_DIR,
        config=text_cfg,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=OFFLOAD_DIR,
        attn_implementation="eager",
    )
    model.eval()
    log(f"model loaded in {time.time() - t0:.1f}s; hf_device_map disk params: "
        f"{sum(v == 'disk' for v in model.hf_device_map.values())}")

    text_model = model.model  # Inkling2LoopTextModel
    layers = text_model.layers
    assert len(layers) == text_cfg.num_hidden_layers

    # ------------------------------------------------------------------
    # Tap hooks: virtual-visit counter + layer-input capture + final norm
    # ------------------------------------------------------------------
    state = {"visit": -1, "taps": {}, "final": None}
    tap_set = set(TAP_VISITS)

    def make_pre_hook():
        def pre_hook(module, args):
            state["visit"] += 1
            v = state["visit"]
            if v in tap_set:
                # args[0] = hidden_states = the attn-res partial = layer input
                state["taps"][v] = args[0][0].detach().to(torch.float32).cpu().clone()
        return pre_hook

    for lyr in layers:
        lyr.register_forward_pre_hook(make_pre_hook())

    def norm_hook(module, args, output):
        state["final"] = output[0].detach().to(torch.float32).cpu().clone()

    text_model.norm.register_forward_hook(norm_hook)

    # ------------------------------------------------------------------
    # Manual greedy decode
    # ------------------------------------------------------------------
    meta_files = []
    generated_ids = []
    step_summaries = []
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    cache = None
    step_times = []

    for step in range(NEW_TOKENS):
        state["visit"] = -1
        state["taps"] = {}
        state["final"] = None
        t0 = time.time()
        out = model(input_ids=ids, use_cache=True, cache=cache)
        dt = time.time() - t0
        step_times.append(dt)
        assert state["visit"] == text_cfg.total_num_hidden_layers - 1, (
            f"visit counter ended at {state['visit']}, expected "
            f"{text_cfg.total_num_hidden_layers - 1}"
        )
        logits = out.logits[0, -1].to(torch.float32).cpu()  # [200058] fp32
        assert logits.shape[-1] == text_cfg.unpadded_vocab_size
        assert torch.isfinite(logits).all(), (
            f"non-finite logits at step {step} - model weights did not load "
            f"correctly (check device_map hooks / offloaded params)"
        )
        cache = out.cache
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)

        fn = f"logits_step{step}.npy"
        np.save(os.path.join(OUT_DIR, fn), logits.numpy())
        meta_files.append(fn)
        for v in TAP_VISITS:
            tap = state["taps"][v].numpy()
            assert tap.ndim == 2 and tap.shape[-1] == text_cfg.hidden_size
            fn = f"hidden_visit{v}_step{step}.npy"
            np.save(os.path.join(OUT_DIR, fn), tap)
            meta_files.append(fn)
        fn = f"hidden_final_step{step}.npy"
        np.save(os.path.join(OUT_DIR, fn), state["final"].numpy())
        meta_files.append(fn)

        top5 = torch.topk(logits, 5)
        step_summaries.append({
            "step": step,
            "input_len": int(ids.shape[-1]),
            "argmax": next_id,
            "top5_ids": top5.indices.tolist(),
            "top5_values": top5.values.tolist(),
            "seconds": round(dt, 2),
        })
        log(f"step {step}: in_len={ids.shape[-1]} -> id {next_id} "
            f"({tok.decode([next_id])!r}) {dt:.1f}s")

        ids = torch.tensor([[next_id]], dtype=torch.long)
        if next_id == 200006:  # <|content_model_end_sampling|>
            log("EOS generated; stopping early")
            break

    generated_text = tok.decode(generated_ids)
    log(f"generated: {generated_text!r}")

    meta = {
        "model_dir": "Inkling-2-Mini",
        "hf_model_id": "thinkingmachines/Inkling-2-Mini",
        "model_class": "Inkling2LoopForCausalLM",
        "model_type": text_cfg.model_type,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "seed": SEED,
        "decoding": "manual greedy argmax of post-softcap logits (no sampling)",
        "prompt": {
            "user_message": USER_PROMPT,
            "chat_template": "chat_template.jinja via apply_chat_template("
                             "add_generation_prompt=True); rendered prefix ends "
                             "after user <|end_message|> (the {% generation %} "
                             "block wraps whole assistant turns, so the model "
                             "emits <|message_model|> itself)",
            "rendered_text": prompt_text,
            "input_ids": prompt_ids,
        },
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "eos_token_id": 200006,
        "steps": step_summaries,
        "hidden_tap_spec": {
            "taps": list(TAP_VISITS),
            "definition": (
                "decoder-layer INPUT hidden_states (the attn-res 'partial'), "
                "captured by forward pre-hook on model.layers[i]; fp32. "
                "virtual visit index = global decoder-layer call counter: "
                "visits 0-5 prelude layers 0-5, 6-17 pass0 layers 6-17, "
                "18-29 pass1 layers 6-17, 30-41 pass2 layers 6-17, "
                "42-47 coda layers 18-23. visit5=layer5 p0, visit23=layer11 p1, "
                "visit41=layer17 p2 (matches mtp target_layer_ids {5,23,41}; "
                "id 48 = final normed hidden = hidden_final_step*)"
            ),
            "shapes": "step0: [prompt_len, 2304]; steps>=1: [1, 2304]",
        },
        "logits_spec": "fp32 [200058] post-softcap logits of last position; "
                       "unpadded vocab (unembed tail rows already dropped)",
        "files": meta_files,
        "notes": [
            "experts dequantized NVFP4/MXFP8->bf16 at load by HF "
            "Inkling2LoopDequantizeExperts; expert params offloaded to "
            "data/hf_offload/ during inference (accelerate disk offload)",
            "vision tower and mtp.safetensors not loaded (text-only harness)",
        ],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done. total step time {sum(step_times):.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
