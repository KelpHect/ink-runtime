"""Bisection dump: every decoder-layer INPUT hidden state for the prompt
forward (step 0), all 48 virtual visits. Same load/offload machinery as
dump_golden.py (see that file for the why of param-level device_map and
streaming hooks). Output: testdata/golden/visits/visitNN_in.npy [25,2304]
fp32 + visitNN_out.npy where NN is the global visit counter.
"""

import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "Inkling-2-Mini")
OUT_DIR = os.path.join(REPO, "testdata", "golden", "visits")
OFFLOAD_DIR = os.path.join(REPO, "data", "hf_offload")

USER_PROMPT = "What is the capital of France? Answer briefly."


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop import Inkling2LoopForCausalLM

    log(f"transformers {transformers.__version__}, torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": USER_PROMPT}],
        add_generation_prompt=True, tokenize=True)
    prompt_ids = enc["input_ids"]

    full_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    text_cfg = full_cfg.get_text_config()

    from accelerate import init_empty_weights
    with init_empty_weights():
        skeleton = Inkling2LoopForCausalLM(text_cfg)
    device_map = {}
    for name, _ in skeleton.named_parameters():
        device_map[name] = "disk" if ".mlp.experts." in name else "cpu"
    for name, _ in skeleton.named_buffers():
        device_map[name] = "cpu"
    del skeleton

    t0 = time.time()
    model = Inkling2LoopForCausalLM.from_pretrained(
        MODEL_DIR, config=text_cfg, dtype=torch.bfloat16,
        device_map=device_map, offload_folder=OFFLOAD_DIR,
        attn_implementation="eager")
    model.eval()
    log(f"model loaded in {time.time() - t0:.1f}s")

    text_model = model.model
    layers = text_model.layers

    from safetensors import safe_open
    EXPERT_PARAMS = ("gate_up_proj", "down_proj")

    def make_streamer(module_name):
        paths = {p: os.path.join(OFFLOAD_DIR, f"{module_name}.{p}.safetensors")
                 for p in EXPERT_PARAMS}

        def pre_hook(module, args):
            for pname, path in paths.items():
                param = module._parameters.get(pname)
                if param is not None and param.is_meta:
                    with safe_open(path, framework="pt") as f:
                        tensor = f.get_tensor(f"{module_name}.{pname}").clone()
                    module._parameters[pname] = torch.nn.Parameter(
                        tensor, requires_grad=False)

        def post_hook(module, args, output):
            for pname in EXPERT_PARAMS:
                module._parameters[pname] = torch.nn.Parameter(
                    torch.empty(0, device="meta"), requires_grad=False)

        return pre_hook, post_hook

    for i, lyr in enumerate(layers):
        pre, post = make_streamer(f"model.layers.{i}.mlp.experts")
        lyr.mlp.experts.register_forward_pre_hook(pre)
        lyr.mlp.experts.register_forward_hook(post)

    state = {"visit": -1, "ins": {}, "outs": {}}

    def in_hook(module, args):
        state["visit"] += 1
        v = state["visit"]
        state["ins"][v] = args[0][0].detach().to(torch.float32).cpu().clone()

    def out_hook(module, args, output):
        v = state["visit"]
        h = output[0] if isinstance(output, tuple) else output
        state["outs"][v] = h[0].detach().to(torch.float32).cpu().clone()

    for lyr in layers:
        lyr.register_forward_pre_hook(in_hook)
        lyr.register_forward_hook(out_hook)

    ids = torch.tensor([prompt_ids], dtype=torch.long)
    t0 = time.time()
    model(input_ids=ids, use_cache=False)
    log(f"forward done in {time.time() - t0:.1f}s, "
        f"visits={state['visit'] + 1}")

    embed_in = state["ins"][0]
    np.save(os.path.join(OUT_DIR, "embed_in.npy"), embed_in.numpy())
    for v in range(state["visit"] + 1):
        np.save(os.path.join(OUT_DIR, f"visit{v:02d}_in.npy"),
                state["ins"][v].numpy())
        np.save(os.path.join(OUT_DIR, f"visit{v:02d}_out.npy"),
                state["outs"][v].numpy())
    log(f"wrote {2 * (state['visit'] + 1) + 1} npy files -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
