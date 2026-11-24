# draft_probe.py — offline MTP/DFlare semantic A/B harness.
#
# Reimplements the CpuDraft block+markov pipeline in torch fp32 with flag-
# switchable interpretations of the ambiguous spec points. Feed the
# taps_w<pos>_t<tok>_step0.bin dumps written by INK_DRAFT_DEBUG runs and
# score p1..p8 hits against the tokens the model actually generated.
#
# CONFIRMED semantics (variant "dflash=1+v2=1", matches vLLM's
# _prepare_dflash_inputs_kernel fill-in layout, validated end-to-end):
#   * anchor = last committed token at its OWN position pos; its fused
#     features are appended to ctx-KV at ctxpos=pos this step.
#   * block slot 0 = embed_norm(anchor) at position pos (seed, unsampled);
#     slots 1..7 = mask_embed at pos+1..pos+7, each predicting the token
#     AT its own position.
#   * markov head: prev0 = anchor; step i consumes slot i+1's hidden/base
#     -> ids[0..6] predict pos+1..pos+7.
#   * hi_src = out8 (normed); fuse = row-softmax; ctx append required.
# Measured: d1 5/6 d2 4/6 on 6 windows (GPU run: d1 22/30 d2 14/29).
#
# Usage: .venv/Scripts/python tools/draft_probe.py [variant ...]
# Variants are dotted flags, e.g.: hi=raw fuse=col slot0emb=next ctxself=0

import glob
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MTP = os.path.join(ROOT, "Inkling-2-Mini", "mtp.safetensors")
MDL = os.path.join(ROOT, "Inkling-2-Mini", "model-00001-of-00001.safetensors")

H, I, HQ, HK, DH = 2304, 9216, 24, 8, 128
V = 200058
EPS = 1e-6
LOCAL = {0, 2}
WIN = 1024

W = {}
with safe_open(MTP, framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()
_mdl = safe_open(MDL, framework="pt")
emb_g = _mdl.get_tensor("model.llm.embed_norm.weight").float()
unemb = _mdl.get_tensor("model.llm.unembed.weight").float()
_embed_slice = _mdl.get_slice("model.llm.embed.weight")


def embed_row(tok):
    return _embed_slice[tok].float()


def rms(x, g, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean() + eps) * g


def rope(v, pos, theta=1e6):
    # GPT-NeoX half-split on each 128-d head; v is [n_head, 128]
    v = v.clone()
    d = torch.arange(64, dtype=torch.float64)
    ang = float(pos) * (theta ** (-d / 64.0))
    c, s = torch.cos(ang).float(), torch.sin(ang).float()
    x1, x2 = v[:, :64], v[:, 64:]
    v[:, :64] = x1 * c - x2 * s
    v[:, 64:] = x2 * c + x1 * s
    return v


def bf16r(x):
    return x.to(torch.bfloat16).float()


def load_taps(path):
    a = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(a.reshape(4, H))


class Ctx:
    def __init__(self):
        self.k = [torch.zeros(0, HK * DH) for _ in range(3)]
        self.v = [torch.zeros(0, HK * DH) for _ in range(3)]
        self.pos = []


def propose(taps, anchor, pos, cx, fl):
    # ---- fused ctx per consumer (rows of fuse_weight = consumers) ----
    fw = W["fuse_weight"]
    if fl.get("fuse") == "col":
        alpha = torch.softmax(fw, dim=0)  # each consumer = a COLUMN
        ctx = torch.stack(
            [rms(alpha[:, c] @ taps, W[f"layers.{c}.dflare_norm.weight"]
                 if c < 3 else W["markov_head.dflare_norm.weight"])
             for c in range(4)])
    else:
        alpha = torch.softmax(fw, dim=1)
        ctx = torch.stack(
            [rms(alpha[c] @ taps, W[f"layers.{c}.dflare_norm.weight"]
                 if c < 3 else W["markov_head.dflare_norm.weight"])
             for c in range(4)])

    # ---- block input ----
    x = torch.zeros(8, H)
    if (fl.get("dflash") or fl.get("slot0emb") == "next") \
            and "next_tok" in fl and not fl.get("v2"):
        x[0] = rms(embed_row(fl["next_tok"]), emb_g)  # bonus = just-sampled
    elif fl.get("slot0emb") == "mask":
        x[0] = W["mask_embed"]
    else:
        x[0] = rms(embed_row(anchor), emb_g)
    x[1:] = W["mask_embed"]

    # ---- ctx-KV append for the anchor's fused features ----
    for li in range(3):
        kc = W[f"layers.{li}.self_attn.k_proj_ctx.weight"] @ ctx[li]
        vc = W[f"layers.{li}.self_attn.v_proj_ctx.weight"] @ ctx[li]
        kc = kc.view(HK, DH)
        kc = torch.stack(
            [rms(kc[h], W[f"layers.{li}.self_attn.k_norm_ctx.weight"])
             for h in range(HK)])
        kc = rope(kc, pos + int(fl.get("ctxpos_off", 0)))
        if fl.get("append", 1):
            cx.k[li] = torch.cat([cx.k[li], kc.view(1, -1)])
            cx.v[li] = torch.cat([cx.v[li], vc.view(1, -1)])
    if fl.get("append", 1):
        cx.pos.append(pos + int(fl.get("ctxpos_off", 0)))

    for li in range(3):
        nctx = len(cx.pos)
        xn = torch.stack([rms(x[j], W[f"layers.{li}.input_layernorm.weight"])
                          for j in range(8)])
        q = (W[f"layers.{li}.self_attn.q_proj.weight"] @ xn.T).T
        kb = (W[f"layers.{li}.self_attn.k_proj.weight"] @ xn.T).T
        vb = (W[f"layers.{li}.self_attn.v_proj.weight"] @ xn.T).T
        q = q.view(8, HQ, DH)
        kb = kb.view(8, HK, DH)
        vb = vb.view(8, HK, DH)
        qn = W[f"layers.{li}.self_attn.q_norm.weight"]
        kn = W[f"layers.{li}.self_attn.k_norm.weight"]
        q = torch.stack([torch.stack([rms(q[j, h], qn) for h in range(HQ)])
                         for j in range(8)])
        kb = torch.stack([torch.stack([rms(kb[j, h], kn) for h in range(HK)])
                          for j in range(8)])
        boff = int(fl.get("blockpos_off", 0))
        if fl.get("dflash"):
            boff = 1 if not fl.get("v2") else 0
        q = torch.stack([rope(q[j], pos + j + boff) for j in range(8)])
        kb = torch.stack([rope(kb[j], pos + j + boff) for j in range(8)])
        att = torch.zeros(8, HQ, DH)
        local = li in LOCAL
        for j in range(8):
            qpos = pos + j + boff
            keys = []
            vals = []
            vis = []
            for kk in range(nctx):
                kpos = cx.pos[kk]
                ok = (not local) or (kpos <= qpos and qpos - kpos < WIN)
                vis.append(ok)
                keys.append(cx.k[li][kk].view(HK, DH))
                vals.append(cx.v[li][kk].view(HK, DH))
            for kk in range(8):
                vis.append(True)
                keys.append(kb[kk])
                vals.append(vb[kk])
            K = torch.stack(keys)  # [nk, HK, DH]
            Vv = torch.stack(vals)
            vism = torch.tensor(vis)
            for h in range(HQ):
                g = h * HK // HQ
                sc = (K[:, g] @ q[j, h]) / (DH ** 0.5)
                sc = sc.masked_fill(~vism, -1e30)
                p = torch.softmax(sc, 0)
                att[j, h] = (p[:, None] * Vv[:, g]).sum(0)
        ao = (W[f"layers.{li}.self_attn.o_proj.weight"]
              @ att.reshape(8, -1).T).T
        x = bf16r(x + ao)
        xn = torch.stack(
            [rms(x[j], W[f"layers.{li}.post_attention_layernorm.weight"])
             for j in range(8)])
        gg = (W[f"layers.{li}.mlp.gate_proj.weight"] @ xn.T).T
        uu = (W[f"layers.{li}.mlp.up_proj.weight"] @ xn.T).T
        aa = gg * torch.sigmoid(gg) * uu
        mo = (W[f"layers.{li}.mlp.down_proj.weight"] @ aa.T).T
        x = bf16r(x + mo)

    out8 = torch.stack([rms(x[j], W["norm.weight"]) for j in range(8)])
    base = (unemb[:V] @ out8.T).T  # [8,V]

    # ---- markov head ----
    ids = []
    state = torch.zeros(256)
    prev = anchor if (not fl.get("dflash") or fl.get("v2")) \
        else fl["next_tok"]
    hi_src = x if fl.get("hi") == "raw" else out8
    f_rnn = ctx[3] if fl.get("f_rnn", 3) == 3 else ctx[int(fl["f_rnn"])]
    # dflash fill-in: 7 sampled slots are 1..7, so markov step i reads
    # hidden/base of slot i+1 and predicts the token AT that slot's pos.
    slot_off = 1 if fl.get("dflash") else 0
    n_steps = 8 if fl.get("mstep8") else 7
    for i in range(n_steps):
        si = min(i + slot_off, 7)
        if fl.get("rnn_order") == "w1first":
            inp = torch.cat([W["markov_head.markov_w1.weight"][prev],
                             bf16r(state), bf16r(hi_src[si]), bf16r(f_rnn)])
        else:
            inp = torch.cat([bf16r(state),
                             W["markov_head.markov_w1.weight"][prev],
                             bf16r(hi_src[si]), bf16r(f_rnn)])
        proj = W["markov_head.rnn_proj.weight"] @ inp \
            + W["markov_head.rnn_proj.bias"]
        g = torch.sigmoid(proj[:256])
        cnd = torch.tanh(proj[256:512])
        lat = torch.tanh(proj[512:768])
        state = g * state + (1 - g) * cnd
        dvec = W["markov_head.hidden_out.weight"] @ lat \
            + W["markov_head.hidden_out.bias"]
        lg = 30 * torch.tanh(bf16r(bf16r(base[si]) + unemb[:V] @ dvec) / 30)
        t = int(lg.argmax())
        ids.append(t)
        prev = t
    if not fl.get("dflash") and not fl.get("mstep8"):
        ids.append(int(base[7].argmax()))
    return ids


def main():
    variants = [dict()]  # baseline
    for a in sys.argv[1:]:
        for grp in a.split(","):
            base = dict()
            for kv in grp.split("+"):
                k, v = kv.split("=")
                try:
                    v = int(v)
                except ValueError:
                    pass
                base[k] = v
            variants.append(base)

    taps_files = sorted(glob.glob(os.path.join(ROOT, "taps_w*_t*_step0.bin")))
    # actual generated tokens by absolute position (pos -> token at pos)
    actual = {26: 200008, 27: 371, 28: 150409, 29: 171960, 30: 149878,
              31: 150744, 32: 151231, 33: 1355, 34: 150377, 35: 25}
    rows = []
    for fl in variants:
        cx = Ctx()
        hits = [0] * 8
        tot = [0] * 8
        per_w = []
        shift = int(fl.get("tap_shift", 0))
        for wi, tf in enumerate(taps_files):
            m = re.search(r"taps_w(\d+)_t(\d+)", tf)
            pos, tok = int(m.group(1)), int(m.group(2))
            use = min(wi + shift, len(taps_files) - 1)
            if pos + 1 in actual:
                fl["next_tok"] = actual[pos + 1]
            ids = propose(load_taps(taps_files[use]), tok, pos, cx, fl)
            w_hits = []
            # dflash fill-in: proposals = 7 tokens at pos+2..pos+8
            # (v2: anchor at own pos → predictions pos+1..pos+7)
            off = int(fl.get("score_off", 1))
            if fl.get("dflash"):
                off = 1 if fl.get("v2") else 2
            for j in range(len(ids)):
                q = pos + off + j
                if q in actual:
                    tot[j] += 1
                    if ids[j] == actual[q]:
                        hits[j] += 1
                        w_hits.append(j + 1)
            per_w.append(f"{pos}:{ids[:4]} hits{w_hits}")
        acc = ", ".join(f"d{j+1}:{hits[j]}/{tot[j]}" for j in range(8)
                        if tot[j])
        print(f"--- {fl}")
        for w in per_w:
            print("    ", w)
        print("     ", acc)


if __name__ == "__main__":
    main()
