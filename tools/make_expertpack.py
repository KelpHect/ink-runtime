"""Repack Inkling-2-Mini MoE expert tensors into data/expertpack.v1.bin + index.

Layout (see research/07-inkling2-loop-internals.md section 6):

  Layers 0..19  -> "nvfp4" experts (modelopt NVFP4, E2M1 packed 2-per-byte):
      w13_weight                    U8      [2304, 1152]   2 fp4 values per byte
      w13_weight_scale              F8_E4M3 [2304, 144]    1 scale byte per 16 values
      w13_weight_amax               F32     scalar         per-expert amax
      w2_weight                     U8      [2304, 576]
      w2_weight_scale               F8_E4M3 [2304, 72]
      w2_weight_amax                F32     scalar
      (w13/w2_weight_outer_scale_denominator F32 [1] is per-LAYER -> index json)

  Layers 20..23 -> "mxfp8" experts (E4M3 values, pow2 scale per 32 values):
      w13_weight                    F8_E4M3 [2304, 2304]
      w13_weight_scale              U8      [2304, 72]     scale = 2^(byte-127)
      w2_weight                     F8_E4M3 [2304, 1152]
      w2_weight_scale               U8      [2304, 36]
      (no amax / outer_scale_denominator tensors exist for mxfp8 layers)

The .bin file is a concatenation over (layer 0..23, expert 0..127) of fixed-size
blocks. Every block starts at a 4096-byte-aligned offset; unused tail bytes are
zero. All tensor bytes are copied verbatim from the safetensors file (row-major,
C order, expert dim sliced off dim 0). No dequantization is performed.

Reads through an OS-mapped view so the whole 19.5 GB file is never resident.
"""

import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "Inkling-2-Mini", "model-00001-of-00001.safetensors")
OUT_BIN = os.path.join(REPO, "data", "expertpack.v1.bin")
OUT_JSON = os.path.join(REPO, "data", "expertpack.v1.json")

NUM_LAYERS = 24
NUM_EXPERTS = 128
ALIGN = 4096

# Per-format field tables: (field_name, checkpoint_suffix, np_dtype_of_bytes)
# Sizes below are per single expert (checkpoint dim 0 sliced off).
NVFP4_FIELDS = [
    ("w13_weight", "w13_weight", "U8"),
    ("w13_weight_scale", "w13_weight_scale", "F8_E4M3"),
    ("w13_weight_amax", "w13_weight_amax", "F32"),
    ("w2_weight", "w2_weight", "U8"),
    ("w2_weight_scale", "w2_weight_scale", "F8_E4M3"),
    ("w2_weight_amax", "w2_weight_amax", "F32"),
]
MXFP8_FIELDS = [
    ("w13_weight", "w13_weight", "F8_E4M3"),
    ("w13_weight_scale", "w13_weight_scale", "U8"),
    ("w2_weight", "w2_weight", "F8_E4M3"),
    ("w2_weight_scale", "w2_weight_scale", "U8"),
]
EXPECTED_SHAPE = {
    "nvfp4": {
        "w13_weight": (128, 2304, 1152),
        "w13_weight_scale": (128, 2304, 144),
        "w13_weight_amax": (128,),
        "w2_weight": (128, 2304, 576),
        "w2_weight_scale": (128, 2304, 72),
        "w2_weight_amax": (128,),
        "w13_weight_outer_scale_denominator": (1,),
        "w2_weight_outer_scale_denominator": (1,),
    },
    "mxfp8": {
        "w13_weight": (128, 2304, 2304),
        "w13_weight_scale": (128, 2304, 72),
        "w2_weight": (128, 2304, 1152),
        "w2_weight_scale": (128, 2304, 36),
    },
}
DTYPE_SIZE = {"U8": 1, "F8_E4M3": 1, "F32": 4}


def align_up(n, a):
    return (n + a - 1) // a * a


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    data_base = 8 + hlen
    return header, data_base


def expert_byte_span(entry, expert):
    """Byte span (relative to tensor data start) of `expert` within a tensor
    whose dim-0 is the expert axis (num_experts entries)."""
    start, end = entry["data_offsets"]
    total = end - start
    per = total // NUM_EXPERTS
    assert per * NUM_EXPERTS == total, "tensor size not divisible by experts"
    return start + expert * per, start + (expert + 1) * per, per


def layer_format(layer):
    return "nvfp4" if layer <= 19 else "mxfp8"


def main():
    header, data_base = read_header(SRC)
    mm = np.memmap(SRC, dtype=np.uint8, mode="r")

    layers_meta = []
    offset = 0
    zeros = bytes(ALIGN)

    os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
    with open(OUT_BIN, "wb", buffering=1 << 22) as out:
        for L in range(NUM_LAYERS):
            fmt = layer_format(L)
            fields = NVFP4_FIELDS if fmt == "nvfp4" else MXFP8_FIELDS
            prefix = f"model.llm.layers.{L}.mlp.experts."

            # Validate shapes/dtypes and gather per-expert spans.
            spans = []
            content_bytes = 0
            field_layout = []
            foff = 0
            for fname, suffix, dtype in fields:
                key = prefix + suffix
                entry = header[key]
                want = EXPECTED_SHAPE[fmt][suffix]
                got = tuple(entry["shape"])
                assert entry["dtype"] == dtype, (key, entry["dtype"], dtype)
                assert got == want, (key, got, want)
                spans.append((fname, entry))
                nbytes = DTYPE_SIZE[dtype]
                for d in want[1:]:
                    nbytes *= d
                field_layout.append(
                    {"name": fname, "dtype": dtype, "offset": foff, "nbytes": nbytes}
                )
                foff += nbytes
                content_bytes += nbytes

            stride = align_up(content_bytes, ALIGN)
            base_offset = offset

            for e in range(NUM_EXPERTS):
                for fname, entry in spans:
                    s, t, _ = expert_byte_span(entry, e)
                    out.write(mm[data_base + s : data_base + t])
                pad = stride - content_bytes
                if pad:
                    out.write(zeros[:pad])
                offset += stride

            meta = {
                "layer": L,
                "format": fmt,
                "base_offset": base_offset,
                "num_experts": NUM_EXPERTS,
                "expert_size_bytes": content_bytes,
                "block_stride": stride,
                "fields": field_layout,
            }
            # Per-layer NVFP4 dequant denominators (scalar F32 each).
            for denom_key in (
                "w13_weight_outer_scale_denominator",
                "w2_weight_outer_scale_denominator",
            ):
                key = prefix + denom_key
                if key in header:
                    s, t = header[key]["data_offsets"]
                    (val,) = struct.unpack(
                        "<f", mm[data_base + s : data_base + t].tobytes()
                    )
                    meta[denom_key] = val
            layers_meta.append(meta)
            print(
                f"layer {L:2d} {fmt:6s} base={base_offset:>12d} "
                f"expert_bytes={content_bytes:>9d} stride={stride}",
                flush=True,
            )

    file_size = offset
    index = {
        "format_version": 1,
        "source_file": os.path.relpath(SRC, REPO).replace("\\", "/"),
        "bin_file": os.path.basename(OUT_BIN),
        "file_size_bytes": file_size,
        "alignment": ALIGN,
        "num_layers": NUM_LAYERS,
        "experts_per_layer": NUM_EXPERTS,
        "expert_order": "block index = layer * 128 + expert",
        "field_notes": {
            "nvfp4": (
                "w*_weight: two E2M1 fp4 values per byte, low nibble = even "
                "element along last dim. w*_weight_scale: one E4M3 scale per 16 "
                "values along last dim (weight_block_shape [1,16]). "
                "w*_weight_amax: per-expert F32 scalar; dequant factor is "
                "amax / outer_scale_denominator (denominator is per-layer, "
                "stored in this index). See Inkling2LoopDequantizeExperts."
            ),
            "mxfp8": (
                "w*_weight: E4M3 values, 1 byte each. w*_weight_scale: U8 "
                "power-of-two exponent, scale = 2^(byte-127), one per 32 values "
                "along last dim (block [32,32] scheme -> 72 resp. 36 scales "
                "per 2304/1152-wide row group). No amax tensors exist."
            ),
        },
        "layers": layers_meta,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(index, f, indent=2)

    print(f"wrote {OUT_BIN} ({file_size} bytes)")
    print(f"wrote {OUT_JSON}")

    # Sanity check: file size and a spot-check of two expert blocks.
    assert os.path.getsize(OUT_BIN) == file_size
    with open(OUT_BIN, "rb") as f:
        for L, e in ((0, 0), (7, 99), (20, 5), (23, 127)):
            meta = layers_meta[L]
            bo = meta["base_offset"] + e * meta["block_stride"]
            # first field is always w13_weight
            key = f"model.llm.layers.{L}.mlp.experts.w13_weight"
            entry = header[key]
            s, t, per = expert_byte_span(entry, e)
            want = bytes(mm[data_base + s : data_base + s + 4096])
            f.seek(bo)
            got = f.read(4096)
            assert got == want, f"block mismatch layer {L} expert {e}"
    print("spot-checks passed")


if __name__ == "__main__":
    main()
