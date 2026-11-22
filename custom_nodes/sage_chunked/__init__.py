"""Run SageAttention 2 over head GROUPS, to cut its scratch allocation.

THE PROBLEM THIS SOLVES. On a 16 GB card the binding limit for H3 is not the
model weights — it is the attention workspace, and it is bigger than anyone
documents. At a 15.1s clip (109,062 tokens padded to 109,184, 56 heads, dim 128):

    int8 Q, K, V   0.73 GiB each  = 2.19 GiB
    output (bf16)                 = 1.46 GiB
                                    --------
                                    3.64 GiB

The 5060 Ti had ~113 MiB free at that moment and sage needed 748 MiB more, so
comfy caught the failure and SILENTLY substituted attention_pytorch — measured
2.9x slower per call (7271.8 ms vs 2499.3 ms). A 15s clip at native therefore
looked "impossible"; it was actually one allocation short.

THE FIX. Attention is independent per head: softmax(QK^T)V for head h touches no
other head. So the heads can be processed in groups and the results
concatenated, and the answer is EXACTLY the same — this is not an approximation,
no windowing, no sparsity, no precision change. Only the int8 scratch scales
with the group size:

    groups   peak      saves
      1      3.64 GiB    —        (current behaviour)
      2      2.55 GiB  1.09 GiB
      4      2.00 GiB  1.64 GiB   <- default
      8      1.73 GiB  1.91 GiB

Against a 0.73 GiB shortfall, 4 groups leaves real margin.

COST. One kernel launch per group instead of one per call, and the output is
written in slices. Both are small next to a multi-second attention call, but it
is not free — measure before assuming.

WHY A NODE. Same mechanism as custom_nodes/sage3_select: comfy's wrap_attn
consults transformer_options["optimized_attention_override"] before dispatching,
so a per-model override needs no patching of comfy itself.

H3 hands attention (1, heads, seq, dim) with skip_reshape=True, which is the
layout this chunks along dim=1. Other layouts are passed through untouched
rather than guessed at.
"""
import logging

import torch


class SageAttentionChunked:
    """Process attention in head groups so its scratch buffer shrinks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "groups": ("INT", {"default": 4, "min": 1, "max": 16,
                                   "tooltip": "Head groups per attention call. 56 heads / 4 = "
                                              "14 per group. Higher saves more memory for more "
                                              "kernel launches. 1 disables chunking."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/attention"
    DESCRIPTION = ("Run SageAttention over head groups to cut its workspace. "
                   "Mathematically exact — heads are independent.")

    def apply(self, model, groups):
        from comfy.ldm.modules.attention import get_attention_function

        sage = get_attention_function("sage", None)
        if sage is None:
            raise RuntimeError("SageAttention is not registered — launch with "
                               "--use-sage-attention so comfy registers it.")

        m = model.clone()
        opts = dict(m.model_options.get("transformer_options", {}))

        def override(func, *args, **kwargs):
            # kwargs pass through UNTOUCHED — _inside_attn_wrapper is wrap_attn's
            # recursion guard and stripping it makes the override re-enter itself
            # forever. (That exact bug made sage3_select fatal until it was found.)
            q, k, v = args[0], args[1], args[2]
            heads = args[3] if len(args) > 3 else kwargs.get("heads")

            # Only the (B, H, N, D) skip_reshape layout is chunked. Anything else
            # is handed straight to sage rather than reshaped on a guess.
            if (groups <= 1 or not kwargs.get("skip_reshape")
                    or q.dim() != 4 or q.shape[1] != heads or heads % groups != 0):
                return sage(*args, **kwargs)

            per = heads // groups
            kw = {**kwargs, "skip_output_reshape": True}

            # Write each group into ONE pre-allocated output. Collecting the
            # groups in a list and torch.cat-ing them would defeat the entire
            # purpose: the four chunks (1.46 GiB total) stay alive while cat
            # allocates a second full tensor, peaking at 2.92 GiB — worse, at
            # that instant, than the 3.64 GiB one-shot call being replaced.
            # Writing in place holds one output plus one group's scratch:
            # 1.46 + 0.55 = 2.00 GiB.
            out = None
            for i in range(0, heads, per):
                # Slices are views, so building a group copies nothing.
                part = sage(q[:, i:i + per], k[:, i:i + per], v[:, i:i + per], per, **kw)
                if out is None:
                    out = torch.empty((part.shape[0], heads) + tuple(part.shape[2:]),
                                      dtype=part.dtype, device=part.device)
                out[:, i:i + per] = part
                del part          # free this group's buffers before the next

            # Re-apply the output shaping the caller asked for, since each group
            # was forced to skip it so the pieces stayed concatenable.
            if not kwargs.get("skip_output_reshape"):
                b, _, n, d = out.shape
                out = out.transpose(1, 2).reshape(b, n, heads * d)
            return out

        opts["optimized_attention_override"] = override
        m.model_options["transformer_options"] = opts
        logging.info(f"SageAttentionChunked: attention runs in {groups} head groups "
                     f"(exact; cuts sage's int8 scratch to ~1/{groups})")
        return (m,)


NODE_CLASS_MAPPINGS = {"SageAttentionChunked": SageAttentionChunked}
NODE_DISPLAY_NAME_MAPPINGS = {"SageAttentionChunked": "SageAttention (chunked heads)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
