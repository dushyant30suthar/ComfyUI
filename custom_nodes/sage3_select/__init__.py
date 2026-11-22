"""Make SageAttention 3 (Blackwell FP4) reachable from a workflow.

ComfyUI v0.30.1 already *dispatches* sage3 — comfy/ldm/modules/attention.py
imports `sageattn3.sageattn3_blackwell` and registers it as attention function
"sage3". What it does not have is any way to SELECT it: `--use-sage-attention`
hard-selects sage2, and no stock node touches the override hook. So sage3 ships
dead unless something sets it.

The hook is in attention.py's wrapper:

    if "optimized_attention_override" in transformer_options:
        return transformer_options["optimized_attention_override"](func, *args, **kwargs)

i.e. the override is called with the original function first, then the real
attention arguments. This node just puts `attention3_sage` there.

Why bother: H3 runs full dense attention over a packed sequence — ~58% of FLOPs
at a 5s clip and ~80% at 15s. SageAttention 2 already measured about -50% at
real clip length on this box; sage3 quantises Q, K AND V to FP4 with per-block
microscaling, which is native on sm_120 (these 5060 Tis), so it is the one
remaining lever of comparable size.

FP4 is a real precision cut, not a free win. Judge output, not just the clock.

MEASURED LIMIT (2026-08-04, 2x 5060 Ti 16 GB) — read before using this at length:

  5.2s clip  (37,710 tokens)  245 ms/call vs sage2's 304 ms   -19.5%, works
  15.1s clip (106,930 tokens) OOM: "Tried to allocate 9.33 GiB", falls back

`attention3_sage` CATCHES the failure and silently reroutes to
attention_pytorch, which at that shape costs 7272 ms against sage2's 2499 ms —
so enabling this node for long clips does not merely lose the benefit, it makes
generation ~2.9x slower per attention call while the graph reports success and
this node logs that sage3 was selected. The crossover between the two shapes is
untested; treat anything past a ~5s clip as unverified and check the server log
for "Error running SageAttention3" before believing a result.
"""
import logging


class SageAttention3Select:
    """Route this model's attention through SageAttention 3 (FP4, sm_120)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/attention"
    DESCRIPTION = "Use SageAttention 3 (Blackwell FP4) for this model's attention."

    def apply(self, model):
        from comfy.ldm.modules.attention import get_attention_function

        fn = get_attention_function("sage3", None)
        if fn is None:
            raise RuntimeError(
                "sage3 is not registered. Install the kernel with:\n"
                "  cd /tmp/SageAttention/sageattention3_blackwell && "
                "uv pip install --python .venv/bin/python --no-build-isolation .\n"
                "It needs nvcc on PATH and CUDAHOSTCXX=/usr/bin/g++-15 "
                "(CUDA 13.3 rejects Fedora's default gcc 16)."
            )

        m = model.clone()
        # copy, don't mutate: model_options is shared with the source patcher and
        # an in-place edit would silently apply sage3 to every other branch of
        # the graph that reused this model.
        opts = dict(m.model_options.get("transformer_options", {}))

        def override(func, *args, **kwargs):
            # Pass kwargs through UNTOUCHED — in particular keep
            # _inside_attn_wrapper, which wrap_attn set to True just before
            # calling us. It is the recursion guard, and it has to survive two
            # hops: `fn` is the *wrapped* sage3, so without the key its own
            # wrapper would consult optimized_attention_override and call us
            # again; and attention3_sage falls back to attention_pytorch
            # (small heads, dim>=256, N<=1024, masked, or kernel error) passing
            # **kwargs on, so that wrapper would re-enter us too. Stripping the
            # key here is what made every K10/K11 row die in ~1s with
            # RecursionError: maximum recursion depth exceeded.
            return fn(*args, **kwargs)

        opts["optimized_attention_override"] = override
        m.model_options["transformer_options"] = opts
        logging.info("SageAttention3Select: attention routed to sage3 (FP4)")
        # Do NOT quote comfy's fallback string verbatim here: the obvious way to
        # check whether sage3 actually ran is to grep the server log for it, and
        # a warning that contains it makes every run look like it fell back.
        logging.warning(
            "SageAttention3Select: sage3 OOMs past ~5s of clip (needs ~9.3 GiB of "
            "workspace at 15s geometry on a 15.5 GiB card) and comfy then falls "
            "back to pytorch attention SILENTLY — which is ~2.9x slower than the "
            "sage2 you would have had. Before trusting a long-clip run, grep the "
            "server log for a SageAttention3 error emitted by comfy itself "
            "(logging.error in comfy/ldm/modules/attention.py attention3_sage).")
        return (m,)


NODE_CLASS_MAPPINGS = {"SageAttention3Select": SageAttention3Select}
NODE_DISPLAY_NAME_MAPPINGS = {"SageAttention3Select": "SageAttention 3 (FP4)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
