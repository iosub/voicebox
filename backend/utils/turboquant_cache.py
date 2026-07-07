"""TurboQuant KV cache compression for transformer inference.

Reduces GPU VRAM usage by ~5x on KV caches during autoregressive
generation via Lloyd-Max quantization with orthogonal rotation.

Falls back silently when turboquant-gpu is not installed or device
is not CUDA.
"""

import functools
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

_TURBOQUANT_AVAILABLE: Optional[bool] = None


def is_turboquant_available() -> bool:
    """Check whether turboquant-gpu is importable."""
    global _TURBOQUANT_AVAILABLE
    if _TURBOQUANT_AVAILABLE is None:
        try:
            from turboquant_gpu import TurboQuantEngine  # noqa: F401

            _TURBOQUANT_AVAILABLE = True
        except ImportError:
            _TURBOQUANT_AVAILABLE = False
    return _TURBOQUANT_AVAILABLE


def _get_head_dim(model) -> int:
    """Detect the per-head dimension from a HuggingFace model config."""
    config = getattr(model, "config", None)
    if config is None:
        return 128

    # Explicit head_dim (some recent configs)
    if hasattr(config, "head_dim"):
        return config.head_dim

    # Encoder-decoder (Whisper)
    if hasattr(config, "d_model") and hasattr(config, "encoder_attention_heads"):
        return config.d_model // config.encoder_attention_heads

    # Causal LMs (Llama, Qwen, etc.)
    if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        return config.hidden_size // config.num_attention_heads

    return 128


def _get_kv_seq_len(past_kv) -> int:
    """Return the sequence-length dimension of a KV cache (0 on failure)."""
    try:
        if hasattr(past_kv, "key_cache") and past_kv.key_cache:
            return past_kv.key_cache[0].shape[2]
        if isinstance(past_kv, (list, tuple)) and past_kv:
            return past_kv[0][0].shape[2]
    except (IndexError, AttributeError):
        pass
    return 0


def _find_compressible_targets(model) -> Tuple[List, object]:
    """Find internal transformer sub-models that use KV caches.

    Handles wrapper patterns:
      - Standard nn.Module with .forward() → patch directly.
      - Qwen3TTSModel wrapper → patch model.model.talker (the actual
        causal LM with KV cache that runs inside GenerationMixin.generate).

    Returns:
        (targets_to_patch, reset_wrapper) where *targets_to_patch* is a list
        of nn.Module instances whose .forward() should be monkey-patched and
        *reset_wrapper* is the object on which generation-reset methods live.
    """
    # Direct nn.Module (Whisper, Llama, etc.)
    if hasattr(model, "forward") and callable(getattr(model, "forward", None)):
        return [model], model

    # Qwen3TTSModel pattern:
    #   Qwen3TTSModel.model → Qwen3TTSForConditionalGeneration
    #     .talker → Qwen3TTSTalkerForConditionalGeneration (has .forward + KV)
    inner = getattr(model, "model", None)
    if inner is not None:
        talker = getattr(inner, "talker", None)
        if talker is not None and hasattr(talker, "forward"):
            return [talker], model
        if hasattr(inner, "forward"):
            return [inner], model

    return [], model


def enable_kv_compression(
    model,
    device: str,
    *,
    total_bits: int = 3,
    min_seq_len: int = 64,
) -> bool:
    """Enable TurboQuant KV cache compression on a model.

    Monkey-patches the internal transformer's ``forward()`` to compress the
    KV cache after the first forward pass that produces a cache longer than
    *min_seq_len*.  Automatically resets when a generation method is called
    so each generation starts fresh.

    Supports wrapper classes like ``Qwen3TTSModel`` that do not themselves
    have a ``.forward()`` but contain an internal transformer (``talker``)
    that does.

    Args:
        model: A model (or wrapper) to enable compression on.
        device: Torch device string.
        total_bits: Quantization bit-width (2 or 3).
        min_seq_len: Minimum KV sequence length to trigger compression.

    Returns:
        ``True`` if compression was enabled, ``False`` otherwise.
    """
    import os
    if os.environ.get("TURBOQUANT_ENABLED", "1").lower() in ("0", "false", "no", "off"):
        logger.info("TurboQuant: disabled via TURBOQUANT_ENABLED env var")
        return False

    # Allow env var overrides for tuning
    total_bits = int(os.environ.get("TURBOQUANT_BITS", str(total_bits)))
    min_seq_len = int(os.environ.get("TURBOQUANT_MIN_SEQ", str(min_seq_len)))

    if not str(device).startswith("cuda"):
        logger.debug("TurboQuant: skipped (device=%s, need cuda)", device)
        return False

    if not is_turboquant_available():
        logger.debug("TurboQuant: skipped (package not installed)")
        return False

    try:
        import torch

        if not torch.cuda.is_available():
            return False

        from turboquant_gpu import TurboQuantEngine

        targets, reset_wrapper = _find_compressible_targets(model)
        if not targets:
            logger.warning(
                "TurboQuant: no patchable sub-models found in %s",
                type(model).__name__,
            )
            return False

        _state = {"compressed": False}

        for target in targets:
            head_dim = _get_head_dim(target)
            engine = TurboQuantEngine(
                head_dim=head_dim,
                total_bits=total_bits,
                device=str(device),
            )

            original_forward = target.forward

            def _make_forward(orig_fwd, eng):
                @functools.wraps(orig_fwd)
                def _forward_with_kv_compression(*args, **kwargs):
                    output = orig_fwd(*args, **kwargs)

                    if _state["compressed"]:
                        return output

                    past_kv = getattr(output, "past_key_values", None)
                    if past_kv is None:
                        return output

                    seq_len = _get_kv_seq_len(past_kv)
                    if seq_len < min_seq_len:
                        return output

                    try:
                        compressed = eng.compress_kv_cache(past_kv)
                        stats = eng.compression_stats(past_kv)
                        output.past_key_values = eng.build_cache(compressed)
                        _state["compressed"] = True
                        logger.info(
                            "TurboQuant: KV cache compressed (%d tokens, %.1fx ratio)",
                            seq_len,
                            stats["ratio"],
                        )
                    except Exception as e:
                        logger.debug("TurboQuant: compression skipped: %s", e)
                        _state["compressed"] = True

                    return output

                return _forward_with_kv_compression

            target.forward = _make_forward(original_forward, engine)

        # Reset compression state before each generation call so
        # each generation starts with a fresh (uncompressed) cache.
        _gen_methods = (
            "generate",
            "generate_voice_clone",
            "generate_custom_voice",
            "generate_voice_design",
        )
        for name in _gen_methods:
            orig = getattr(reset_wrapper, name, None)
            if orig is None or not callable(orig):
                continue

            def _make_resetter(fn):
                def _wrapper(*a, **kw):
                    _state["compressed"] = False
                    return fn(*a, **kw)

                return _wrapper

            setattr(reset_wrapper, name, _make_resetter(orig))

        patched_names = [type(t).__name__ for t in targets]
        logger.info(
            "TurboQuant: enabled on %s → %s (%d-bit)",
            type(model).__name__,
            ", ".join(patched_names),
            total_bits,
        )
        return True

    except Exception as e:
        logger.warning("TurboQuant: failed to enable: %s", e)
        return False
