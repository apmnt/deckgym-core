import torch
import torch.nn as nn


_COMPILED_PREFIX = "_orig_mod."


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device")
    return device


def configure_device(device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported AMP dtype: {name}")


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")
    return torch.compile(module)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith(_COMPILED_PREFIX) for key in state_dict):
        return state_dict
    return {
        key[len(_COMPILED_PREFIX) :] if key.startswith(_COMPILED_PREFIX) else key: value
        for key, value in state_dict.items()
    }


def _load_checkpoint(checkpoint, map_location) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Returns (state_dict, config). Handles both the modern wrapped format
    ({"state_dict": ..., "config": ...}) and legacy raw state dicts.
    weights_only restricts unpickling to tensors/containers (no code exec)."""
    obj = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return _normalize_state_dict(obj["state_dict"]), obj.get("config")
    return _normalize_state_dict(obj), None


def load_agent_state(agent: nn.Module, checkpoint, map_location) -> None:
    state_dict, _ = _load_checkpoint(checkpoint, map_location)
    agent.load_state_dict(state_dict)


def agent_from_checkpoint(checkpoint, obs_dim: int, act_feat_dim: int, map_location) -> nn.Module:
    """Build whichever architecture/size the checkpoint was trained with."""
    from agent import agent_from_state_dict

    state_dict, config = _load_checkpoint(checkpoint, map_location)
    return agent_from_state_dict(state_dict, obs_dim, act_feat_dim, config)


def save_agent_state(agent: nn.Module, checkpoint) -> None:
    state_dict = _normalize_state_dict(agent.state_dict())
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
            "config": getattr(agent, "config", None),
        },
        checkpoint,
    )
