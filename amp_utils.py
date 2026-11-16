import torch


def bf16_supported(device=None):
    """True when the active CUDA device can run bfloat16 autocast."""
    if not torch.cuda.is_available():
        return False
    if device is not None and str(device).startswith('cpu'):
        return False
    if hasattr(torch.cuda, 'is_bf16_supported'):
        return torch.cuda.is_bf16_supported()
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8  # Ampere and newer


def resolve_amp(enabled, dtype_name, device, logger=None):
    """Pick autocast dtype + grad-scaler state.

    Returns ``(use_amp, dtype, scaler_enabled)``. bf16 runs unscaled on Ampere+;
    on older GPUs (T4 = Turing) we fall back to fp16 with grad scaling.
    """
    use_amp = bool(enabled) and device.type == 'cuda'
    dtype = torch.bfloat16 if str(dtype_name).lower() in ('bf16', 'bfloat16') else torch.float16

    if use_amp and dtype is torch.bfloat16 and not bf16_supported(device):
        msg = 'bf16 autocast unsupported on this GPU, falling back to fp16'
        (logger.warning if logger else print)(msg)
        dtype = torch.float16

    scaler_enabled = use_amp and dtype is torch.float16
    if logger is not None:
        logger.info(f'amp: {use_amp} dtype: {str(dtype).split(".")[-1]} '
                    f'grad_scaler: {scaler_enabled}')
    return use_amp, dtype, scaler_enabled
