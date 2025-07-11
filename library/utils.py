import logging
import sys
import threading
from typing import *
import json
import struct
from logging import LogRecord
import os

import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import EulerAncestralDiscreteScheduler
import diffusers.schedulers.scheduling_euler_ancestral_discrete
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteSchedulerOutput
import cv2
from PIL import Image
import numpy as np
from safetensors.torch import load_file

_logging_configured = False
_original_log_level = None

def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


# region Logging


def add_logging_arguments(parser):
    parser.add_argument(
        "--console_log_level",
        type=str,
        default=None, # Default to None, will fallback to INFO or forced DEBUG
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level, default is INFO / ログレベルを設定する。デフォルトはINFO",
    )
    parser.add_argument(
        "--console_log_file",
        type=str,
        default=None,
        help="Log to a file instead of stderr / 標準エラー出力ではなくファイルにログを出力する",
    )
    parser.add_argument("--console_log_simple", action="store_true", help="Simple log output / シンプルなログ出力")

class CustomFormatter(logging.Formatter): # Copied from my previous suggestion
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s %(levelname)-8s %(name)-25s %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
    
def setup_logging(args=None, log_level_override=None, reset=False):
    global _logging_configured, _original_log_level

    if _logging_configured and not reset:
        if log_level_override:
            current_level_val = getattr(logging, str(log_level_override).upper(), None)
            if current_level_val is not None and logging.root.level != current_level_val:
                logging.root.setLevel(current_level_val)
                for handler_loop_var in logging.root.handlers: # CORRECTED: handler_ H -> handler_loop_var
                    handler_loop_var.setLevel(current_level_val) # CORRECTED
        return

    if log_level_override:
        log_level_str = str(log_level_override).upper()
        if _original_log_level is None and args and args.console_log_level:
             _original_log_level = args.console_log_level
        elif _original_log_level is None:
             _original_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    elif args and args.console_log_level:
        log_level_str = args.console_log_level.upper()
        _original_log_level = log_level_str
    else:
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        _original_log_level = log_level_str

    if not log_level_override and not (args and args.console_log_level):
        log_level_str = "DEBUG" # Force DEBUG for diagnostics

    log_level_val = getattr(logging, log_level_str, logging.INFO)

    if reset or not _logging_configured:
        for handler_loop_var_clear in logging.root.handlers[:]: # CORRECTED: handler_R -> handler_loop_var_clear
            logging.root.removeHandler(handler_loop_var_clear) # CORRECTED
            if hasattr(handler_loop_var_clear, 'close'): # CORRECTED
                handler_loop_var_clear.close() # CORRECTED

    msg_init = None
    handler_to_add = None
    formatter_to_use = None # Use a distinct variable for the formatter

    if args and args.console_log_file:
        handler_to_add = logging.FileHandler(args.console_log_file, mode="w")
        formatter_to_use = logging.Formatter( # Using the more detailed formatter for files too
            fmt="%(asctime)s %(levelname)-8s %(name)-25s %(message)s (%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console
                handler_to_add = RichHandler(
                    console=Console(stderr=True), show_time=True, show_level=True, show_path=False, # Show path can be noisy
                    markup=False, log_time_format="[%Y-%m-%d %H:%M:%S]", level=log_level_val,
                    rich_tracebacks=True, # Good for exceptions
                )
                # RichHandler uses its own formatter, so set formatter_to_use to None
                formatter_to_use = None
                msg_init = "Using Rich logging."
            except ImportError:
                msg_init = "rich is not installed, using basic StreamHandler logging with CustomFormatter"
                handler_to_add = logging.StreamHandler(sys.stdout)
                formatter_to_use = CustomFormatter() # Use the custom one for basic stream handler

        if handler_to_add is None: # Fallback if Rich isn't used and simple is requested
            handler_to_add = logging.StreamHandler(sys.stdout)
            formatter_to_use = CustomFormatter() # Use the custom one for basic stream handler

    if formatter_to_use:
        handler_to_add.setFormatter(formatter_to_use)
    
    handler_to_add.setLevel(log_level_val) # Set level on the handler itself

    # Configure the root logger
    # logging.basicConfig(level=log_level_val, handlers=[handler_to_add], force=True) # basicConfig can sometimes be tricky with existing configs

    # More direct configuration of the root logger:
    root_logger = logging.getLogger()
    # Clear any handlers basicConfig might have added if we're re-configuring
    if reset or not _logging_configured: # Only clear if truly resetting or first config
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
            if hasattr(h, 'close'): h.close()
    
    root_logger.addHandler(handler_to_add)
    root_logger.setLevel(log_level_val)


    if msg_init is not None:
        local_logger = logging.getLogger("library.utils.init")
        local_logger.info(msg_init)

    test_logger = logging.getLogger("library.utils.test")
    test_logger.info(f"Logging has been set up. Effective Level: {logging.getLevelName(root_logger.getEffectiveLevel())} (Requested: {log_level_str}).")
    test_logger.debug(f"This is a DEBUG test message from library.utils.setup_logging.")

    _logging_configured = True


# Force DEBUG for this diagnostic session by calling with log_level_override
# This will be executed when utils.py is imported.
setup_logging(log_level_override="DEBUG", reset=True) # Add reset=True for safety during diagnostics

logger = logging.getLogger(__name__) # `__name__` will be 'library.utils'


def swap_weight_devices(layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # # convert to float16 if float8 is not supported
            # print(f"Warning: {dtype_str} is not supported in this PyTorch version. Converting to float16.")
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def load_safetensors(
    path: str, device: Union[str, torch.device], 
    disable_mmap: bool = False, 
    dtype: Optional[torch.dtype] = torch.float32, # Default dtype for conversion *after* loading
    dtype_load_as_is: bool = False # New flag
) -> dict[str, torch.Tensor]:
    target_device = torch.device(device) # Ensure it's a torch.device object

    if disable_mmap:
        state_dict = {}
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                tensor = f.get_tensor(key) # Loads in its stored dtype to CPU
                if not dtype_load_as_is and dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                state_dict[key] = tensor.to(target_device)
        return state_dict
    else:
        # safetensors.torch.load_file loads directly to the specified device
        # and can cast dtype *if the tensor is not already on that device*.
        # To load as is first, then cast, it's often easier to load to CPU.
        if dtype_load_as_is:
            # Load to CPU first, keeping original dtypes
            state_dict = load_file(path, device="cpu")
            # Then manually move/cast
            for key in state_dict.keys():
                tensor = state_dict[key]
                if dtype is not None: # This dtype is for the final model, not necessarily for loading
                    # This logic might be slightly off. If dtype_load_as_is is true,
                    # we might not want to cast with `dtype` here.
                    # The casting should happen after model.load_state_dict, when model.to(dtype) is called.
                    # So, if dtype_load_as_is, we just load and move to device.
                    pass # No dtype conversion here if loading as is
                state_dict[key] = tensor.to(target_device)
            return state_dict
        else:
            # Original behavior: load_file handles device and can do some dtype conversion
            try:
                state_dict = load_file(path, device=device)
            except Exception: # Broader catch if specific device load fails
                 logger.warning(f"load_file to device '{device}' failed, trying CPU then moving.")
                 state_dict = load_file(path, device="cpu")
                 for key in state_dict.keys():
                     state_dict[key] = state_dict[key].to(target_device)

            if dtype is not None:
                for key in state_dict.keys():
                    if state_dict[key].dtype != dtype: # Only cast if different
                        state_dict[key] = state_dict[key].to(dtype=dtype)
            return state_dict

# endregion

# region Image utils


def pil_resize(image, size, interpolation):
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False

    if has_alpha:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    resized_pil = pil_image.resize(size, resample=interpolation)

    # Convert back to cv2 format
    if has_alpha:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGBA2BGRA)
    else:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

    return resized_cv2


def resize_image(image: np.ndarray, width: int, height: int, resized_width: int, resized_height: int, resize_interpolation: Optional[str] = None):
    """
    Resize image with resize interpolation. Default interpolation to AREA if image is smaller, else LANCZOS.

    Args:
        image: numpy.ndarray
        width: int Original image width
        height: int Original image height
        resized_width: int Resized image width
        resized_height: int Resized image height
        resize_interpolation: Optional[str] Resize interpolation method "lanczos", "area", "bilinear", "bicubic", "nearest", "box"

    Returns:
        image
    """

    # Ensure all size parameters are actual integers
    width = int(width)
    height = int(height)
    resized_width = int(resized_width)
    resized_height = int(resized_height)

    if resize_interpolation is None:
        if width >= resized_width and height >= resized_height:
            resize_interpolation = "area"
        else:
            resize_interpolation = "lanczos"

    # we use PIL for lanczos (for backward compatibility) and box, cv2 for others
    use_pil = resize_interpolation in ["lanczos", "lanczos4", "box"]

    resized_size = (resized_width, resized_height)
    if use_pil:
        interpolation = get_pil_interpolation(resize_interpolation)
        image = pil_resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (PIL)")
    else:
        interpolation = get_cv2_interpolation(resize_interpolation)
        image = cv2.resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (cv2)")

    return image


def get_cv2_interpolation(interpolation: Optional[str]) -> Optional[int]:
    """
    Convert interpolation value to cv2 interpolation integer

    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    """
    if interpolation is None:
        return None 

    if interpolation == "lanczos" or interpolation == "lanczos4":
        # Lanczos interpolation over 8x8 neighborhood 
        return cv2.INTER_LANCZOS4
    elif interpolation == "nearest":
        # Bit exact nearest neighbor interpolation. This will produce same results as the nearest neighbor method in PIL, scikit-image or Matlab. 
        return cv2.INTER_NEAREST_EXACT
    elif interpolation == "bilinear" or interpolation == "linear":
        # bilinear interpolation
        return cv2.INTER_LINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # bicubic interpolation 
        return cv2.INTER_CUBIC
    elif interpolation == "area":
        # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        return cv2.INTER_AREA
    elif interpolation == "box":
        # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        return cv2.INTER_AREA
    else:
        return None

def get_pil_interpolation(interpolation: Optional[str]) -> Optional[Image.Resampling]:
    """
    Convert interpolation value to PIL interpolation

    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    """
    if interpolation is None:
        return None 

    if interpolation == "lanczos":
        return Image.Resampling.LANCZOS
    elif interpolation == "nearest":
        # Pick one nearest pixel from the input image. Ignore all other input pixels.
        return Image.Resampling.NEAREST
    elif interpolation == "bilinear" or interpolation == "linear":
        # For resize calculate the output pixel value using linear interpolation on all pixels that may contribute to the output value. For other transformations linear interpolation over a 2x2 environment in the input image is used.
        return Image.Resampling.BILINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # For resize calculate the output pixel value using cubic interpolation on all pixels that may contribute to the output value. For other transformations cubic interpolation over a 4x4 environment in the input image is used.
        return Image.Resampling.BICUBIC
    elif interpolation == "area":
        # Image.Resampling.BOX may be more appropriate if upscaling 
        # Area interpolation is related to cv2.INTER_AREA
        # Produces a sharper image than Resampling.BILINEAR, doesn’t have dislocations on local level like with Resampling.BOX.
        return Image.Resampling.HAMMING
    elif interpolation == "box":
        # Each pixel of source image contributes to one pixel of the destination image with identical weights. For upscaling is equivalent of Resampling.NEAREST.
        return Image.Resampling.BOX
    else:
        return None

def validate_interpolation_fn(interpolation_str: str) -> bool:
    """
    Check if a interpolation function is supported
    """
    return interpolation_str in ["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area", "box"]

# endregion

# TODO make inf_utils.py
# region Gradual Latent hires fix


class GradualLatent:
    def __init__(
        self,
        ratio,
        start_timesteps,
        every_n_steps,
        ratio_step,
        s_noise=1.0,
        gaussian_blur_ksize=None,
        gaussian_blur_sigma=0.5,
        gaussian_blur_strength=0.5,
        unsharp_target_x=True,
    ):
        self.ratio = ratio
        self.start_timesteps = start_timesteps
        self.every_n_steps = every_n_steps
        self.ratio_step = ratio_step
        self.s_noise = s_noise
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_strength = gaussian_blur_strength
        self.unsharp_target_x = unsharp_target_x

    def __str__(self) -> str:
        return (
            f"GradualLatent(ratio={self.ratio}, start_timesteps={self.start_timesteps}, "
            + f"every_n_steps={self.every_n_steps}, ratio_step={self.ratio_step}, s_noise={self.s_noise}, "
            + f"gaussian_blur_ksize={self.gaussian_blur_ksize}, gaussian_blur_sigma={self.gaussian_blur_sigma}, gaussian_blur_strength={self.gaussian_blur_strength}, "
            + f"unsharp_target_x={self.unsharp_target_x})"
        )

    def apply_unshark_mask(self, x: torch.Tensor):
        if self.gaussian_blur_ksize is None:
            return x
        blurred = transforms.functional.gaussian_blur(x, self.gaussian_blur_ksize, self.gaussian_blur_sigma)
        # mask = torch.sigmoid((x - blurred) * self.gaussian_blur_strength)
        mask = (x - blurred) * self.gaussian_blur_strength
        sharpened = x + mask
        return sharpened

    def interpolate(self, x: torch.Tensor, resized_size, unsharp=True):
        org_dtype = x.dtype
        if org_dtype == torch.bfloat16:
            x = x.float()

        x = torch.nn.functional.interpolate(x, size=resized_size, mode="bicubic", align_corners=False).to(dtype=org_dtype)

        # apply unsharp mask / アンシャープマスクを適用する
        if unsharp and self.gaussian_blur_ksize:
            x = self.apply_unshark_mask(x)

        return x


class EulerAncestralDiscreteSchedulerGL(EulerAncestralDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resized_size = None
        self.gradual_latent = None

    def set_gradual_latent_params(self, size, gradual_latent: GradualLatent):
        self.resized_size = size
        self.gradual_latent = gradual_latent

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            # logger.warning(
            print(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        device = model_output.device
        if self.resized_size is None:
            prev_sample = sample + derivative * dt

            noise = diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=device, generator=generator
            )
            s_noise = 1.0
        else:
            print("resized_size", self.resized_size, "model_output.shape", model_output.shape, "sample.shape", sample.shape)
            s_noise = self.gradual_latent.s_noise

            if self.gradual_latent.unsharp_target_x:
                prev_sample = sample + derivative * dt
                prev_sample = self.gradual_latent.interpolate(prev_sample, self.resized_size)
            else:
                sample = self.gradual_latent.interpolate(sample, self.resized_size)
                derivative = self.gradual_latent.interpolate(derivative, self.resized_size, unsharp=False)
                prev_sample = sample + derivative * dt

            noise = diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                (model_output.shape[0], model_output.shape[1], self.resized_size[0], self.resized_size[1]),
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )

        prev_sample = prev_sample + noise * sigma_up * s_noise

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


# endregion
