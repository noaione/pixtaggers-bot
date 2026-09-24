import ctypes
import importlib
import site
import sys
import time
from hashlib import md5
from importlib import util as importutil
from pathlib import Path

import onnxruntime as ort

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "engine-data"


__preloaded_ort__: type[ort] | None = None  # pyright: ignore[reportInvalidTypeForm]


def _get_onnxruntime():
    global __preloaded_ort__

    if __preloaded_ort__ is not None:
        return __preloaded_ort__

    for sp in site.getsitepackages():
        ort_dll = Path(sp) / "onnxruntime" / "capi" / "onnxruntime.dll"
        if ort_dll.exists() and sys.platform == "win32":
            ctypes.WinDLL(str(ort_dll))
            break

    # Preload TensorRT if available
    is_tensorrt_available = importutil.find_spec("tensorrt")
    if is_tensorrt_available is not None:
        import tensorrt  # type: ignore # ruff: ignore[unused-import]

    import onnxruntime as ort  # type: ignore

    is_trt_ep_available = importutil.find_spec("onnxruntime_ep_nv_tensorrt_rtx")

    if sys.platform != "darwin":
        ort.preload_dlls()

    if is_trt_ep_available is not None:
        import onnxruntime_ep_nv_tensorrt_rtx as trt_ep  # type: ignore

        ort.register_execution_provider_library(trt_ep.get_ep_name(), trt_ep.get_library_path())  # type: ignore
    __preloaded_ort__ = ort  # type: ignore

    return ort  # type: ignore


_TRT_RUNTIME_VERSION_MODULES: tuple[tuple[str, str], ...] = (
    ("trt", "tensorrt"),
    ("ep", "onnxruntime_ep_nv_tensorrt_rtx"),
    ("ort", "onnxruntime"),
)


def _get_trt_runtime_version_tag() -> str:
    """
    Build a cache-key tag out of the TRT-RTX runtime component versions.

    TensorRT compiles engines against a fixed engine serialization version and refuses to
    deserialize anything produced by an incompatible runtime (the
    ``stdVersionRead == kSERIALIZATION_VERSION`` failure). Folding the runtime versions into
    the cache key makes sure a stale engine is rebuilt instead of being reused after an upgrade.

    :return: A filesystem-safe version tag, or ``"unknown"`` if nothing could be detected.
    """

    versions: list[str] = []
    for tag, module_name in _TRT_RUNTIME_VERSION_MODULES:
        if importutil.find_spec(module_name) is None:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module_version = getattr(module, "__version__", None)
        versions.append(f"{tag}{module_version}" if module_version is not None else f"{tag}unknown")
    if not versions:
        return "unknown"
    return "-".join(versions)


def get_torch_memory_limit_and_rtx(device_id: int) -> tuple[int, int] | None:
    try:
        import torch.cuda  # type: ignore

        cu_major, _ = torch.cuda.get_device_capability()
        return torch.cuda.get_device_properties(device_id).total_memory, cu_major
    except (ImportError, AssertionError):
        return None


def _get_nvrtx_compiled_model_path(model_path: Path, data_dir: Path, cache_key: str) -> Path:
    cache_hash = md5(cache_key.encode("utf-8")).hexdigest()
    return data_dir / "rtx_compiled" / f"{model_path.stem}_{cache_hash}_ctx.onnx"


def _compile_nvrtx_model(
    model_path: Path,
    data_dir: Path,
    cache_key: str,
    ep_name: str,
    ep_config: dict[str, str],
) -> Path:
    ort = _get_onnxruntime()

    model_comp = _get_nvrtx_compiled_model_path(model_path, data_dir, cache_key)
    model_comp.parent.mkdir(parents=True, exist_ok=True)
    if model_comp.exists():
        print(f"Found pre-compiled model {model_path.stem} for TRT-RTX engine")
        return model_comp

    sess_opt = ort.SessionOptions()
    sess_opt.add_provider(ep_name, ep_config)
    model_data_path = model_path.with_suffix(".onnx.data")
    # get file size, if larger than 1.5gb, we don't embed ep context (although the recommended is 2gb)
    model_stat = model_path.stat().st_size
    if model_data_path.is_file():
        model_stat += model_data_path.stat().st_size
    is_large_size = model_stat > 1.5 * (1024**3)
    print(f"Model size is {model_stat / (1024**3):.2f}GB, is large size?", is_large_size)
    compiler = ort.ModelCompiler(
        sess_opt,
        model_path,
        embed_compiled_data_into_model=not is_large_size,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    )
    print(f"Pre-compiling model {model_path.stem} for TRT-RTX engine")
    st_time = time.time()
    compiler.compile_to_file(str(model_comp))
    et_time = time.time()
    print(f"Compiled model {model_path.stem} for TRT-RTX engine (in {(et_time - st_time):2f}s)")
    return model_comp


def prepare_model_runtime_builders(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
    with_nvrtx: bool = False,
) -> ort.InferenceSession:
    ort = _get_onnxruntime()
    cache_dir = DATA_DIR / "trt_engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_rtx_dir = DATA_DIR / "trtrtx_engines"
    cache_rtx_dir.mkdir(parents=True, exist_ok=True)

    hashed_path = md5(str(model_path.resolve()).encode("utf-8")).hexdigest()
    runtime_version = _get_trt_runtime_version_tag()
    cache_prefix = f"ptag_{hashed_path}_{runtime_version}"

    torch_info = get_torch_memory_limit_and_rtx(device_id)
    memory_limit = int(torch_info[0] * 0.75) if torch_info else 2 * (1024**3)  # 2GB or 75% of GPU memory
    has_trt_rtx = torch_info[1] >= 8 if torch_info else False
    if with_nvrtx and not has_trt_rtx:
        print("TensorRT RTX is not supported on this GPU. Falling back to TensorRT.")
        with_nvrtx = False
    needs_trt_compat = torch_info[1] >= 12 if torch_info else False

    memory_limit = 2 * (1024**3)
    trt_ep_config = {
        "device_id": device_id,
        "trt_sparsity_enable": True,
        "trt_max_workspace_size": memory_limit,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_engines",
        "trt_engine_cache_prefix": cache_prefix,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
        "trt_build_heuristics_enable": True,
        "trt_builder_optimization_level": 3,
        "trt_context_memory_sharing_enable": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": str(DATA_DIR),
        "trt_detailed_build_log": True if is_verbose else False,
        "trt_engine_hw_compatible": needs_trt_compat,
    }
    trtrtx_ep_config = {
        "device_id": str(device_id),
        "enable_cuda_graph": "1",  # although by default this is already enabled
        "nv_max_workspace_size": str(memory_limit),
        "nv_detailed_build_log": "1" if is_verbose else "0",
        "nv_runtime_cache_path": str(cache_rtx_dir),
    }

    ep_devices = {ep_device.ep_name: ep_device for ep_device in ort.get_ep_devices()}
    has_good_ep_devices = False
    raw_providers = set(ort.get_available_providers())
    print("Available providers:", raw_providers)
    print("Available EP devices:", list(ep_devices.keys()))

    providers = []
    ep_providers = []
    if sys.platform != "darwin":
        has_nvrtx = False
        if "NvTensorRTRTXExecutionProvider" in raw_providers and with_nvrtx:
            providers.append(("NvTensorRTRTXExecutionProvider", trtrtx_ep_config))
            has_nvrtx = True
        elif "nv_tensorrt_rtx" in raw_providers and with_nvrtx:
            providers.append(("nv_tensorrt_rtx", trtrtx_ep_config))
            has_nvrtx = True
        elif "nv_tensorrt_rtx" in ep_devices and with_nvrtx:
            ep_providers.append((ep_devices["nv_tensorrt_rtx"], trtrtx_ep_config))
            has_nvrtx = True
            has_good_ep_devices = True
        if "TensorrtExecutionProvider" in raw_providers and not has_nvrtx:
            providers.append(("TensorrtExecutionProvider", trt_ep_config))
        if "CUDAExecutionProvider" in raw_providers:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": memory_limit,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": True,
                    "prefer_nhwc": True,
                },
            ))
        if not providers and not ep_providers:
            providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo"}))
    else:
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "1",
                    "ModelCacheDirectory": str(cache_dir),
                    "SpecializationStrategy": "FastPrediction",
                },
            ),
        ]

    if not providers and not ep_providers:
        raise RuntimeError(
            "No suitable ONNX Runtime execution providers found. "
            "Ensure you have compatible hardware and the necessary dependencies installed."
        )

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    sess_opt = ort.SessionOptions()
    for ep_devices, ep_config in ep_providers:
        sess_opt.add_provider_for_devices([ep_devices], ep_config)

    selected_model_path = model_path
    for ep_name, ep_config in providers:
        if ep_name in ["nv_tensorrt_rtx", "NvTensorRTRTXExecutionProvider"]:
            selected_model_path = _compile_nvrtx_model(model_path, DATA_DIR, cache_prefix, ep_name, ep_config)
            break
    if has_good_ep_devices:
        providers = None  # Use EP devices for NvRTX

    session = ort.InferenceSession(selected_model_path, sess_options=sess_opt, providers=providers, enable_fallback=0)
    print("ONNX active providers:", session.get_providers())
    return session
