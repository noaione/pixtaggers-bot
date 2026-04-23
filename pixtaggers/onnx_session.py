import sys
from hashlib import md5
from pathlib import Path

import onnxruntime as ort

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "engine-data"


def prepare_model_runtime_builders(
    model_path: Path,
    *,
    device_id: int = 0,
    is_verbose: bool = False,
) -> ort.InferenceSession:
    cache_dir = DATA_DIR / "trt_engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_rtx_dir = DATA_DIR / "trtrtx_engines"
    cache_rtx_dir.mkdir(parents=True, exist_ok=True)

    hashed_path = md5(str(model_path.resolve()).encode("utf-8")).hexdigest()
    cache_prefix = f"nmodel_{hashed_path}"

    memory_limit = 2 * (1024**3)
    trt_ep_config = {
        "device_id": device_id,
        "trt_sparsity_enable": True,
        "trt_max_workspace_size": memory_limit,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_engines",
        "trt_engine_cache_prefix": cache_prefix,
        "trt_build_heuristics_enable": True,
        "trt_builder_optimization_level": 3,
        "trt_context_memory_sharing_enable": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": str(DATA_DIR),
        "trt_detailed_build_log": True if is_verbose else False,
    }
    # trtrtx_ep_config = {
    #     "device_id": device_id,
    #     "nv_max_workspace_size": memory_limit,
    #     "nv_detailed_build_log": True if is_verbose else False,
    #     "nv_runtime_cache_path": str(cache_rtx_dir),
    # }

    if sys.platform != "darwin":
        providers = [
            # ("NvTensorRTRTXExecutionProvider", trtrtx_ep_config),
            ("TensorrtExecutionProvider", trt_ep_config),
            (
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
            ),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo"}),
        ]
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

    verb_level = 0 if is_verbose else 3
    ort.set_default_logger_severity(verb_level)
    ort.set_default_logger_verbosity(verb_level)

    sess_opt = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    return session
