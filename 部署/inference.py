"""
Swin-B FP32 Frame-Level Inference Benchmark
============================================

功能：
1. 加载   微调模型
2. 对单张图片进行 FP32 推理
3. 模型预热
4. 连续测试 350 帧
5. 使用 torch.cuda.Event 测量 GPU inference latency
6. 统计 Mean / Min / P50 / P90 / P95 / P99 / Max latency
7. 计算 FPS、GPU 显存、温度、频率、利用率、功耗
8. 分析：
   - 7-frame latency trend
   - high-frequency jitter
   - system jitter
   - cumulative drift
   - thermal component
   - DVFS event
   - latency spike
   - burst
9. 保存逐帧 CSV
10. 保存完整终端日志

------------------------------------------------------------
推荐 Python 环境
------------------------------------------------------------

建议：

    Python >= 3.10
    PyTorch >= 2.x
    CUDA-compatible PyTorch
    torchvision
    numpy
    Pillow
    PyYAML
    yacs

例如：

    pip install numpy pillow pyyaml yacs

PyTorch / torchvision 推荐根据 NVIDIA CUDA 环境安装官方 CUDA 版本。

模型工程本身还需要：

    config.py
    models.py
    configs/
    training_outputs/

与本脚本位于正确的项目目录结构中。

------------------------------------------------------------
可选 CUDA / PyTorch 加速
------------------------------------------------------------

代码已经启用：

    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

注意：

TF32 仍然使用 FP32 Tensor 输入和模型参数，
但在支持 TF32 的 NVIDIA GPU 上，
部分矩阵乘法可能使用 Tensor Core 加速。

如果论文实验要求“严格 IEEE FP32”，
可以关闭 TF32：

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

------------------------------------------------------------
运行
------------------------------------------------------------

确保：

    test.jpg

位于项目根目录。

然后执行：

    python your_script_name.py

程序将生成：

    SwinB_Subject08_Frame_Latency.csv

以及：

    inference_logs/
        SwinB_Subject08_Inference_YYYYMMDD_HHMMSS.log

------------------------------------------------------------
建议 benchmark 条件
------------------------------------------------------------

正式性能测试前建议：

1. 关闭其他 GPU 程序
2. 保证 GPU 已完成驱动初始化
3. 使用固定输入尺寸
4. 固定 batch size
5. 执行充分 warm-up
6. 连续运行多次实验
7. 保持相同 CUDA / PyTorch / driver 环境
8. 如果研究 GPU DVFS，建议同时记录：
       temperature
       SM clock
       utilization
       power

如果只追求最纯粹 inference latency，
可以设置：

    ENABLE_GPU_MONITORING = False

以减少 nvidia-smi 查询对运行环境的扰动。
"""

import csv
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import get_config
from models import build_model


# ==========================================================
# 路径
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CFG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "swin_base__800ep"
    / "simmim_finetune__swin_base__img224_window7__800ep.yaml"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "swin_subject08"
    / "simmim_finetune"
    / "fatiguev2_subject08_reglight_100ep"
    / "best_ba_weights.pth"
)

IMAGE_PATH = PROJECT_ROOT / "test.jpg"

SAVE_CSV = (
    PROJECT_ROOT
    / "SwinB_Subject08_Frame_Latency.csv"
)

LOG_DIR = (
    PROJECT_ROOT
    / "inference_logs"
)


# ==========================================================
# 测试参数
# ==========================================================

DEVICE_ID = 0

IMAGE_SIZE = 224
BATCH_SIZE = 1

# 正式统计帧数
FRAME_NUM = 350

# 模型预热次数
WARMUP_NUM = 50

# 是否采集 GPU 状态
ENABLE_GPU_MONITORING = True

# 每隔多少帧采样一次 GPU 状态
GPU_STATUS_INTERVAL = 10

CLASS_NAMES = [
    "fatigue",
    "nofatigue",
]


# ==========================================================
# PyTorch / CUDA 性能选项
# ==========================================================

# True:
# 对固定输入尺寸的网络，cuDNN 会搜索较优 kernel。
ENABLE_CUDNN_BENCHMARK = True

# True:
# 支持 TF32 的 GPU 上允许矩阵计算使用 TF32。
ENABLE_TF32 = True

# high:
# PyTorch 对 float32 matmul 使用较高性能策略。
MATMUL_PRECISION = "high"


# ==========================================================
# SimMIM 参数
# ==========================================================

class Args:
    cfg = str(CFG_PATH)
    opts = None

    batch_size = BATCH_SIZE
    data_path = ""

    pretrained = None
    resume = None

    accumulation_steps = None
    use_checkpoint = False

    amp_opt_level = "O0"

    output = ""
    tag = ""

    eval = True
    throughput = False

    local_rank = DEVICE_ID


args = Args()


# ==========================================================
# 日志
# ==========================================================

class TeeLogger:
    """
    将 stdout / stderr 同时写入：
    1. 终端
    2. 日志文件
    """

    def __init__(
        self,
        terminal,
        log_file,
    ):
        self.terminal = terminal
        self.log_file = log_file

    def write(
        self,
        message,
    ):
        self.terminal.write(
            message
        )

        self.log_file.write(
            message
        )

        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


def create_log_path():
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        LOG_DIR
        / f"SwinB_Subject08_Inference_{timestamp}.log"
    )


# ==========================================================
# GPU 状态
# ==========================================================

def get_gpu_status(
    device_id=0,
):
    """
    使用 nvidia-smi 获取当前 GPU 状态。

    返回：
        GPU Temperature (C)
        GPU SM Clock (MHz)
        GPU Utilization (%)
        GPU Power (W)

    注意：
        nvidia-smi 属于外部系统查询。
        如果追求极致纯净 benchmark，
        可以关闭 ENABLE_GPU_MONITORING。
    """

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu="
                "temperature.gpu,"
                "clocks.sm,"
                "utilization.gpu,"
                "power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )

        line = (
            result.decode(
                "utf-8"
            )
            .strip()
            .splitlines()[0]
        )

        values = [
            value.strip()
            for value in line.split(",")
        ]

        if len(values) != 4:
            raise ValueError(
                "Unexpected nvidia-smi output: "
                f"{line}"
            )

        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
        )

    except (
        subprocess.SubprocessError,
        ValueError,
        IndexError,
        OSError,
    ) as error:

        print(
            "Warning: failed to read GPU status:",
            repr(error),
        )

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


# ==========================================================
# 检查点处理
# ==========================================================

def extract_state_dict(
    checkpoint,
):
    """
    从不同 checkpoint 格式中提取 state_dict。
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        return checkpoint

    possible_keys = [
        "model",
        "state_dict",
        "model_state_dict",
        "net",
        "network",
        "weights",
    ]

    for key in possible_keys:
        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            print(
                f"Using checkpoint key: {key}"
            )

            return value

    tensor_count = sum(
        isinstance(
            value,
            torch.Tensor,
        )
        for value in checkpoint.values()
    )

    if tensor_count > 0:
        print(
            "Checkpoint is a raw state_dict."
        )

        return checkpoint

    raise ValueError(
        "Unable to locate model weights. "
        "Available checkpoint keys: "
        f"{list(checkpoint.keys())}"
    )


def clean_state_dict(
    state_dict,
):
    """
    清除常见 checkpoint key 前缀。
    """

    cleaned_state_dict = {}

    removable_prefixes = [
        "module.",
        "model.",
    ]

    for (
        original_key,
        value,
    ) in state_dict.items():

        key = original_key
        changed = True

        while changed:
            changed = False

            for prefix in removable_prefixes:
                if key.startswith(
                    prefix
                ):
                    key = key[
                        len(prefix):
                    ]

                    changed = True

        cleaned_state_dict[
            key
        ] = value

    return cleaned_state_dict


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        "\nCheckpoint loading result:"
    )

    print(
        load_result
    )

    if load_result.missing_keys:
        print(
            "Missing keys:",
            len(
                load_result.missing_keys
            ),
        )

        for key in (
            load_result.missing_keys[:20]
        ):
            print(
                "  ",
                key,
            )

    if load_result.unexpected_keys:
        print(
            "Unexpected keys:",
            len(
                load_result.unexpected_keys
            ),
        )

        for key in (
            load_result.unexpected_keys[:20]
        ):
            print(
                "  ",
                key,
            )

    if (
        not load_result.missing_keys
        and not load_result.unexpected_keys
    ):
        print(
            "All model parameters "
            "loaded successfully."
        )


# ==========================================================
# 模型输出处理
# ==========================================================

def extract_model_output(
    output,
):
    """
    兼容：
        Tensor
        tuple/list
        dict
    类型模型输出。
    """

    if isinstance(
        output,
        (tuple, list),
    ):
        if len(output) == 0:
            raise ValueError(
                "Model returned "
                "an empty tuple/list."
            )

        output = output[0]

    if isinstance(
        output,
        dict,
    ):
        possible_keys = [
            "logits",
            "output",
            "pred",
            "prediction",
        ]

        for key in possible_keys:
            if key in output:
                output = output[
                    key
                ]

                break

        else:
            raise ValueError(
                "Unsupported dictionary "
                "model output. "
                "Available keys: "
                f"{list(output.keys())}"
            )

    if not isinstance(
        output,
        torch.Tensor,
    ):
        raise TypeError(
            "Model output must be "
            "a Tensor, "
            f"but received {type(output)}"
        )

    return output


# ==========================================================
# 数值工具
# ==========================================================

def moving_average_reflect(
    values,
    window_size=7,
):
    """
    Reflect-padding moving average。

    相比简单 padding，
    边界区域通常更加平滑。
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if window_size <= 1:
        return values.copy()

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be odd."
        )

    if len(values) < window_size:
        return np.full_like(
            values,
            np.mean(values),
        )

    padding = (
        window_size // 2
    )

    padded_values = np.pad(
        values,
        pad_width=padding,
        mode="reflect",
    )

    kernel = (
        np.ones(
            window_size,
            dtype=np.float64,
        )
        / window_size
    )

    return np.convolve(
        padded_values,
        kernel,
        mode="valid",
    )


# ==========================================================
# 模型预热
# ==========================================================

def warm_up_model(
    model,
    image,
    device,
    iterations,
):
    """
    模型预热。

    目的：
    1. CUDA context 初始化
    2. cuDNN kernel selection
    3. cache 建立
    4. 减少首次推理异常值
    """

    print(
        f"\nWarming up for "
        f"{iterations} iterations..."
    )

    with torch.inference_mode():

        for index in range(
            iterations
        ):
            _ = model(
                image
            )

            if index == 0:
                torch.cuda.synchronize(
                    device
                )

                print(
                    "First inference completed."
                )

    torch.cuda.synchronize(
        device
    )

    print(
        "Warm-up completed."
    )


# ==========================================================
# 正式测试
# ==========================================================

def benchmark_model(
    model,
    image,
    device,
):
    """
    使用 CUDA Event 测量 GPU inference latency。

    注意：

    latency 直接使用：

        start_event.elapsed_time(end_event)

    单位为 ms。

    本版本不执行任何额外缩放。
    """

    frame_latency = np.zeros(
        FRAME_NUM,
        dtype=np.float64,
    )

    temperatures = np.zeros(
        FRAME_NUM,
        dtype=np.float64,
    )

    clocks = np.zeros(
        FRAME_NUM,
        dtype=np.float64,
    )

    utilizations = np.zeros(
        FRAME_NUM,
        dtype=np.float64,
    )

    powers = np.zeros(
        FRAME_NUM,
        dtype=np.float64,
    )

    # ------------------------------------------------------
    # 每帧 CUDA Event
    # ------------------------------------------------------

    start_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            FRAME_NUM
        )
    ]

    end_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            FRAME_NUM
        )
    ]

    # ------------------------------------------------------
    # 整段 CUDA Event
    # ------------------------------------------------------

    total_start_event = (
        torch.cuda.Event(
            enable_timing=True
        )
    )

    total_end_event = (
        torch.cuda.Event(
            enable_timing=True
        )
    )

    # ------------------------------------------------------
    # GPU 初始状态
    # ------------------------------------------------------

    if ENABLE_GPU_MONITORING:

        (
            current_temperature,
            current_clock,
            current_utilization,
            current_power,
        ) = get_gpu_status(
            DEVICE_ID
        )

    else:

        current_temperature = 0.0
        current_clock = 0.0
        current_utilization = 0.0
        current_power = 0.0

    # 首次读取失败时使用 0
    current_temperature = float(
        np.nan_to_num(
            current_temperature,
            nan=0.0,
        )
    )

    current_clock = float(
        np.nan_to_num(
            current_clock,
            nan=0.0,
        )
    )

    current_utilization = float(
        np.nan_to_num(
            current_utilization,
            nan=0.0,
        )
    )

    current_power = float(
        np.nan_to_num(
            current_power,
            nan=0.0,
        )
    )

    last_output = None

    # ------------------------------------------------------
    # 清空 peak memory counter
    # ------------------------------------------------------

    torch.cuda.reset_peak_memory_stats(
        device
    )

    torch.cuda.synchronize(
        device
    )

    print(
        f"\nStarting benchmark for "
        f"{FRAME_NUM} frames..."
    )

    # ------------------------------------------------------
    # 正式 benchmark
    # ------------------------------------------------------

    with torch.inference_mode():

        total_start_event.record()

        for frame_index in range(
            FRAME_NUM
        ):

            # ------------------------------------------------
            # GPU 状态采样
            # ------------------------------------------------

            if (
                ENABLE_GPU_MONITORING
                and frame_index > 0
                and frame_index
                % GPU_STATUS_INTERVAL
                == 0
            ):

                (
                    sampled_temperature,
                    sampled_clock,
                    sampled_utilization,
                    sampled_power,
                ) = get_gpu_status(
                    DEVICE_ID
                )

                # 读取失败：
                # 沿用上一有效值。

                if not np.isnan(
                    sampled_temperature
                ):
                    current_temperature = (
                        sampled_temperature
                    )

                if not np.isnan(
                    sampled_clock
                ):
                    current_clock = (
                        sampled_clock
                    )

                if not np.isnan(
                    sampled_utilization
                ):
                    current_utilization = (
                        sampled_utilization
                    )

                if not np.isnan(
                    sampled_power
                ):
                    current_power = (
                        sampled_power
                    )

            temperatures[
                frame_index
            ] = current_temperature

            clocks[
                frame_index
            ] = current_clock

            utilizations[
                frame_index
            ] = current_utilization

            powers[
                frame_index
            ] = current_power

            # ------------------------------------------------
            # CUDA latency measurement
            # ------------------------------------------------

            start_events[
                frame_index
            ].record()

            last_output = model(
                image
            )

            end_events[
                frame_index
            ].record()

        total_end_event.record()

    # ------------------------------------------------------
    # 等待所有 CUDA kernel 完成
    # ------------------------------------------------------

    torch.cuda.synchronize(
        device
    )

    # ------------------------------------------------------
    # 提取逐帧 GPU 时间
    # ------------------------------------------------------

    for frame_index in range(
        FRAME_NUM
    ):

        frame_latency[
            frame_index
        ] = start_events[
            frame_index
        ].elapsed_time(
            end_events[
                frame_index
            ]
        )

    # ------------------------------------------------------
    # 连续整段 GPU 时间
    # ------------------------------------------------------

    total_gpu_time_ms = (
        total_start_event.elapsed_time(
            total_end_event
        )
    )

    # ------------------------------------------------------
    # 关键：
    #
    # 不做任何除法 / 人工缩放。
    #
    # frame_latency 和 total_gpu_time_ms
    # 均保留 CUDA Event 原始测量结果。
    # ------------------------------------------------------

    print(
        "Benchmark completed."
    )

    return {
        "latency": frame_latency,
        "total_gpu_time_ms": (
            total_gpu_time_ms
        ),
        "temperatures": temperatures,
        "clocks": clocks,
        "utilizations": utilizations,
        "powers": powers,
        "last_output": last_output,
    }


# ==========================================================
# 延迟分析
# ==========================================================

def analyze_latency(
    latency,
    temperatures,
    clocks,
):
    """
    对逐帧 latency 做简单分解和异常检测。

    注意：
    这些量属于统计/分析量，
    并不意味着严格物理因果分解。
    """

    # ------------------------------------------------------
    # 7-frame 平滑趋势
    # ------------------------------------------------------

    trend = moving_average_reflect(
        latency,
        window_size=7,
    )

    # ------------------------------------------------------
    # 高频 jitter
    # ------------------------------------------------------

    high_frequency_jitter = (
        latency - trend
    )

    # ------------------------------------------------------
    # 去均值后的 system jitter
    # ------------------------------------------------------

    system_jitter = (
        high_frequency_jitter
        - np.mean(
            high_frequency_jitter
        )
    )

    # ------------------------------------------------------
    # 累积趋势变化
    # ------------------------------------------------------

    cumulative_drift = np.cumsum(
        trend - trend[0]
    )

    # ------------------------------------------------------
    # GPU 温度变化
    # ------------------------------------------------------

    thermal_component = (
        temperatures
        - temperatures[0]
    )

    thermal_component = np.nan_to_num(
        thermal_component,
        nan=0.0,
    )

    # ------------------------------------------------------
    # 中心化 trend
    # ------------------------------------------------------

    centered_trend_component = (
        trend
        - np.mean(
            trend
        )
    )

    # ------------------------------------------------------
    # DVFS 检测
    #
    # 如果 SM clock 相对初始 clock
    # 变化超过 50 MHz，
    # 标记为 DVFS event。
    # ------------------------------------------------------

    base_clock = clocks[0]

    dvfs_event = np.zeros(
        len(latency),
        dtype=np.int32,
    )

    dvfs_effect = np.zeros(
        len(latency),
        dtype=np.float64,
    )

    if base_clock > 0:

        for (
            index,
            clock,
        ) in enumerate(
            clocks
        ):

            if abs(
                clock
                - base_clock
            ) > 50:

                dvfs_event[
                    index
                ] = 1

                dvfs_effect[
                    index
                ] = (
                    latency[index]
                    - trend[index]
                )

    # ------------------------------------------------------
    # residual
    # ------------------------------------------------------

    residual = (
        latency - trend
    )

    residual_mean = float(
        np.mean(
            residual
        )
    )

    residual_std = float(
        np.std(
            residual
        )
    )

    # ------------------------------------------------------
    # Spike detection
    #
    # residual > mean + 3 sigma
    # ------------------------------------------------------

    spike_event = np.zeros(
        len(latency),
        dtype=np.int32,
    )

    spike_effect = np.zeros(
        len(latency),
        dtype=np.float64,
    )

    if residual_std > 0:

        spike_threshold = (
            residual_mean
            + 3.0
            * residual_std
        )

        spike_mask = (
            residual
            > spike_threshold
        )

        spike_event[
            spike_mask
        ] = 1

        spike_effect[
            spike_mask
        ] = residual[
            spike_mask
        ]

    else:

        spike_threshold = (
            residual_mean
        )

    # ------------------------------------------------------
    # Burst detection
    #
    # 连续 >= 3 个 spike
    # 视为 burst。
    # ------------------------------------------------------

    burst_effect = np.zeros(
        len(latency),
        dtype=np.float64,
    )

    burst_start = None

    for index in range(
        len(latency)
    ):

        if spike_event[
            index
        ] == 1:

            if burst_start is None:
                burst_start = index

        elif burst_start is not None:

            burst_length = (
                index
                - burst_start
            )

            if burst_length >= 3:

                burst_effect[
                    burst_start:index
                ] = residual[
                    burst_start:index
                ]

            burst_start = None

    # 最后一段 burst
    if burst_start is not None:

        burst_length = (
            len(latency)
            - burst_start
        )

        if burst_length >= 3:

            burst_effect[
                burst_start:
            ] = residual[
                burst_start:
            ]

    return {
        "trend": trend,

        "high_frequency_jitter": (
            high_frequency_jitter
        ),

        "system_jitter": (
            system_jitter
        ),

        "cumulative_drift": (
            cumulative_drift
        ),

        "thermal_component": (
            thermal_component
        ),

        "centered_trend_component": (
            centered_trend_component
        ),

        "dvfs_event": (
            dvfs_event
        ),

        "dvfs_effect": (
            dvfs_effect
        ),

        "residual": (
            residual
        ),

        "spike_threshold": (
            spike_threshold
        ),

        "spike_event": (
            spike_event
        ),

        "spike_effect": (
            spike_effect
        ),

        "burst_effect": (
            burst_effect
        ),
    }


# ==========================================================
# 保存 CSV
# ==========================================================

def save_csv(
    save_path,
    benchmark_result,
    analysis_result,
):
    latency = benchmark_result[
        "latency"
    ]

    temperatures = benchmark_result[
        "temperatures"
    ]

    clocks = benchmark_result[
        "clocks"
    ]

    utilizations = benchmark_result[
        "utilizations"
    ]

    powers = benchmark_result[
        "powers"
    ]

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with save_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "Frame Index",

                "Latency (ms)",

                "7-Frame Trend (ms)",

                "High-Frequency Jitter (ms)",

                "System Jitter (ms)",

                "Cumulative Drift (ms)",

                "Temperature Change (C)",

                "Centered Trend Component (ms)",

                "GPU Temperature (C)",

                "GPU SM Clock (MHz)",

                "GPU Utilization (%)",

                "GPU Power (W)",

                "DVFS Event",

                "DVFS Effect (ms)",

                "Spike Event",

                "Spike Effect (ms)",

                "Burst Effect (ms)",
            ]
        )

        for index in range(
            len(latency)
        ):

            writer.writerow(
                [
                    index,

                    latency[
                        index
                    ],

                    analysis_result[
                        "trend"
                    ][index],

                    analysis_result[
                        "high_frequency_jitter"
                    ][index],

                    analysis_result[
                        "system_jitter"
                    ][index],

                    analysis_result[
                        "cumulative_drift"
                    ][index],

                    analysis_result[
                        "thermal_component"
                    ][index],

                    analysis_result[
                        "centered_trend_component"
                    ][index],

                    temperatures[
                        index
                    ],

                    clocks[
                        index
                    ],

                    utilizations[
                        index
                    ],

                    powers[
                        index
                    ],

                    analysis_result[
                        "dvfs_event"
                    ][index],

                    analysis_result[
                        "dvfs_effect"
                    ][index],

                    analysis_result[
                        "spike_event"
                    ][index],

                    analysis_result[
                        "spike_effect"
                    ][index],

                    analysis_result[
                        "burst_effect"
                    ][index],
                ]
            )


# ==========================================================
# 推理主体
# ==========================================================

def run_inference():

    # ------------------------------------------------------
    # CUDA 检查
    # ------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    torch.cuda.set_device(
        DEVICE_ID
    )

    device = torch.device(
        f"cuda:{DEVICE_ID}"
    )

    # ------------------------------------------------------
    # CUDA / cuDNN 性能设置
    # ------------------------------------------------------

    torch.backends.cudnn.benchmark = (
        ENABLE_CUDNN_BENCHMARK
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        ENABLE_TF32
    )

    torch.backends.cudnn.allow_tf32 = (
        ENABLE_TF32
    )

    try:
        torch.set_float32_matmul_precision(
            MATMUL_PRECISION
        )

    except Exception:
        pass

    # ------------------------------------------------------
    # GPU 信息
    # ------------------------------------------------------

    device_name = (
        torch.cuda.get_device_name(
            DEVICE_ID
        )
    )

    device_properties = (
        torch.cuda.get_device_properties(
            DEVICE_ID
        )
    )

    total_memory_gb = (
        device_properties.total_memory
        / 1024
        / 1024
        / 1024
    )

    # ------------------------------------------------------
    # 环境输出
    # ------------------------------------------------------

    print(
        "=" * 76
    )

    print(
        "Swin-B FP32 Frame-Level "
        "Inference Benchmark"
    )

    print(
        "=" * 76
    )

    print(
        "Run time:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "GPU device:",
        device_name,
    )

    print(
        "CUDA device ID:",
        DEVICE_ID,
    )

    print(
        "GPU memory:",
        f"{total_memory_gb:.2f} GB",
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA runtime:",
        torch.version.cuda,
    )

    print(
        "cuDNN:",
        torch.backends.cudnn.version(),
    )

    print(
        "Tensor dtype:",
        "torch.float32",
    )

    print(
        "TF32 allowed:",
        ENABLE_TF32,
    )

    print(
        "cuDNN benchmark:",
        ENABLE_CUDNN_BENCHMARK,
    )

    print(
        "Float32 matmul precision:",
        MATMUL_PRECISION,
    )

    print(
        "Config:",
        CFG_PATH,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Image:",
        IMAGE_PATH,
    )

    print(
        "Input shape:",
        (
            BATCH_SIZE,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    )

    print(
        "Frames:",
        FRAME_NUM,
    )

    print(
        "Warm-up:",
        WARMUP_NUM,
    )

    print(
        "Latency scaling:",
        "None",
    )

    print(
        "Latency unit:",
        "Raw CUDA Event milliseconds",
    )

    print(
        "GPU monitoring:",
        ENABLE_GPU_MONITORING,
    )

    print(
        "GPU status interval:",
        GPU_STATUS_INTERVAL,
    )

    print(
        "=" * 76
    )

    # ------------------------------------------------------
    # 文件检查
    # ------------------------------------------------------

    if not CFG_PATH.is_file():

        raise FileNotFoundError(
            "Config does not exist:\n"
            f"{CFG_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{CHECKPOINT_PATH}"
        )

    if not IMAGE_PATH.is_file():

        raise FileNotFoundError(
            "Image does not exist:\n"
            f"{IMAGE_PATH}"
        )

    # ======================================================
    # 构建模型
    # ======================================================

    print(
        "\nBuilding model..."
    )

    config = get_config(
        args
    )

    model = build_model(
        config,
        is_pretrain=False,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    # ======================================================
    # 加载权重
    # ======================================================

    load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    # 确保权重为 FP32
    model = model.float()

    model.eval()

    print(
        "\nModel prepared in FP32."
    )

    # ======================================================
    # 图像预处理
    # ======================================================

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=(
                    transforms
                    .InterpolationMode
                    .BICUBIC
                ),
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    with Image.open(
        IMAGE_PATH
    ) as pil_image:

        image = pil_image.convert(
            "RGB"
        )

        image = transform(
            image
        )

    image = image.unsqueeze(
        0
    )

    image = image.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).contiguous()

    print(
        "Input tensor:",
        tuple(
            image.shape
        ),
    )

    print(
        "Input dtype:",
        image.dtype,
    )

    # ======================================================
    # 预热
    # ======================================================

    warm_up_model(
        model=model,
        image=image,
        device=device,
        iterations=WARMUP_NUM,
    )

    # ======================================================
    # 正式测试
    # ======================================================

    benchmark_result = benchmark_model(
        model=model,
        image=image,
        device=device,
    )

    latency = benchmark_result[
        "latency"
    ]

    # ======================================================
    # 延迟分析
    # ======================================================

    analysis_result = analyze_latency(
        latency=latency,

        temperatures=benchmark_result[
            "temperatures"
        ],

        clocks=benchmark_result[
            "clocks"
        ],
    )

    # ======================================================
    # 预测结果
    # ======================================================

    output = extract_model_output(
        benchmark_result[
            "last_output"
        ]
    )

    if output.ndim != 2:

        raise ValueError(
            "Expected logits shape "
            "[batch, classes], "
            f"received {tuple(output.shape)}"
        )

    if output.shape[1] != len(
        CLASS_NAMES
    ):

        raise ValueError(
            f"Model has {output.shape[1]} outputs, "
            "but CLASS_NAMES contains "
            f"{len(CLASS_NAMES)} names."
        )

    probabilities = torch.softmax(
        output.float(),
        dim=1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=1,
    ).item()

    confidence = probabilities[
        0,
        prediction,
    ].item()

    # ======================================================
    # Latency 统计
    # ======================================================

    mean_latency = float(
        np.mean(
            latency
        )
    )

    minimum_latency = float(
        np.min(
            latency
        )
    )

    p50_latency = float(
        np.percentile(
            latency,
            50,
        )
    )

    p90_latency = float(
        np.percentile(
            latency,
            90,
        )
    )

    p95_latency = float(
        np.percentile(
            latency,
            95,
        )
    )

    p99_latency = float(
        np.percentile(
            latency,
            99,
        )
    )

    maximum_latency = float(
        np.max(
            latency
        )
    )

    latency_std = float(
        np.std(
            latency
        )
    )

    latency_cv = (
        latency_std
        / mean_latency

        if mean_latency > 0

        else np.nan
    )

    # ======================================================
    # FPS
    # ======================================================

    theoretical_fps = (
        1000.0
        / mean_latency

        if mean_latency > 0

        else float(
            "inf"
        )
    )

    total_gpu_time_ms = (
        benchmark_result[
            "total_gpu_time_ms"
        ]
    )

    continuous_fps = (
        FRAME_NUM
        * 1000.0
        / total_gpu_time_ms

        if total_gpu_time_ms > 0

        else float(
            "inf"
        )
    )

    # ======================================================
    # GPU memory
    # ======================================================

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024
        / 1024
    )

    peak_reserved_mb = (
        torch.cuda.max_memory_reserved(
            device
        )
        / 1024
        / 1024
    )

    # ======================================================
    # GPU 状态
    # ======================================================

    temperatures = benchmark_result[
        "temperatures"
    ]

    clocks = benchmark_result[
        "clocks"
    ]

    utilizations = benchmark_result[
        "utilizations"
    ]

    powers = benchmark_result[
        "powers"
    ]

    mean_temperature = float(
        np.mean(
            temperatures
        )
    )

    maximum_temperature = float(
        np.max(
            temperatures
        )
    )

    mean_clock = float(
        np.mean(
            clocks
        )
    )

    minimum_clock = float(
        np.min(
            clocks
        )
    )

    maximum_clock = float(
        np.max(
            clocks
        )
    )

    mean_utilization = float(
        np.mean(
            utilizations
        )
    )

    maximum_utilization = float(
        np.max(
            utilizations
        )
    )

    mean_power = float(
        np.mean(
            powers
        )
    )

    maximum_power = float(
        np.max(
            powers
        )
    )

    # ======================================================
    # 保存 CSV
    # ======================================================

    save_csv(
        save_path=SAVE_CSV,
        benchmark_result=benchmark_result,
        analysis_result=analysis_result,
    )

    # ======================================================
    # 最终报告
    # ======================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "Prediction Result"
    )

    print(
        "=" * 76
    )

    print(
        "Prediction:",
        CLASS_NAMES[
            prediction
        ],
    )

    print(
        "Class ID:",
        prediction,
    )

    print(
        "Confidence:",
        f"{confidence:.6f}",
    )

    print(
        "\nClass probabilities:"
    )

    for (
        index,
        class_name,
    ) in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_name}: "
            f"{probabilities[0, index].item():.6f}"
        )

    # ======================================================
    # Performance
    # ======================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "Performance"
    )

    print(
        "=" * 76
    )

    print(
        "Latency source:",
        "Raw CUDA Event timing",
    )

    print(
        "Latency scaling:",
        "None",
    )

    print(
        "Mean frame latency:",
        f"{mean_latency:.4f} ms",
    )

    print(
        "Minimum latency:",
        f"{minimum_latency:.4f} ms",
    )

    print(
        "P50 latency:",
        f"{p50_latency:.4f} ms",
    )

    print(
        "P90 latency:",
        f"{p90_latency:.4f} ms",
    )

    print(
        "P95 latency:",
        f"{p95_latency:.4f} ms",
    )

    print(
        "P99 latency:",
        f"{p99_latency:.4f} ms",
    )

    print(
        "Maximum latency:",
        f"{maximum_latency:.4f} ms",
    )

    print(
        "Latency standard deviation:",
        f"{latency_std:.4f} ms",
    )

    print(
        "Latency coefficient of variation:",
        f"{latency_cv:.6f}",
    )

    print(
        "Theoretical FPS:",
        f"{theoretical_fps:.2f} FPS",
    )

    print(
        "Continuous GPU time:",
        f"{total_gpu_time_ms:.4f} ms",
    )

    print(
        "Continuous throughput:",
        f"{continuous_fps:.2f} FPS",
    )

    # ======================================================
    # Memory
    # ======================================================

    print(
        "Peak allocated GPU memory:",
        f"{peak_allocated_mb:.2f} MB",
    )

    print(
        "Peak reserved GPU memory:",
        f"{peak_reserved_mb:.2f} MB",
    )

    # ======================================================
    # GPU Status
    # ======================================================

    print(
        "Average GPU temperature:",
        f"{mean_temperature:.2f} C",
    )

    print(
        "Maximum GPU temperature:",
        f"{maximum_temperature:.2f} C",
    )

    print(
        "Average GPU SM clock:",
        f"{mean_clock:.0f} MHz",
    )

    print(
        "GPU SM clock range:",
        f"{minimum_clock:.0f}-"
        f"{maximum_clock:.0f} MHz",
    )

    print(
        "Average GPU utilization:",
        f"{mean_utilization:.2f} %",
    )

    print(
        "Maximum GPU utilization:",
        f"{maximum_utilization:.2f} %",
    )

    print(
        "Average GPU power:",
        f"{mean_power:.2f} W",
    )

    print(
        "Maximum GPU power:",
        f"{maximum_power:.2f} W",
    )

    # ======================================================
    # Jitter / DVFS / Spike
    # ======================================================

    print(
        "DVFS event frames:",
        int(
            np.sum(
                analysis_result[
                    "dvfs_event"
                ]
            )
        ),
    )

    print(
        "Spike threshold:",
        f"{analysis_result['spike_threshold']:.6f} ms",
    )

    print(
        "Spike frames:",
        int(
            np.sum(
                analysis_result[
                    "spike_event"
                ]
            )
        ),
    )

    print(
        "Burst frames:",
        int(
            np.count_nonzero(
                analysis_result[
                    "burst_effect"
                ]
            )
        ),
    )

    print(
        "-" * 76
    )

    print(
        "CSV saved:",
        SAVE_CSV,
    )

    print(
        "=" * 76
    )


# ==========================================================
# 日志入口
# ==========================================================

def main():

    log_path = create_log_path()

    log_file = log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    )

    original_stdout = (
        sys.stdout
    )

    original_stderr = (
        sys.stderr
    )

    sys.stdout = TeeLogger(
        original_stdout,
        log_file,
    )

    sys.stderr = TeeLogger(
        original_stderr,
        log_file,
    )

    try:

        print(
            "#" * 76
        )

        print(
            "Inference run started:",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        print(
            "Log file:",
            log_path,
        )

        print(
            "#" * 76
        )

        run_inference()

        print(
            "\nInference run "
            "completed successfully."
        )

    except Exception:

        print(
            "\nInference run failed."
        )

        traceback.print_exc()

        raise

    finally:

        print(
            "\nRun finished:",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        print(
            "Log saved:",
            log_path,
        )

        print(
            "#" * 76
        )

        sys.stdout = (
            original_stdout
        )

        sys.stderr = (
            original_stderr
        )

        log_file.close()

        print(
            "Log saved:",
            log_path,
        )


if __name__ == "__main__":
    main()
