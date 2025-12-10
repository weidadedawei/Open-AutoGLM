# Open-AutoGLM (Mac MLX 本地版) 技术手册

本文档专为 **纯本地化 Open-AutoGLM** 系统编写。本系统基于 Apple MLX 深度学习框架，在 macOS 上流畅运行 **zai-org/AutoGLM-Phone-9B** 端侧大模型，实现**完全离线、隐私安全、零云端依赖**的智能手机代理。

技术底座源自 [Open-AutoGLM 官方项目](https://github.com/zai-org/Open-AutoGLM)。

---

## 目录

- [1. 核心技术原理](#1-核心技术原理)
- [2. 系统架构](#2-系统架构)
- [3. 部署方案对比](#3-部署方案对比)
- [4. 本地使用指南](#4-本地使用指南)
- [5. 性能优化技术详解](#5-性能优化技术详解)
- [6. 调试与日志](#6-调试与日志)
- [7. API 参考](#7-api-参考)
- [8. 故障排查指南](#8-故障排查指南)
- [9. 能力边界](#9-能力边界)
- [10. 项目引用](#10-项目引用)

---

## 1. 核心技术原理

系统模仿人类"认知-决策-行动"的闭环流程，利用本地部署的多模态大模型（MLLM）实现自主操作。

### 1.1 Agent 架构概览

Open-AutoGLM 采用三层架构设计：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent 三层架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │   感知层         │  截图 (screencap) / UI解析 (uiautomator)           │
│  │   Perception    │  状态检测 (当前App/Activity)                        │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │   推理层         │  多模态编码 (Vision Encoder)                       │
│  │   Reasoning     │  语言模型 (GLM-4V 架构)                             │
│  │                 │  思维链推理 (Chain-of-Thought)                      │
│  │                 │  动作生成 (JSON Action)                             │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │   执行层         │  Tap/Swipe (adb input)                            │
│  │   Execution     │  Type (ADB Keyboard)                               │
│  │                 │  Launch (am start) / Back/Home (keyevent)          │
│  └─────────────────┘                                                    │
│                                                                         │
│           ↑ 循环执行，直到任务完成                                        │
│           └─────────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 纯本地感知 (Local Perception)

Agent 通过 ADB 获取手机实时状态，无需上传任何数据到云端：

| 感知类型 | 技术实现 | 数据格式 |
|----------|----------|----------|
| 视觉感知 | `adb shell screencap -p` | PNG 图像 (1080×2400+) |
| 结构化感知 | `adb shell uiautomator dump` | XML 元素树 |
| 状态感知 | `adb shell dumpsys activity` | 当前 App/Activity 信息 |

**核心代码位置**：`phone_agent/adb/screenshot.py`

```python
def get_screenshot(device_id: str = None) -> Image.Image:
    """通过 ADB 获取屏幕截图，返回 PIL Image"""
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])
    cmd.extend(["exec-out", "screencap", "-p"])
    
    result = subprocess.run(cmd, capture_output=True)
    return Image.open(io.BytesIO(result.stdout))
```

### 1.3 端侧推理 (On-Device Reasoning)

依托 **MLX** 框架，**AutoGLM-Phone-9B** 模型直接在 Mac 的 GPU/NPU 上运行。

#### 推理流程

```
用户输入任务 ──► PhoneAgent ──► 获取截图 ──► MLXModelClient
                                                   │
                                                   ▼
                                            多模态编码
                                                   │
                                                   ▼
                                            思维链推理 (CoT)
                                                   │
                                                   ▼
                                            输出: <think>...</think><answer>{JSON}</answer>
                                                   │
                                                   ▼
                                            解析动作 ──► ADB 执行 ──► 循环
```

#### 思维链 (Chain-of-Thought) 机制

模型输出遵循特定格式：

```
<think>
当前屏幕显示手机桌面，可以看到多个应用图标。
任务是打开微信，我需要找到微信图标。
观察到微信图标在屏幕中间偏上位置，坐标大约是 (540, 980)。
下一步应该点击这个位置来启动微信。
</think>
<answer>
{"_metadata": "do", "action": "Tap", "element": [540, 980]}
</answer>
```

### 1.4 物理执行 (Physical Execution)

生成的指令转化为 ADB 命令发送给安卓设备：

| 操作类型 | ADB 命令 | 参数说明 |
|----------|----------|----------|
| Tap | `adb shell input tap x y` | 坐标 [x, y] |
| Swipe | `adb shell input swipe x1 y1 x2 y2 duration` | 起止坐标 + 时长(ms) |
| Type | `adb shell am broadcast -a ADB_INPUT_TEXT --es msg "text"` | 文本内容 |
| Launch | `adb shell am start -n package/activity` | 包名/Activity |
| Back | `adb shell input keyevent 4` | KEYCODE_BACK |
| Home | `adb shell input keyevent 3` | KEYCODE_HOME |

---

## 2. 系统架构

本架构专为 Mac Apple Silicon 优化。

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    Mac 本地环境 (Apple Silicon)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   用户指令 ──► main.py ──► PhoneAgent                            │
│                              │                                   │
│                              ▼                                   │
│                       MLXModelClient                             │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                           │
│                    │   MLX 框架       │                           │
│                    │  (Metal 加速)   │                           │
│                    └────────┬────────┘                           │
│                             │                                    │
│                             ▼                                    │
│                    AutoGLM-Phone-9B                              │
│                      (本地权重)                                   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │               优化层                                      │   │
│   │   图像降采样 / KV Cache 量化 / 显存回收                    │   │
│   └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ ADB Shell
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    安卓设备 (USB/WiFi)                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ADB Bridge ◄──► 手机 App                                       │
│       │                                                          │
│       └── 操作指令 / 截图 / XML                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件说明

| 组件 | 文件位置 | 职责 |
|------|----------|------|
| PhoneAgent | `phone_agent/agent.py` | 核心调度器 |
| MLXModelClient | `phone_agent/model/mlx_client.py` | 本地推理客户端 |
| ActionHandler | `phone_agent/actions/handler.py` | 动作解析与执行 |
| ADBConnection | `phone_agent/adb/connection.py` | 设备连接管理 |
| Prompts | `phone_agent/config/prompts_*.py` | 系统提示词 |

---

## 3. 部署方案对比

Open-AutoGLM 支持多种部署方式：

| 方案 | 硬件要求 | 隐私 | 延迟 | 成本 | 适用场景 |
|------|----------|------|------|------|----------|
| 本地 MLX | Mac M1+ (16GB+) | 完全本地 | 13-20s/步 | 电费 | 隐私敏感 |
| 本地 vLLM | NVIDIA GPU (24GB+) | 完全本地 | 3-8s/步 | 电费 | 高性能 |
| 云端 API | 任意电脑 | 上传云端 | 2-5s/步 | Token 费 | 快速体验 |

### 3.1 本地 MLX 模式（本手册重点）

优势：隐私安全、零成本、离线可用

### 3.2 本地 vLLM 模式

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name autoglm-phone-9b \
    --model zai-org/AutoGLM-Phone-9B \
    --port 8000

python main.py --base-url http://localhost:8000/v1 --model autoglm-phone-9b "任务"
```

### 3.3 云端 API 模式

```bash
export PHONE_AGENT_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export PHONE_AGENT_API_KEY="your-api-key"
python main.py --model glm-4v-plus "任务"
```

---

## 4. 本地使用指南

### 4.1 环境安装

```bash
pip install mlx mlx-vlm torch torchvision transformers
pip install -r requirements.txt
pip install -e .
```

### 4.2 模型下载

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com  # 可选：国内镜像
huggingface-cli download --resume-download zai-org/AutoGLM-Phone-9B --local-dir ./models/AutoGLM-Phone-9B
```

### 4.3 运行

```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开微信"
```

### 4.4 可选：4-bit 量化

适用于 16GB 内存的 Mac：

```bash
python -m mlx_vlm.convert \
    --hf-path ./models/AutoGLM-Phone-9B \
    -q --q-bits 4 \
    --mlx-path ./autoglm-9b-4bit

python main.py --local --model ./autoglm-9b-4bit "打开微信"
```

### 4.5 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--local` | 本地推理 | `--local` |
| `--model` | 模型路径 | `--model ./models/AutoGLM-Phone-9B` |
| `--device-id` | 设备 ID | `--device-id 192.168.1.100:5555` |
| `--lang` | 语言 | `--lang en` |
| `--list-apps` | 列出应用 | - |
| `--list-devices` | 列出设备 | - |

---

## 5. 性能优化技术详解

### 5.1 内置优化

| 优化项 | 技术 | 效果 |
|--------|------|------|
| 智能降采样 | PIL resize to 1024px | 计算量 -60% |
| KV Cache 量化 | kv_bits=8 | 显存 -30% |
| 显存回收 | mx.clear_cache() | 防止泄漏 |

### 5.2 智能图像降采样

```python
def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
```

### 5.3 KV Cache 量化

```python
output = mlx_vlm.generate(
    self.model,
    self.processor,
    kv_bits=8,  # 8-bit KV Cache
    ...
)
```

### 5.4 显存回收

```python
mx.clear_cache()
gc.collect()
```

---

## 6. 调试与日志

### 6.1 Verbose 模式输出

```text
==================================================
Phone Agent - AI-powered phone automation
==================================================
Model: ./autoglm-9b-4bit
Task: 打开网易云音乐

思考过程:
--------------------------------------------------
当前在系统桌面，需要先启动网易云音乐应用...
--------------------------------------------------
执行动作: Launch "网易云音乐"
--------------------------------------------------
Step Time: 13.99s
```

### 6.2 保存截图调试

```python
screenshot = get_screenshot(self.device_id)
screenshot.save(f"debug_step_{self._step_count}.png")
```

---

## 7. API 参考

### 7.1 ModelConfig

```python
@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 3000
    temperature: float = 0.1
    is_local: bool = False
```

### 7.2 AgentConfig

```python
@dataclass
class AgentConfig:
    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    verbose: bool = True
```

### 7.3 PhoneAgent

```python
class PhoneAgent:
    def run(self, task: str) -> str: ...
    def step(self, task: str | None = None) -> StepResult: ...
    def reset(self) -> None: ...
```

### 7.4 支持的动作

| 动作 | JSON 格式 |
|------|-----------|
| Tap | `{"action": "Tap", "element": [x, y]}` |
| Swipe | `{"action": "Swipe", "element": [[x1,y1], [x2,y2]]}` |
| Type | `{"action": "Type", "text": "..."}` |
| Launch | `{"action": "Launch", "app": "微信"}` |
| Back | `{"action": "Back"}` |
| Home | `{"action": "Home"}` |

---

## 8. 故障排查指南

### 8.1 设备未找到

```bash
adb kill-server
adb start-server
adb devices
```

### 8.2 模型加载失败

```bash
huggingface-cli download --resume-download zai-org/AutoGLM-Phone-9B --local-dir ./models/AutoGLM-Phone-9B
```

### 8.3 内存不足

使用 4-bit 量化版本（见 4.4 节）

### 8.4 文本输入不工作

```bash
adb shell ime list -a | grep ADB
```

### 8.5 编码问题

```bash
# Windows
set PYTHONIOENCODING=utf-8
# Linux/Mac
export PYTHONIOENCODING=utf-8
```

---

## 9. 能力边界

### 优势

- 隐私安全：数据永不离开本机
- 离线可用：断网也能运行
- 零成本：无 API 费用

### 限制

- 推理延迟：13-18 秒/步
- 无法进行高频操作（如游戏）
- 无法识别音频内容

---

## 10. 项目引用

```bibtex
@article{liu2024autoglm,
  title={Autoglm: Autonomous foundation agents for guis},
  author={Liu, Xiao and Qin, Bo and others},
  journal={arXiv preprint arXiv:2411.00820},
  year={2024}
}
```

官方项目：[https://github.com/zai-org/Open-AutoGLM](https://github.com/zai-org/Open-AutoGLM)
