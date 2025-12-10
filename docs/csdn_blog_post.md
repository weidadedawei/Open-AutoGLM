# 【零基础入门】Open-AutoGLM 完全指南：Mac 本地部署 AI 手机助理（原理+部署+优化）

> **摘要**：本教程教你在 Mac (Apple Silicon) 上部署智谱 AutoGLM-Phone-9B 多模态大模型，实现完全本地化、隐私安全、零成本的手机 AI 助理。从原理到部署、从操作到优化，一文搞定！

---

## 目录

- [1. 什么是 Open-AutoGLM？](#1-什么是-open-autoglm)
- [2. 核心原理解析](#2-核心原理解析)
- [3. 环境准备（超详细）](#3-环境准备超详细)
- [4. 模型下载与部署](#4-模型下载与部署)
- [5. 实战操作指南](#5-实战操作指南)
- [6. 性能优化详解](#6-性能优化详解)
- [7. API 与进阶用法](#7-api-与进阶用法)
- [8. 常见问题 FAQ](#8-常见问题-faq)

---

## 1. 什么是 Open-AutoGLM？

### 1.1 项目简介

**Open-AutoGLM** 是智谱 AI 开源的手机 AI 助理框架。它能让你的 Mac 变成一个"超级大脑"，通过 USB 或 WiFi 远程控制你的安卓手机，自动完成各种任务。

想象一下这些场景：
- "帮我在饿了么点一份黄焖鸡米饭"
- "打开微信给妈妈发消息说我今晚不回家吃饭"
- "在网易云音乐搜索周杰伦的歌并播放"
- "打开 B 站搜索 Python 教程"

这些以前需要你亲自动手的操作，现在只需一句话，AI 就能帮你完成！

### 1.2 为什么选择本地部署？

| 对比项 | 云端 API 模式 | 本地 MLX 模式 |
|--------|--------------|--------------|
| **隐私安全** | 截图上传云端 | 数据永不出本机 |
| **运行成本** | 按 Token 收费 | 电费即成本 |
| **网络依赖** | 断网不可用 | 完全离线可用 |
| **响应延迟** | 网络延迟波动 | 本地计算稳定 |

### 1.3 适合谁？

- **开发者**：想了解 AI Agent 如何工作
- **隐私敏感用户**：不希望手机截图上传云端
- **极客玩家**：想在本地玩转多模态大模型
- **学习者**：想学习 MLX、ADB、多模态模型的实际应用

---

## 2. 核心原理解析

### 2.1 AI Agent 工作原理

Open-AutoGLM 采用经典的 **感知-思考-行动 (Perception-Thinking-Action)** 循环：

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 工作循环                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │  感 知   │ ──→ │  思 考   │ ──→ │  行 动   │          │
│   │          │     │          │     │          │          │
│   │ 截图     │     │ 理解状态 │     │ 点击     │          │
│   │ UI解析   │     │ 规划步骤 │     │ 滑动     │          │
│   │ App状态  │     │ 生成指令 │     │ 输入     │          │
│   └──────────┘     └──────────┘     └──────────┘          │
│        ↑                                  │                 │
│        └──────────────────────────────────┘                 │
│                     循环执行                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 三层架构详解

**感知层 (Perception Layer)**

| 感知类型 | 技术实现 | 数据格式 |
|----------|----------|----------|
| 视觉感知 | `adb shell screencap -p` | PNG 图像 |
| 结构化感知 | `adb shell uiautomator dump` | XML 元素树 |
| 状态感知 | `adb shell dumpsys activity` | App/Activity 信息 |

**推理层 (Reasoning Layer)**

AutoGLM-Phone-9B 是一个 **视觉-语言模型 (VLM)**：

```
输入: [系统提示] + [任务描述] + [手机截图]
           ↓
     多模态编码器 (Vision Encoder)
           ↓
       Transformer 推理
           ↓
输出: <think>推理过程</think><answer>{"action": "Tap", "element": [500, 300]}</answer>
```

模型会先在 `<think>` 标签中进行推理（类似 ChatGPT o1 的思考过程），然后在 `<answer>` 标签中输出具体的 JSON 操作指令。

**执行层 (Execution Layer)**

| 操作类型 | ADB 命令 | 说明 |
|----------|----------|------|
| Tap | `adb shell input tap x y` | 点击坐标 |
| Swipe | `adb shell input swipe x1 y1 x2 y2` | 滑动 |
| Type | `adb shell am broadcast -a ADB_INPUT_TEXT` | 输入文字 |
| Launch | `adb shell am start -n package/activity` | 启动应用 |

### 2.3 MLX 框架介绍

**MLX** 是苹果公司专门为 Apple Silicon (M1/M2/M3/M4) 开发的深度学习框架：

- **统一内存架构**：GPU 和 CPU 共享内存，无需复制数据
- **延迟编译**：只编译实际执行的代码路径
- **原生 Metal 加速**：充分利用 Apple GPU

对于本项目，MLX 让我们能在 Mac 上高效运行 9B 参数的多模态模型！

---

## 3. 环境准备（超详细）

### 3.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 系统版本 | macOS 13.3+ | macOS 14+ (Sonoma) |
| 芯片 | M1 | M1 Max / M2 Pro 及以上 |
| 内存 | 16GB (量化后) | 32GB+ |
| 硬盘 | 20GB 可用空间 | 50GB+ SSD |
| Python | 3.10+ | 3.11 |

### 3.2 安装 Python 环境

**方法 A：使用 Homebrew + pyenv（推荐）**

```bash
# 1. 安装 Homebrew (如果没有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 pyenv
brew install pyenv

# 3. 配置 shell (以 zsh 为例)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 4. 安装 Python 3.11
pyenv install 3.11.9
pyenv global 3.11.9

# 5. 验证安装
python --version  # 应该显示 Python 3.11.9
```

**方法 B：使用 Conda**

```bash
# 1. 下载 Miniforge (适合 Apple Silicon 的 Conda)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# 2. 安装
bash Miniforge3-MacOSX-arm64.sh

# 3. 创建虚拟环境
conda create -n autoglm python=3.11
conda activate autoglm
```

### 3.3 安装 ADB 工具

**ADB (Android Debug Bridge)** 是连接 Mac 和安卓手机的桥梁。

```bash
# 使用 Homebrew 安装
brew install android-platform-tools

# 验证安装
adb version
```

### 3.4 配置安卓手机

**步骤 1：开启开发者模式**

1. 打开 **设置 → 关于手机**
2. 连续点击 **版本号** 7 次
3. 看到提示"您已进入开发者模式"

> 不同品牌手机的位置可能略有不同。华为在"关于手机"，小米在"我的设备"。

**步骤 2：开启 USB 调试**

1. 返回 **设置 → 系统 → 开发者选项**
2. 开启 **USB 调试**
3. 开启 **USB 安装** (如果有)
4. 关闭 **监控 ADB 安装应用** (如果有)

> 部分手机需要重启后设置才能生效！

**步骤 3：连接并授权**

1. 使用**数据线**（不是纯充电线！）连接手机和 Mac
2. 手机上会弹出授权窗口，勾选"始终允许"并点击确定
3. 在终端验证连接：

```bash
adb devices
# 输出应该类似：
# List of devices attached
# ABCD1234567890    device
```

### 3.5 安装 ADB Keyboard

ADB Keyboard 是一个特殊的输入法，允许通过 ADB 命令输入中文。

1. 下载 APK：[ADBKeyboard.apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)

2. 通过 ADB 安装：
```bash
adb install ADBKeyboard.apk
```

3. 设置为当前输入法：
   - 手机上进入 **设置 → 语言和输入法 → 管理键盘**
   - 启用 **ADB Keyboard**

4. 验证安装：
```bash
adb shell ime list -a | grep ADB
# 应该输出: com.android.adbkeyboard/.AdbIME
```

### 3.6 安装项目依赖

```bash
# 1. 克隆项目
git clone https://github.com/zai-org/Open-AutoGLM.git
cd Open-AutoGLM

# 2. 安装 MLX 相关依赖
pip install mlx mlx-vlm torch torchvision transformers

# 3. 安装项目依赖
pip install -r requirements.txt
pip install -e .

# 4. 验证安装
python -c "import mlx; import phone_agent; print('安装成功！')"
```

---

## 4. 模型下载与部署

### 4.1 下载模型

**方法 A：使用 HuggingFace CLI（推荐）**

```bash
# 安装 CLI 工具
pip install -U "huggingface_hub[cli]"

# 设置国内镜像（可选，加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（约 20GB）
huggingface-cli download --resume-download zai-org/AutoGLM-Phone-9B --local-dir ./models/AutoGLM-Phone-9B
```

**方法 B：使用 ModelScope（国内最快）**

```bash
pip install modelscope

python -c "from modelscope import snapshot_download; snapshot_download('ZhipuAI/AutoGLM-Phone-9B', local_dir='./models/AutoGLM-Phone-9B')"
```

### 4.2 启动运行

下载完成后即可运行：

```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开微信"
```

### 4.3 可选：4-bit 量化（推荐 16GB 内存用户）

如果你的 Mac 内存只有 16GB，或希望更快的推理速度，可以对模型进行量化：

**量化效果对比：**

| 对比项 | 原始模型 (FP16) | 4-bit 量化 |
|--------|----------------|------------|
| 模型大小 | ~20GB | ~6.5GB |
| 内存占用 | 需 32GB+ | 16GB 即可 |
| 推理速度 | 较慢 | 提升约 3x |
| 精度损失 | 基准 | 约 1-2% |

**量化步骤：**

```bash
# 执行量化转换（约 15-20 分钟）
python -m mlx_vlm.convert \
    --hf-path ./models/AutoGLM-Phone-9B \
    -q \
    --q-bits 4 \
    --mlx-path ./autoglm-9b-4bit
```

**使用量化模型运行：**

```bash
python main.py --local --model ./autoglm-9b-4bit "打开B站搜索二次元"
```

---

## 5. 实战操作指南

### 5.1 基础命令

**交互模式：**

```bash
python main.py --local --model ./models/AutoGLM-Phone-9B

# 然后输入任务：
> 打开微信
> 搜索张三并发送消息你好
> 退出
```

**单任务模式：**

```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开抖音刷5个视频"
```

### 5.2 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--local` | 使用本地 MLX 推理 | `--local` |
| `--model` | 模型路径 | `--model ./models/AutoGLM-Phone-9B` |
| `--device-id` | 指定设备 | `--device-id 192.168.1.100:5555` |
| `--lang` | 语言 (cn/en) | `--lang en` |
| `--list-apps` | 列出支持的应用 | `--list-apps` |
| `--list-devices` | 列出连接的设备 | `--list-devices` |

### 5.3 任务示例

**社交通讯：**
```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开微信给张三发消息说：下午三点开会"
```

**电商购物：**
```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开淘宝搜索蓝牙耳机按价格排序"
```

**美食外卖：**
```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开美团外卖点一份黄焖鸡米饭"
```

**视频娱乐：**
```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开B站搜索Python教程"
```

**音乐播放：**
```bash
python main.py --local --model ./models/AutoGLM-Phone-9B "打开网易云音乐搜索周杰伦的晴天并播放"
```

### 5.4 WiFi 远程调试

无需 USB 线也能控制手机！

**步骤 1：开启无线调试**
1. 确保手机和 Mac 在同一 WiFi 下
2. 进入 **开发者选项 → 无线调试**
3. 开启无线调试，记下 IP 和端口

**步骤 2：连接设备**
```bash
# 连接远程设备
adb connect 192.168.1.100:5555

# 验证连接
adb devices

# 使用远程设备执行任务
python main.py --local --model ./models/AutoGLM-Phone-9B \
    --device-id 192.168.1.100:5555 \
    "打开抖音刷视频"
```

### 5.5 支持的操作类型

| 操作 | 说明 |
|------|------|
| `Tap` | 点击指定坐标 |
| `Swipe` | 滑动屏幕 |
| `Type` | 输入文本 |
| `Launch` | 启动应用 |
| `Back` | 返回上一页 |
| `Home` | 返回桌面 |
| `Long Press` | 长按 |
| `Double Tap` | 双击 |
| `Wait` | 等待页面加载 |
| `Take_over` | 请求人工接管 |

---

## 6. 性能优化详解

### 6.1 内置优化（自动生效）

我们在代码中实现了三项关键优化：

**优化 1：智能图像降采样**

现代手机屏幕动辄 2K/4K，直接处理太慢。系统自动将图像长边限制在 1024 像素以内。

| 原始尺寸 | 处理后尺寸 | 像素减少 |
|----------|------------|----------|
| 2400×1080 | 1024×460 | 82% |
| 1920×1080 | 1024×576 | 72% |

**优化 2：KV Cache 量化**

推理时启用 `kv_bits=8`，将 KV Cache 从 FP16 量化到 INT8：
- 显存占用降低约 30%
- 推理速度略有提升

**优化 3：显存强制回收**

每步推理后强制执行 `mx.clear_cache()` 和 `gc.collect()`：
- 防止"越用越卡"
- 长时间运行保持稳定

### 6.2 手动优化建议

1. **关闭不必要的后台应用**：MLX 推理需要大量内存
2. **使用有线连接**：USB 比 WiFi 更稳定，截图传输更快
3. **降低手机亮度**：高亮度截图文件更大
4. **定期重启模型**：如果变慢了，Ctrl+C 终止后重新启动

### 6.3 性能参考

在 Mac Studio M1 Max (32GB) 上使用 4-bit 量化模型：

| 阶段 | 耗时 |
|------|------|
| 模型加载 | 约 30 秒 |
| 单步推理 | 13-18 秒 |
| 截图获取 | 0.5-1 秒 |

完整任务示例："打开网易云音乐搜索歌曲一滴泪的时间播放"
- 总步数：6 步
- 总耗时：约 2 分 18 秒

---

## 7. API 与进阶用法

### 7.1 Python API 调用

```python
from phone_agent import PhoneAgent
from phone_agent.model import ModelConfig
from phone_agent.agent import AgentConfig

# 配置模型
model_config = ModelConfig(
    model_name="./models/AutoGLM-Phone-9B",
    is_local=True,
    max_tokens=3000,
    temperature=0.1,
)

# 配置 Agent
agent_config = AgentConfig(
    max_steps=50,
    verbose=True,
    lang="cn",
)

# 创建并运行
agent = PhoneAgent(
    model_config=model_config,
    agent_config=agent_config,
)

result = agent.run("打开抖音刷3个视频")
print(f"任务结果: {result}")
```

### 7.2 自定义回调函数

处理敏感操作和人工接管场景：

```python
def my_confirmation(message: str) -> bool:
    """敏感操作确认（如支付）"""
    print(f"检测到敏感操作: {message}")
    return input("是否继续？(y/n): ").lower() == "y"

def my_takeover(message: str) -> None:
    """人工接管（如登录验证）"""
    print(f"需要人工操作: {message}")
    input("完成后按回车继续...")

agent = PhoneAgent(
    confirmation_callback=my_confirmation,
    takeover_callback=my_takeover,
)
```

### 7.3 批量执行任务

```python
tasks = [
    "打开微信给张三发消息：会议改到下午4点",
    "打开支付宝查看余额",
    "打开美团查看最近订单",
]

for task in tasks:
    result = agent.run(task)
    print(f"完成: {task}")
    agent.reset()
```

### 7.4 配置参数参考

**ModelConfig 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | - | 模型路径 |
| `is_local` | bool | False | 使用本地推理 |
| `max_tokens` | int | 3000 | 最大输出 token |
| `temperature` | float | 0.1 | 采样温度 |

**AgentConfig 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_steps` | int | 100 | 最大执行步数 |
| `device_id` | str | None | ADB 设备 ID |
| `lang` | str | cn | 语言 |
| `verbose` | bool | True | 显示详细输出 |

---

## 8. 常见问题 FAQ

### Q1: 设备未找到

```bash
adb devices  # 输出为空
```

**解决方案：**
```bash
adb kill-server
adb start-server
adb devices
```

常见原因：
- 数据线是纯充电线
- 没有在手机上授权
- 开发者选项未正确开启

### Q2: 模型加载失败 / 下载中断

```bash
# 使用断点续传
huggingface-cli download --resume-download zai-org/AutoGLM-Phone-9B --local-dir ./models/AutoGLM-Phone-9B

# 或使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: 内存不足 (Killed / MemoryError)

1. 使用 4-bit 量化版本（见 4.3 节）
2. 关闭其他应用
3. 重启 Mac 后再试

### Q4: 文本输入不工作

1. 确认已安装 ADB Keyboard
2. 确认已在系统中启用
3. 验证安装：
```bash
adb shell ime list -a | grep ADB
```

### Q5: 截图失败 (黑屏)

这是系统安全机制，某些应用（支付、银行）禁止截图。模型会自动请求人工接管。

### Q6: 运行变慢 / 卡顿

```bash
# 终止并重新启动
Ctrl+C
python main.py --local --model ./models/AutoGLM-Phone-9B "你的任务"
```

### Q7: WiFi 连接失败

1. 确保手机和电脑在同一 WiFi
2. 确保手机开启了无线调试
3. 检查防火墙是否阻止 5555 端口

### Q8: Windows/Linux 编码问题

```bash
# Windows
set PYTHONIOENCODING=utf-8

# Linux
export PYTHONIOENCODING=utf-8
```

---

## 总结

通过本教程，你已经学会了：

- Open-AutoGLM 的工作原理（感知-思考-行动循环）
- MLX 框架的优势和使用方法
- 完整的环境配置和模型部署
- 可选的 4-bit 量化优化
- 实战操作和进阶用法

**项目地址**: https://github.com/zai-org/Open-AutoGLM

---

**如果觉得有帮助，请点赞收藏！有问题欢迎评论区讨论！**
