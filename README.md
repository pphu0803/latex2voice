# latex-tts

让 TTS 流畅朗读 LaTeX 数学公式（英语优先）。本地离线部署。

## 架构

```
LaTeX 文本
   │
   ▼
[TN 模块]  Qwen3-4B LoRA 微调：LaTeX → 英语大白话朗读文本
   │         "x^2 + y^2 = z^2" → "x squared plus y squared equals z squared"
   ▼
[TTS]      Kokoro-82M（本地离线，英语）
   │
   ▼
音频
   │
   ▼
[ASR]      whisper-large-v3-turbo（回转评测）
   │
   ▼
转写文本 ──→ 与原公式语义比对（sympy 等价 / LLM judge）→ 按难度 L1-L4 报告通过率
```

核心立场：**不微调 TTS 本体**。公式读不好是前端不认识 LaTeX，拆成 TN 文本任务用小模型解决。

## 目录

```
src/latextts/
  formula_gen.py    # 公式生成器（L1-L4 难度分级）
  tn_data.py        # TN 训练数据合成（公式+canonical 读法对）
  rules_en.py       # LaTeX→英语朗读规则库（核心资产）
  tn_infer.py       # TN 推理（LoRA 加载）
  tts_kokoro.py     # Kokoro TTS 封装
  asr_whisper.py    # Whisper 回转封装
  eval_roundtrip.py # 端到端评测（sympy 等价 + 守卫）
scripts/
  demo_tier0.py     # 0 档 demo：规则 TN → TTS → ASR → 比对
  download_models.py# ModelScope 下载 Qwen3-4B / Kokoro
configs/
  difficulty.yaml   # 难度分级定义
data/
  formulas/         # 生成的公式集（jsonl）
  tn_train/         # TN 训练数据
  tn_eval/          # 评测集
  audio/            # TTS 音频输出
models/             # 本地模型权重落点
```

## 模型清单

| 用途 | 模型 | 状态 |
|---|---|---|
| ASR 回转评测 | whisper-large-v3-turbo | ✅ 本机 HF 缓存已有（3.1G） |
| TN 基座 | Qwen3-4B | 待下载（ModelScope） |
| TTS | Kokoro-82M | 待下载（HF） |
| TN 训练 | LoRA r=32 | 本机 8060S |

## 快速开始

```bash
cd /c/Code/latex-tts
python -m src.latextts.formula_gen --level L1 --n 20   # 生成公式
python scripts/demo_tier0.py                            # 0 档 demo
```
