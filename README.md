# Qwen3-TTS 固定音色推理

使用 Qwen3-TTS-12Hz-0.6B-Base 和仓库中固定的参考音频合成中文语音。参考音频为原录音中 `speaker_01/00071_265.86_270.96.wav` 片段。脚本使用声纹向量模式，不需要参考音频的文字稿；生成的音色与发音可能随参考音质和生成结果变化。

## 准备环境

需要 Python 3.10–3.12、Git LFS，以及约 2.4 GB 的模型存储空间。支持 CPU 和 CUDA GPU；CPU 推理速度较慢。

```bash
git lfs install
git clone https://github.com/ykfthu/yawang.git
cd yawang
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需 GPU 推理，请根据机器的 CUDA 环境安装与之匹配的 [PyTorch](https://pytorch.org/get-started/locally/)。

## 合成

```bash
python inference.py '不会真有人觉得带贾云鹏打海斗是放松吧' -o output.wav
```

默认自动选择 CUDA（可用时）或 CPU。可指定设备、随机种子和 CPU 线程数：

```bash
python inference.py '今天晚上斗不斗？' -o result.wav --device cpu --seed 42 --threads 8
```

输出为模型生成的 WAV 音频。运行 `python inference.py --help` 查看参数。模型权重以 Git LFS 存储；若提示权重未下载，请先运行 `git lfs pull`。

## 文件

- `inference.py`：推理入口。
- `reference/speaker_01.wav`：固定的参考音频。
- `model/Qwen3-TTS-12Hz-0.6B-Base/`：模型、语音 tokenizer、配置及上游模型卡。

模型来自 [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)，模型卡标明许可证为 Apache-2.0。推理依赖来自 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。
