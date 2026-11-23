#!/usr/bin/env python3
"""Synthesize Chinese speech using the bundled Qwen3-TTS voice reference."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="要合成的中文文本")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8, help="CPU 推理线程数")
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("文本不能为空")
    if args.threads < 1:
        parser.error("线程数必须大于零")
    model_dir = ROOT / "model" / "Qwen3-TTS-12Hz-0.6B-Base"
    reference = ROOT / "reference" / "speaker_01.wav"
    for path in (model_dir / "model.safetensors", model_dir / "speech_tokenizer" / "model.safetensors", reference):
        if not path.is_file() or path.stat().st_size < 1024:
            parser.error(f"文件不存在或 Git LFS 权重未下载：{path}")

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map=device,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    wavs, sample_rate = model.generate_voice_clone(
        text=args.text, language="Chinese", ref_audio=str(reference),
        x_vector_only_mode=True, max_new_tokens=2048,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), wavs[0], sample_rate)
    print(f"已保存：{args.output}（{len(wavs[0]) / sample_rate:.2f} 秒，{sample_rate} Hz）")


if __name__ == "__main__":
    main()
