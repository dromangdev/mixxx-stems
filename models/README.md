# AI Stem Separation Models for Mixxx

This directory holds the ONNX deep neural network models used by Mixxx's AI Stem Separation engine (`mixxx::HTDemucsEngine`).

## Supported Models

### 1. HTDemucs Fine-Tuned (HTDemucs-FT) [Active Default]
High-fidelity specialist models trained with 4x extended training steps by Meta AI:
- `htdemucs_ft_vocals.onnx` (158 MB FP16 weights): Specialised vocal separation.
- `htdemucs_ft_drums.onnx` (158 MB FP16 weights): Specialised drum separation.
- Download from: `https://huggingface.co/StemSplitio/htdemucs-ft-onnx`

### 2. HTDemucs v4 Standard
Unified 4-stem Hybrid Transformer model (Alexander Défossez / Meta AI):
- `htdemucs.onnx` (158 MB FP16 weights)
- Download from: `https://huggingface.co/StemSplitio/htdemucs-onnx`

### 3. SCNet Large (Sparse Compression Network)
Winner of Sony MDX-C 2023 Challenge (4 stems: drums, bass, other, vocals):
- `scnet-large.int8.onnx` (43 MB INT8 weights)
- Download from: `https://huggingface.co/audiojs/scnet-large`

### 4. BS-RoFormer (Band-Split RoPE Transformer)
Band-Split RoPE Transformer by viperx / lucidrains (SDR 12.9 dB):
- `bs_roformer_ep317_quantized.onnx` (158 MB UINT8 weights)
- Download from: `https://huggingface.co/xycld/BS-RoFormer-ONNX`

## Storage Note
Due to file sizes (>100MB), `.onnx` files are excluded from git tracking and should be stored locally or cached in:
`~/Library/Application Support/Mixxx/models/` or `<mixxx-root>/models/`.
