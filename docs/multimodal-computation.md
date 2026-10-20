# Multimodal computation foundation: image, audio and video

Scope: **bounded, executable mathematical adapters**, not a native media loader or large pretrained model implementation. All four operations are registered in Learning Studio's free-form catalogue and share its graph readiness, concrete numerical interpreter, NumPy graph exporter and PyTorch CPU exporter.

## Core representation contract

The existing micro-interpreter caps matrix outputs at 8 rows and 16 columns. This initial phase does **not** pretend that a two-axis tensor is intrinsically an image, waveform or video. Each block declares exactly how the host's numeric matrix encodes other axes.

| Block | Input 2D carrier | Interpretation | Output |
| --- | --- | --- | --- |
| `vision_patchify` | `[H, W*C]` | HWC, channel-last image | `[(H/Ph)*(W/Pw), Ph*Pw*C]` |
| `video_tubelet` | `[T*H,W*C]` | THWC, frame-major video | `[(T/Pt)*(H/Ph)*(W/Pw),Pt*Ph*Pw*C]` |
| `audio_stft_power` | `[1,Nsamples]` | Mono waveform, sample positions | `[1+floor((Nsamples-Nfft)/hop),Nfft/2+1]` |
| `audio_log_power` | `[frames,freq]` | Nonnegative power values | Same shape |

Patch ordering is token-major by spatial y then x; pixels within each image patch are y→x→channel. Tubelets traverse coarse time→y→x; inside each tubelet are dt→dy→dx→channel. Neither block projects features: attach `linear` or explicit learned matrices to obtain patch/tubelet embeddings.

The STFT uses periodic Hann `w[n]=0.5-0.5*cos(2*pi*n/Nfft)`, direct real-input DFT at frequencies `0..Nfft/2`, squared magnitude, `center=False`, and no padding. The log transform is exactly `log10(max(power,floor))`. **It is not Whisper log-Mel**: there is no resampling, mel filterbank, Whisper's relative clipping or final normalization; no real audio file is decoded. An actual Mel filterbank can later be an explicit parameter matrix connected through the existing MatMul block, which permits inspecting every coefficient.

## Real implementation comparison

- **ViT:** Hugging Face `ViTPatchEmbeddings` uses a learned `Conv2d(kernel_size=patch_size,stride=patch_size)`, followed by spatial flattening/transposition. Our spatial Patchify is solely the unweighted rearrangement preceding a separate projection; arbitrary real Conv2d weights are not reproduced. Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
- **ViViT:** official `VivitTubeletEmbeddings` uses learned Conv3d for spatiotemporal tubelets. This bounded Tubelet block computes only non-overlapping patches; it can feed existing self attention but is not equivalent to Conv3d until a mathematically matched learned projection is added. Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vivit/modeling_vivit.py
- **Whisper:** official `log_mel_spectrogram` computes periodic-Hann STFT with NFFT=400 and hop=160 at 16kHz, multiplies the **power spectrum** by a separately loaded Mel filterbank, and performs the log/clamp/affine path. Here NFFT is 2/4/8/16 due to the tiny CPU teaching matrix. Source: https://github.com/openai/whisper/blob/main/whisper/audio.py
- **Wan2.2 video VAE:** actual causal 3D convolutions pad the temporal axis asymmetrically and use temporal feature caches during chunked encoding. Our no-weight tubelets are a different calculation, intentionally not described as a VAE replica. Source: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py

## Free composition and cross modality

Lessons 46–49 demonstrate `image_vit_patch`, `audio_spectrogram`, `video_vivit_tubelet`, and `multimodal_cross_attention`. The final lesson uses:

```text
pixel [H,W*C] ─ Patchify ─ Linear ─── Q ──── Cross-Attention ── Q ── Cross-Attention → fused image tokens
                                            K,V ↑                     K,V ↑
wave [1,N] ─ STFT ─ Log10 ─ Linear ───────────┘                       │
frames [T*H,W*C] ─ Tubelet ─ Linear ──────────────────────────────────┘
```

The shape contracts ensure Q and K widths agree and Key/Value token counts agree. This is **a runnable mathematical fusion graph**; it is not a claim about CLIP, Flamingo, Qwen-Omni, audio/video alignment or any pretrained weights.

## Evidence and tests

- `tests/multimodal_contracts.js`: explicit image pixel and video frame order, analytically checked constant-wave periodic Hann FFT, log10 clipping, shape rejection, four executable graph lesson contracts and source-backed knowledge cards.
- `tests/multimodal_codegen_parity.js`: run every node of all four graphs in separate NumPy and CPU PyTorch processes and compare against the independent JavaScript interpreter, not a baked array of the expected output.
- `tests/browser_multimodal.py`: run the same four graphs in Chromium, inspect the source cards, download independent NumPy/PyTorch scripts and save real screenshots.
- `graph_cost_report.js` intentionally does not count preprocessing FFT or patch reshapes in its partial MatMul estimator; it exposes them as non-GEMM operators. No GPU timing or peak VRAM claim is made.

## Subsequent expansion

The next architecture boundary is an explicitly typed, rank-agnostic tensor IR with axis labels and storage/stride semantics. On that foundation, introduce reusable Conv2d/Conv3d, resampling, Mel filters, VAE encode/decode, diffusion scheduler/noise prediction, audio token codec quantization and joint/spatial/temporal attention. Each should have a documented paper-source distinction and actual CPU execution plus native NumPy/PyTorch parity before being called an executable block.
