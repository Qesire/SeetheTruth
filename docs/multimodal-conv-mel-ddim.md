# Executable multimodal primitives: Conv2D, HTK Mel and epsilon-DDIM

Three additional **freely composable** graph nodes operate on the existing bounded educational matrices while preserving explicit external layout contracts. No checkpoint is loaded or external multimedia decoded.

## Vision Conv2D

- Input carrier `X: [H,W*Cin]` means channel-last HWC; true model weight is an independent graph input, `kernel: [Cout,Kh*Kw*Cin]`. The kernel may be a persistent, trainable `parameter` node, so the graph preserves weight identity and PyTorch autograd.
- Real `valid` cross-correlation, with stride `S` and **no padding or bias**:
  `Y[y,x,oc] = Σ_dy,dx,ci X[S*y+dy,(S*x+dx)*Cin+ci] * W[oc,(dy*Kw+dx)*Cin+ci]`.
- Output `[floor((H-Kh)/S)+1, (floor((W-Kw)/S)+1)*Cout]`. This flattened output representation is distinct from a real NCHW activation layout. PyTorch backend converts to `[1,Cin,H,W]` and `[Cout,Cin,Kh,Kw]`, calls `torch.nn.functional.conv2d`, then converts to the same HWC carrier. NumPy and browser implement the same sums. Trainable kernel VJP is verified against analytical sums **in real PyTorch**, not claimed for every JavaScript optimizer.
- Relation: patch embedding in ViT uses kernel size=stride=patch size to fuse local rearrangement and trainable spatial projection. It is not equivalent to bare Patchify without a learned kernel. Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

## HTK Mel power filterbank

- Takes actual *nonnegative* STFT power `S[frame,k]`, with `k=0..Nfft/2`.
- `m(f)=2595 log10(1+f/700)` converts Hz to an explicit HTK Mel coordinate. Linearly spaced Mel band edges are converted back to Hz; triangular weights at FFT bin frequency `k*sampleRate/Nfft` determine `W[k,m]`.
- Computes `M[t,m] = Σ_k S[t,k]*W[k,m]`; the Mel weights depend only on specified `fftSize`, `sampleRate`, `bands`, and remain visible and reproducible. Shape `[frames,bands]` is verified independently.
- No Slaney normalization, real-signal sample-rate conversion, VAD, filterbank file loading, or Whisper relative log-clipping is implied. Small Nfft can yield Mel filters with no sampled support. Whisper uses its own stored Mel filters: https://github.com/openai/whisper/blob/main/whisper/audio.py

## DDIM deterministic epsilon update

- A single `diffusion_ddim_step` consumes `x_t` and explicit `epsilon` with the same actual shape.
- Validates cumulative `0<alphaT<=alphaPrevious<=1` and implements the eta=0, epsilon-prediction special case:
  `x0 = (xt - sqrt(1-alphaT)*epsilon)/sqrt(alphaT)`;
  `xprev = sqrt(alphaPrevious)*x0 + sqrt(1-alphaPrevious)*epsilon`.
- The compiler uses equivalent scalar coefficients but still computes real outputs from runtime `xt` and `epsilon`, rather than embedding the outputs.
- No model forward, stochasticity, `prediction_type` variant, scheduler timestep mapping, clipping, CFG, VAE decoder or image/video generation is claimed. The complete diffusers `DDIMScheduler.step` has additional branches: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

## Evidence and working examples

Lessons **50/51/52** instantiate a trainable-kernel image Conv2D → SiLU → spatial attention graph, audio Hann-STFT Power → HTK Mel → Log10 → Attention, and a two-step deterministic DDIM chain. Each remains editable; the learner may replace kernels, weights, inputs and follow-up operators at will while the existing DAG and shape checks remain strict.

The mathematical evidence gates are `tests/multimodal_advanced_contracts.js`, `tests/concept_flow_contracts.js` and actual JS→NumPy→PyTorch full-graph forward parity + real PyTorch autograd for Conv2D in `tests/multimodal_advanced_codegen_parity.js`. Chromium additionally runs, inspects source cards, downloads generated code and saves screenshots in the `multimodal_screenshots/` Actions artifact.

These are concrete educational CPU calculations; not evidence of GPU performance or real-model parameter provenance. The correct next boundary is a tensor IR whose explicitly labeled named axes and strides persist across graph nodes (including true NCHW, BCTHW and THWC), rather than depending on an educational carrier declaration at the operator boundary.
