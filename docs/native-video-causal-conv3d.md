# Native rank-5 causal video convolution and chunked history

This is an **implemented bounded mathematical subset** of temporal-causal video convolution, rather than a generic video-generation decoder or a checkpoint reproduction.

## Five axes are real, not names added to a flat matrix

- The public Learning Studio edge currently carries a small `[T*H,W*Cin]` matrix. At this node boundary, `makeNativeVideo` reconstructs an actual `TensorIR.make([1,T,H,W,Cin], ['B','T','H','W','C'], data)` tensor and validates its axes. The external matrix layout is explicitly **THWC**, in frame-major row order.
- The independent weight input (and optional persistent trainable `Parameter` graph node) is a numerical matrix `[Cout,Kt*Kh*Kw*Cin]`. Internally it becomes a genuine rank-5 `[Cout,Kt,Kh,Kw,Cin]` tensor, with axes `['O','Kt','Kh','Kw','I']`.
- The output is computed as a true `[1,Cout,T,Hout,Wout]` semantic tensor, then mapped back to `[T*Hout,Wout*Cout]` for composable downstream microblocks. No real media codec or video file is loaded.

The **cross-correlation** is

```text
Y[o,t,y,x] = Σ_(dt,dy,dx,ci)
  X[0,t+dt-(Kt-1),y*stride+dy,x*stride+dx,ci] *
  W[o,dt,dy,dx,ci]
```

Only valid temporal indices contribute; t preceding frame 0 gives strict zero temporal padding. Time stride is exactly 1. Spatial padding is valid, no bias/dilation/groups; `Hout=floor((H-Kh)/stride)+1`, analogous for width.

## Pure stream state

`VideoConv3D.initialize(params,kernel)` creates an immutable empty state. `VideoConv3D.pushChunk(state, matrix)` takes a whole number of THWC frames and returns `{output,state:next}` without mutating the previous state. Only the latest `Kt-1` preceding frames are retained; the mathematical result of any future chunk agrees with running the entire prefix through `nativeConv3D`.

The implementation deliberately does **not** conflate this feature cache with an Attention KV Cache. Nor is the small in-memory history a GPU allocation strategy or proof of throughput gains. The last-step output must not change if any future video frame is altered.

## Independent backends and trainability

NumPy-generated standalone code uses exact index sums with a strict `if t+dt>=Kt-1` guard. PyTorch exports `F.pad(input,(0,0,0,0,Kt-1,0))` followed by real `F.conv3d`, using `[1,Cin,T,H,W]` and `[Cout,Cin,Kt,Kh,Kw]` layouts, then permuting output to the workbench carrier. The graph-connected persistent `Parameter` provides real PyTorch Autograd gradients; the JavaScript per-kind interpreter is forward-only for this node, so there is **no unsupported claim** of a generic JS Conv3D VJP.

### Evidence gates

- `tests/video_causal_conv3d_contracts.js`: exact hand-calculated kernel, rank-5 axes, strict future-frame noninterference, full-sequence versus [1,1,1,1]/[1,2,1]/[2,2] partitions, state immutability and malformed input rejection.
- `tests/video_causal_conv3d_codegen_parity.js`: compare **every graph node** across independent JavaScript, standalone NumPy and actual PyTorch CPU; independently derive the sum-of-patches weight gradient for a real PyTorch Autograd check.
- `tests/browser_multimodal.py`: real Chromium numeric execution, source inspection, NumPy/PyTorch downloaded exports and screenshot of lesson 53.

Wan2.2's real `CausalConv3d` uses asymmetric temporal padding and optional cached input; the encoder/decoder additionally have stateful residual blocks, 3D down/upsampling, a decoder, temporal chunk conventions and learned model parameters. Source: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py

PyTorch's `F.conv3d` documents the native channels-first input/weight contracts: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html

This lesson is **not** a reproduction of Wan's full VAE or its chunk-boundary / temporal subsampling implementation. A full model reconstruction must first trace the official weights, stride, normalization, padding and cache update flow through its real source.
