# mha
Multi-head attention.

## Reference

Papers:

* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135);
* [Self-attention Does Not Need O(n2) Memory](https://arxiv.org/abs/2112.05682);
* [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867);

Implementations:

* [cudnnMultiHeadAttnForward](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnForward);
* [APEX FHMA](https://github.com/NVIDIA/apex/tree/master/apex/contrib/fmha);
* [APEX fast multihead attention](https://github.com/NVIDIA/apex/tree/master/apex/contrib/multihead_attn);
* [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention);
* [fmha\_v2](https://gitlab-master.nvidia.com/yko/fmha_v2);
* [Triton flash attention](https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py);
* [PyTorch flash attention](https://github.com/pytorch/pytorch/tree/v1.13.0/aten/src/ATen/native/transformers/cuda/flash_attn);

