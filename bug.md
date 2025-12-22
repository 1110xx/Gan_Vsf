 ================================================================================
                    联合训练: Encoder + TemporalPredHead
================================================================================
Dataset: ./data/ECG5000
Device: cuda
Num nodes: 140
Subset ratio: 0.3
Hidden dim: 64
Batch size: 32
Epochs: 100
Learning rate: 0.0001
Loss weights: λ_pred=1.0, λ_recon=0.1
Pred warmup epochs: 10

Model parameters:
  Encoder: 114,113
  Pred Head: 32,285
  Decoder: 705
==================================
 File "/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/yxtang/Gan_VSF/model/pred_decoder.py", line 58, in forward
    h = self.time_proj(h)
  File "/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (286720x12 and 140x140)
