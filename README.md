# mHC.cu
unofficial CUDA implementation of mHC: Manifold-Constrained Hyper-Connections by DeepseekAI

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 -DMHC_ENABLE_PDL=ON # test platform is H100 SXM5
cmake --build build -j4
```

For multi-architecture builds:
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100"
```

## Tests

### Forward Pass
```bash
./build/test_rmsnorm
./build/test_sinkhorn_knopp
./build/test_stream_mix_tc
./build/test_fused_rmsnorm_matmul
./build/test_mhc_layer
```

### Backward Pass
```bash
./build/test_rmsnorm_backward
./build/test_fused_rmsnorm_matmul_backward
./build/test_stream_aggregate_backward
./build/test_stream_distribute_backward
./build/test_stream_mix_backward
```

## Benchmarks

### Forward Pass
```bash
./build/bench_rmsnorm
./build/bench_sinkhorn_knopp
./build/bench_fused_rmsnorm_matmul
./build/bench_stream_ops
./build/bench_mhc_layer
```

### Backward Pass
```bash
./build/bench_rmsnorm_backward
./build/bench_sinkhorn_knopp_backward
./build/bench_fused_rmsnorm_matmul_backward
./build/bench_stream_ops_backward
```

## Paper

**mHC: Manifold-Constrained Hyper-Connections**  
https://arxiv.org/abs/2512.24880

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang

DeepSeek-AI

## Citation

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```
