# Marlin CUDA Kernel Credits

This CUDA kernel implementation is derived from the Marlin project.

## Source

- **Repository:** https://github.com/IST-DASLab/marlin
- **License:** Apache-2.0
- **Original Files:**
  - `marlin_cuda_kernel.cu`
  - `marlin_cuda.cpp`

## Citation

If you use this kernel, please cite:

```bibtex
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}
```

## Original Authors

- Elias Frantar (elias.frantar@ist.ac.at)
- Roberto L. Castro
- Jiale Chen
- Torsten Hoefler
- Dan Alistarh

## About Marlin

Marlin is an FP16xINT4 LLM inference kernel that achieves near-ideal ~4x speedups
up to medium batch sizes of 16-32 tokens. It is optimized for NVIDIA GPUs with
compute capability 8.0 or higher (Ampere, Ada Lovelace, Hopper architectures).

## Modifications for Keras Integration

- Integrated into Keras quantization infrastructure
- Added Python bindings compatible with Keras ops
- Wrapped with Keras-style API for seamless layer integration
