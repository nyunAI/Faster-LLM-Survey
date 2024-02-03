# Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward

This repository contains the open-source code and benchmark results for the paper - Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward.<br>
The benchmark assesses the performance of various compression and inference methods.

> [**Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward**]()<br>
> [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Raghav Magazine](), [Shubham Kushwaha](https://linkedin.com/in/shwoobham), [Deepak Gupta](https://dkgupta90.github.io/), [Merouane Debbah]()<br>Nyun AI, Transmute AI Lab, KU 6G Research Cente




<!-- ## Updates
### <Month> <Date>, <Year> : <Title> -->

## Getting Started

All the experiments are performed in isolated Python3.10 environments with method-specific requirements such as package & library versions. The exact repository and branch details can be inferred from [.gitmodules](.gitmodules)

## Repository Organization

The repository follows a structured format with a branch naming convention of "A100\<method>", where \<method> denotes the specific evaluation method. The organization within each branch is outlined as follows:

- **engine/:** This directory contains the implementation of engine methods along with setup, generation, and bench scripts.
  
- **prune/:** Here, you'll find the implementation of pruning methods along with associated setup, generation, and bench scripts.

- **quant/:** This directory houses the implementation of quantization methods, complete with setup, generation, and bench scripts.

- **exports/:** A shared export folder is structured as `exports/MODEL_NAME/METHOD_TYPE/METHOD_NAME/`.

- **experiments/:** This section holds formatted benchmarking results in notebooks, featuring metrics such as RM (run memory), WM (weight memory), and various GPU utilization graphs.

Please note that all setup, generation, and benchmarking scripts (.sh) strive to be kept up-to-date with the latest runs and are tailored to Python 3.10 with CUDA 12.1 (or the version necessary for the method). Adjustments to the scripts may be required, or a different script should be utilized.

## Branch Overview

Discover various branches dedicated to evaluated methods within the repository:

- [A100Exllama](../../tree/A100Exllama): Investigates the Exllama engine with GPTQ quantization.
  
- [A100Exllamav2](../../tree/A100Exllamav2): Explores the latest ExllamaV2 featuring EXL2 & GPTQ quantizations.

- [A100Llamacpp](../../tree/A100Llamacpp): Examines the CPP implementation of the Llama architecture for enhanced speed.

- [A100MLCLLM](../../tree/A100MLCLLM): Explores MLCLLM, offering extensive hardware and platform support.

- [A100TGI](../../tree/A100TGI): Investigates the Text Generation Inference toolkit, employed for LLM inferences in production.

- [A100VLLM](../../tree/A100VLLM): Explores VLLM.

- [A100TensorRTLLM](../../tree/A100TensorRTLLM): Investigates NVIDIA's TensorRTLLM inference engine.

- [A100GPTQ](../../tree/A100GPTQ): Explores the GPTQ quantization method through AutoGPTQ.

- [A100HF](../../tree/A100HF): Investigates multiple quantization methods, alongside baseline generation speeds for each method.

- [A1000mniquant](../../tree/A1000mniquant): Explores the OmniQuant quantization method.

Note: Each branch is equipped with its own set of updated scripts, which may or may not be synchronized with other branches. Additionally, specific quantization methods might lack dedicated branches; however, the corresponding scripts can be directly referenced in the respective branches or from the main branch. Models and scales directly obtained from the HF Hub were also utilized as needed.

## Results Overview

### Pruning

| Method           | Sparsity | RM (GB) | WM (GB) | Tokens/s | Perplexity |
|------------------|----------|---------|---------|----------|------------|
| Baseline         | -        | 26.16   | 12.55   | 30.90    | 12.62      |
| Wanda-SP         | 20%      | -       | -       | -        | 22.12      |
| Wanda-SP         | 50%      | -       | -       | -        | 366.43     |
| LLM-Pruner       | 20%      | 10.38   | 10.09   | 32.57    | 19.77      |
| LLM-Pruner       | 50%      | 6.54    | 6.23    | 40.95    | 112.44     |
| LLM-Pruner*      | 20%      | 10.38   | 10.09   | 32.57    | 17.37      |
| LLM-Pruner*      | 50%      | 6.54    | 6.23    | 40.95    | 38.12      |
| FLaP             | 20%      | 9.72    | 9.44    | 33.90    | 14.62      |
| FLaP             | 50%      | 6.26    | 6.07    | 42.88    | 31.80      |

\* with fine-tuning

<!-- **Analysis:**
- The baseline model has a sparsity level of 0%.
- Wanda-SP introduces sparsity at 20% and 50%, significantly impacting perplexity.
- LLM-Pruner achieves sparsity of 20% and 50% with variations in running and weight memory.
- FLaP demonstrates sparsity at 20% and 50%, influencing both memory and tokens/s metrics. -->

### Quantization

| Method                   | Inference Engine | WM (GB) | RM (GB) | Tokens/s | Perplexity |
|--------------------------|-------------------|---------|---------|----------|------------|
| Baseline FP16            | PyTorch           | 12.55   | 26.16   | 30.90    | 5.85       |
| GPTQ 2bit                | PyTorch           | 2.11    | 2.98    | 20.91    | NaN        |
| GPTQ 3bit                | PyTorch           | 2.87    | 3.86    | 21.24    | 7.36       |
| GPTQ 4bit                | PyTorch           | 3.63    | 4.65    | 21.63    | 6.08       |
| GPTQ 8bit                | PyTorch           | 6.67    | 7.62    | 21.36    | 5.86       |
| AWQ 4bit GEMM            | PyTorch           | 3.68    | 4.64    | 28.51    | 6.02       |
| AWQ 4bit GEMV            | PyTorch           | 3.68    | 4.64    | 31.81    | 6.02       |
| QLoRA (NF4)              | PyTorch           | 3.56    | 4.84    | 19.70    | 6.02       |
| LLM.int8()               | PyTorch           | 6.58    | 7.71    | 5.24     | 5.89       |
| K-Quants 4bit            | Llama.cpp         | 3.80    | 7.38    | 104.45   | 5.96       |
| OmniQuant 3bit           | MLC-LLM           | 3.20    | 5.10    | 83.4     | 6.65       |
| OmniQuant 4bit           | MLC-LLM           | 3.80    | 5.70    | 134.2    | 5.97       |

<!-- **Analysis:**
- Baseline FP16 serves as the reference for comparison.
- GPTQ 2bit exhibits lower memory consumption but leads to perplexity issues.
- Various quantization methods show diverse impacts on memory, tokens/s, and perplexity.
- LLM.int8() significantly reduces memory but at the expense of tokens/s.
- K-Quants 4bit in Llama.cpp achieves a balance between memory and tokens/s.
- OmniQuant methods in MLC-LLM show competitive performance in terms of memory and tokens/s. -->

### Engine Results

| Method            | Hardware Support    | Quantization Type       | WM (GB) | RM (GB) | Tokens/s | Perplexity |
|-------------------|---------------------|-------------------------|---------|---------|----------|------------|
| Llama.cpp         | NVIDIA GPU          | GGUF K-Quant 2bit       | 2.36    | 3.69    | 102.15   | 6.96       |
|                   | AMD GPU             | GGUF 4bit               | 3.56    | 4.88    | 128.97   | 5.96       |
|                   | Apple Silicon       | GGUF AWQ 4bit           | 3.56    | 4.88    | 129.25   | 5.91       |
|                   | CPU                 | GGUF K-Quant 4bit       | 3.59    | 4.90    | 109.72   | 5.87       |
|                   |                     | GGUF 8bit               | 6.67    | 7.78    | 93.39    | 5.79       |
|                   |                     | GGUF FP16               | 12.55   | 13.22   | 66.81    | 5.79       |
| ExLlama           | NVIDIA GPU          | GPTQ 4bit               | 3.63    | 5.35    | 77.10    | 6.08       |
|                   | AMD GPU             |                         |         |         |          |            |
| ExLlamav2         | NVIDIA GPU          | EXL2 2bit               | 2.01    | 5.21    | 153.75   | 20.21      |
|                   | AMD GPU             | EXL2 4bit               | 3.36    | 6.61    | 131.68   | 6.12       |
|                   |                     | GPTQ 4bit               | 3.63    | 6.93    | 151.30   | 6.03       |
|                   |                     | EXL2 8bit               | 6.37    | 9.47    | 115.81   | 5.76       |
|                   |                     | FP16                    | 12.55   | 15.09   | 67.70    | 5.73       |
| vLLM              | NVIDIA GPU          | AWQ GEMM 4bit           | 3.62    | 34.55   | 114.43   | 6.02       |
|                   | AMD GPU             | GPTQ 4bit               | 3.63    | 36.51   | 172.88   | 6.08       |
|                   |                     | FP16                    | 12.55   | 35.92   | 79.74    | 5.85       |
| TensorRT-LLM      | NVIDIA GPU          | AWQ GEMM 4bit           | 3.42    | 5.69    | 194.86   | 6.02       |
|                   |                     | GPTQ 4bit               | 3.60    | 5.88    | 202.16   | 6.08       |
|                   |                     | INT8                    | 6.53    | 8.55    | 143.57   | 5.89       |
|                   |                     | FP16                    | 12.55   | 14.61   | 83.43    | 5.85       |
| TGI               | AMD GPU             | AWQ GEMM 4bit           | 3.62    | 7.97    | 30.80    | 6.02       |
|                   | NVIDIA GPU          | AWQ GEMV 4bit           | 3.62    | 7.96    | 34.22    | 6.02       |
|                   | Intel GPU           | GPTQ 4bit               | 3.69    | 39.39   | 34.86    | 6.08       |
|                   | AWS Inferentia2     | FP4                     | 12.55   | 17.02   | 34.38    | 6.15       |
|                   |                     | NF4                     | 12.55   | 17.02   | 33.93    | 6.02       |
|                   |                     | INT8                    | 12.55   | 11.66   | 5.39     | 5.89       |
|                   |                     | FP16                    | 12.55   | 17.02   | 34.23    | 5.85       |
| MLC-LLM           | NVIDIA GPU          | OmniQuant 3bit          | 3.2     | 5.1     | 83.4     | 6.65       |
|                   | AMD GPU, CPU, WebGPU| OmniQuant 4bit          | 3.8     | 5.7     | 134.2    | 5.97       |
|                   | Apple Silicon, Intel GPU, WASM, Adreno Mali | FP16 | 12.55   | 15.38   | 87.37    | 5.85       |

<!-- **Analysis:**
- Each method exhibits varying performance metrics across different hardware and quantization types.
- Llama.cpp with GGUF K-Quant 2bit shows competitive tokens/s and perplexity.
- ExLlamav2 with EXL2 2bit achieves high tokens/s but at the expense of increased memory consumption.
- TensorRT-LLM with INT8 quantization stands out for high tokens/s and relatively lower memory usage.
- TGI on AMD GPU demonstrates lower tokens/s compared to other hardware setups.
- MLC-LLM showcases promising results across diverse hardware, especially with OmniQuant 4bit quantization. -->

<!-- ## Citation

If you find our project is helpful, please feel free to leave a star and cite our paper:
```BibTeX
@misc{chavan2023oneforall,
      title={Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward}, 
      author={Arnav Chavan and Zhuang Liu and Deepak Gupta and Eric Xing and Zhiqiang Shen},
      year={2023},
      eprint={2306.07967},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` -->
    

### Acknowledgements

We extend our gratitude to the following repositories and sources for providing essential methods, engines, and datasets utilized in our benchmarking project:

1. [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) - Hugging Face model repository for Llama-2-7b.
2. [llama.cpp](https://github.com/ggerganov/llama.cpp) - Source for llama.cpp, a key engine method used in our benchmarks.
3. [exllama](https://github.com/turboderp/exllama) - Repository for the ExLlama engine method.
4. [exllamav2](https://github.com/turboderp/exllamav2) - Source for ExLlamaV2 engine method.
5. [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) - Alpaca dataset on Hugging Face, utilized in our benchmarks.
6. [squeezellm](https://github.com/SqueezeAILab/SqueezeLLM) - Repository for SqueezeLLM quantization method.
7. [squeezellmgradients](https://github.com/kssteven418/SqueezeLLM-gradients.git) - Repository for SqueezeLLM-gradients.
8. [omniquant](https://github.com/OpenGVLab/OmniQuant.git) - Source for OmniQuant quantization method.
9. [mlcllm](https://github.com/mlc-ai/mlc-llm.git) - Repository for the MLC-LLM engine method.
10. [llmpruner](https://github.com/horseee/LLM-Pruner.git) - Source for LLM-Pruner pruning method.
11. [tensorrtllm](https://github.com/NVIDIA/TensorRT-LLM.git) - Source for TensorRT-LLM engine method (branch: release/0.5.0).
12. [autogptq](https://github.com/AutoGPTQ/AutoGPTQ) - Repository for AutoGPTQ, offering quantization package based on the GPTQ algorithm.
13. [autoawq](https://github.com/casper-hansen/AutoAWQ) - Repository for AutoAWQ, implementing the AWQ algorithm for 4-bit quantization.
14. [vllm](https://github.com/vllm-project/vllm) - Source for vllm package offering the inference and serving engine

These resources have been instrumental in conducting the benchmarks and evaluations. We appreciate the creators and maintainers of these repositories for their valuable contributions.
