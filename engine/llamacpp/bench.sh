METHOD=engine/llamacpp
METHOD_PATH=$METHOD/llama.cpp

MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=model/$MODEL_NAME
EXPORTS=exports/$MODEL_NAME
METHOD_EXPORTS=$EXPORTS/$METHOD

mkdir -p $METHOD_EXPORTS

# Function to start memory tracker
start_mem_tracker() {
    nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > $METHOD_EXPORTS/gpu_bench_mem_usage_$1.csv &
}

# List of model types
MODEL_GGUF=ggml-model-f16
MODEL_Q4_0_GGUF=ggml-model-q4_0
MODEL_Q8_0_GGUF=ggml-model-q8_0
MODEL_Q2_K_GGUF=ggml-model-q2_k
MODEL_Q4_K_M_GGUF=ggml-model-q4_k_m
MODEL_Q4_K_S_GGUF=ggml-model-q4_k_s
MODEL_AWQ_GGUF=ggml-model-f32-awq4g128
MODEL_AWQ_Q4_0_GGUF=ggml-model-awq_q4_0

# Loop through each model type
for MODEL_TYPE in "$MODEL_GGUF" "$MODEL_Q4_0_GGUF" "$MODEL_Q8_0_GGUF" "$MODEL_Q2_K_GGUF" "$MODEL_Q4_K_M_GGUF" "$MODEL_Q4_K_S_GGUF" "$MODEL_AWQ_GGUF" "$MODEL_AWQ_Q4_0_GGUF"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE.gguf

    start_mem_tracker $MODEL_TYPE

    # Run llama-bench for the current model type
    $METHOD_PATH/llama-bench -m $MODEL_PATH -ngl 999 -t 32 -b 1 -p 128 -n 128 2>&1 | tee $METHOD_EXPORTS/bench_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    sleep 2
    killall nvidia-smi
    echo "Benchmark for $MODEL_TYPE complete"
    sleep 5
done


FILE_PATH=datasets/wikitext-2-raw
# Loop through each model type
for MODEL_TYPE in "$MODEL_GGUF" "$MODEL_Q4_0_GGUF" "$MODEL_Q8_0_GGUF" "$MODEL_Q2_K_GGUF" "$MODEL_Q4_K_M_GGUF" "$MODEL_Q4_K_S_GGUF" "$MODEL_AWQ_GGUF" "$MODEL_AWQ_Q4_0_GGUF"; do
    MODEL_PATH=$METHOD_EXPORTS/$MODEL_TYPE.gguf

    start_mem_tracker $MODEL_TYPE

    # Run llama-bench for the current model type
    $METHOD_PATH/perplexity -m $MODEL_PATH -ngl 999 -t 32 -f $FILE_PATH/wiki.test.raw 2>&1 | tee $METHOD_EXPORTS/bench_ppl_$MODEL_TYPE.log

    # Stop memory tracker for the current model type
    sleep 2
    killall nvidia-smi
    echo "PPL for $MODEL_TYPE complete"
    sleep 5
done



# usage: ./llama-bench [options]
# options:
#   -h, --help
#   -m, --model <filename>            (default: models/7B/ggml-model-q4_0.gguf)
#   -p, --n-prompt <n>                (default: 512)
#   -n, --n-gen <n>                   (default: 128)
#   -b, --batch-size <n>              (default: 512)
#   --memory-f32 <0|1>                (default: 0)
#   -t, --threads <n>                 (default: 16)
#   -ngl N, --n-gpu-layers <n>        (default: 99)
#   -mg i, --main-gpu <i>             (default: 0)
#   -mmq, --mul-mat-q <0|1>           (default: 1)
#   -ts, --tensor_split <ts0/ts1/..>
#   -r, --repetitions <n>             (default: 5)
#   -o, --output <csv|json|md|sql>    (default: md)
#   -v, --verbose                     (default: 0)


# usage: ./perplexity [options]

# options:
#   -h, --help            show this help message and exit
#       --version         show version and build info
#   -i, --interactive     run in interactive mode
#   --interactive-first   run in interactive mode and wait for input right away
#   -ins, --instruct      run in instruction mode (use with Alpaca models)
#   -cml, --chatml        run in chatml mode (use with ChatML-compatible models)
#   --multiline-input     allows you to write or paste multiple lines without ending each in '\'
#   -r PROMPT, --reverse-prompt PROMPT
#                         halt generation at PROMPT, return control in interactive mode
#                         (can be specified more than once for multiple prompts).
#   --color               colorise output to distinguish prompt and user input from generations
#   -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
#   -t N, --threads N     number of threads to use during generation (default: 6)
#   -tb N, --threads-batch N
#                         number of threads to use during batch and prompt processing (default: same as --threads)
#   -p PROMPT, --prompt PROMPT
#                         prompt to start generation with (default: empty)
#   -e, --escape          process prompt escapes sequences (\n, \r, \t, \', \", \\)
#   --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
#   --prompt-cache-all    if specified, saves user input and generations to cache as well.
#                         not supported with --interactive or other interactive options
#   --prompt-cache-ro     if specified, uses the prompt cache but does not update it.
#   --random-prompt       start with a randomized prompt.
#   --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string
#   --in-prefix STRING    string to prefix user inputs with (default: empty)
#   --in-suffix STRING    string to suffix after user inputs with (default: empty)
#   -f FNAME, --file FNAME
#                         prompt file to start generation.
#   -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
#   -c N, --ctx-size N    size of the prompt context (default: 512, 0 = loaded from model)
#   -b N, --batch-size N  batch size for prompt processing (default: 512)
#   --samplers            samplers that will be used for generation in the order, separated by ';', for example: "top_k;tfs;typical;top_p;min_p;temp"
#   --sampling-seq        simplified sequence for samplers that will be used (default: kfypmt)
#   --top-k N             top-k sampling (default: 40, 0 = disabled)
#   --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
#   --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
#   --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
#   --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
#   --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
#   --repeat-penalty N    penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)
#   --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
#   --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
#   --mirostat N          use Mirostat sampling.
#                         Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
#                         (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
#   --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
#   --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
#   -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
#                         modifies the likelihood of token appearing in the completion,
#                         i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
#                         or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
#   --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)
#   --grammar-file FNAME  file to read grammar from
#   --cfg-negative-prompt PROMPT
#                         negative prompt to use for guidance. (default: empty)
#   --cfg-negative-prompt-file FNAME
#                         negative prompt file to use for guidance. (default: empty)
#   --cfg-scale N         strength of guidance (default: 1.000000, 1.0 = disable)
#   --rope-scaling {none,linear,yarn}
#                         RoPE frequency scaling method, defaults to linear unless specified by the model
#   --rope-scale N        RoPE context scaling factor, expands context by a factor of N
#   --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
#   --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N
#   --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)
#   --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
#   --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
#   --yarn-beta-slow N    YaRN: high correction dim or alpha (default: 1.0)
#   --yarn-beta-fast N    YaRN: low correction dim or beta (default: 32.0)
#   --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
#   --no-penalize-nl      do not penalize newline token
#   --temp N              temperature (default: 0.8)
#   --logits-all          return logits for all tokens in the batch (default: disabled)
#   --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f
#   --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: 400)
#   --keep N              number of tokens to keep from the initial prompt (default: 0, -1 = all)
#   --draft N             number of tokens to draft for speculative decoding (default: 8)
#   --chunks N            max number of chunks to process (default: -1, -1 = all)
#   -np N, --parallel N   number of parallel sequences to decode (default: 1)
#   -ns N, --sequences N  number of sequences to decode (default: 1)
#   -pa N, --p-accept N   speculative decoding accept probability (default: 0.5)
#   -ps N, --p-split N    speculative decoding split probability (default: 0.1)
#   -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)
#   --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md
#   --image IMAGE_FILE    path to an image file. use with multimodal models
#   --mlock               force system to keep model in RAM rather than swapping or compressing
#   --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
#   --numa                attempt optimizations that help on some NUMA systems
#                         if run without this previously, it is recommended to drop the system page cache before using this
#                         see https://github.com/ggerganov/llama.cpp/issues/1437
#   -ngl N, --n-gpu-layers N
#                         number of layers to store in VRAM
#   -ngld N, --n-gpu-layers-draft N
#                         number of layers to store in VRAM for the draft model
#   -ts SPLIT --tensor-split SPLIT
#                         how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1
#   -mg i, --main-gpu i   the GPU to use for scratch and small tensors
#   -nommq, --no-mul-mat-q
#                         use cuBLAS instead of custom mul_mat_q CUDA kernels.
#                         Not recommended since this is both slower and uses more VRAM.
#   -gan N, --grp-attn-n N
#                         group-attention factor (default: 1)
#   -gaw N, --grp-attn-w N
#                         group-attention width (default: 512.0)
#   --verbose-prompt      print prompt before generation
#   -dkvc, --dump-kv-cache
#                         verbose print of the KV cache
#   -nkvo, --no-kv-offload
#                         disable KV offload
#   -ctk TYPE, --cache-type-k TYPE
#                         KV cache data type for K (default: f16)
#   -ctv TYPE, --cache-type-v TYPE
#                         KV cache data type for V (default: f16)
#   --simple-io           use basic IO for better compatibility in subprocesses and limited consoles
#   --lora FNAME          apply LoRA adapter (implies --no-mmap)
#   --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)
#   --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
#   -m FNAME, --model FNAME
#                         model path (default: models/7B/ggml-model-f16.gguf)
#   -md FNAME, --model-draft FNAME
#                         draft model for speculative decoding
#   -ld LOGDIR, --logdir LOGDIR
#                         path under which to save YAML logs (no logging if unset)
#   --override-kv KEY=TYPE:VALUE
#                         advanced option to override model metadata by key. may be specified multiple times.
#                         types: int, float, bool. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
#   -stc N --print-token-count N
#                         print token count every N tokens (default: -1)
