# reference - https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_gptq.py#L266
from pathlib import Path
import time
import subprocess
import os

THIS = Path(".")
BASE = THIS / ".." / ".."
BASE = BASE.absolute().resolve()

MODEL = BASE / "model"
EXPORTS = BASE / "exports"

MODEL_NAME="Llama-2-7b-hf"

MODEL_PATH = MODEL / MODEL_NAME
EXPORTS = EXPORTS / MODEL_NAME

METHOD = "quant/hf"
METHOD_EXPORTS = EXPORTS / METHOD

# print paths
print("BASE:", BASE)
print("MODEL:", MODEL)
print("MODEL_PATH:", MODEL_PATH)
print("METHOD_EXPORTS:", METHOD_EXPORTS)

import torch
from transformers import BitsAndBytesConfig, AwqConfig, GPTQConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import AWQLinearVersion, ExllamaVersion, QuantizationMethod

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

quant_configs = [

    # bitsandbytes
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16), # 4bit
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), # 4bit
    BitsAndBytesConfig(load_in_8bit=True), # 8bit

    # awq
    AwqConfig(bits=4, group_size=128, zero_point=True, version=AWQLinearVersion.GEMM), # 4bit gemm
    AwqConfig(bits=4, group_size=128, zero_point=True, version=AWQLinearVersion.GEMV), # 4bit gemv

    # gptq
    GPTQConfig(bits=2, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=False), # 2bit
    GPTQConfig(bits=3, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=False), # 3bit
    GPTQConfig(bits=4, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=False), # 4bit
    GPTQConfig(bits=8, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=False), # 8bit

    # exllama v1
    GPTQConfig(bits=4, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=True, exllama_config={ "version": ExllamaVersion.ONE}), # 4bit

    # exllama v2
    GPTQConfig(bits=4, tokenizer=tokenizer, dataset="wikitext2", group_size=128, use_exllama=True, exllama_config={ "version": ExllamaVersion.TWO}), # 4bit
]

AWQ_EXPORTS=EXPORTS / "quant/awq"
GPTQ_EXPORTS=EXPORTS / "quant/gptq"

# tokenizer := tokenizer
def model_loader(quant_config):
    device_map = "cuda:0"
    model_path = MODEL_PATH

    if quant_config.quant_method == QuantizationMethod.AWQ:
        model_path = AWQ_EXPORTS / quant_config.version.name.lower()
    
    if quant_config.quant_method == QuantizationMethod.GPTQ:
        model_path = METHOD_EXPORTS / f"{quant_config.quant_method.name.lower()}_{quant_config.bits}"

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )


if __name__ == "__main__":
    
    # create export dir
    if not METHOD_EXPORTS.exists():
        METHOD_EXPORTS.mkdir(parents=True)

    # take input args bits for quantization
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="quant method to quantize to (0, 1, 2, 3, 4, 5, 6, 7, 'baseline')")
    args = parser.parse_args()


    # load model
    if args.method == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map = "cuda:0",
            torch_dtype=torch.float16,
        )
        q = "baseline"
    else: 
        # quantization config
        quant_config = quant_configs[int(args.method)]
        
        if quant_config.quant_method == QuantizationMethod.AWQ:
            q = f"{quant_config.quant_method.name.lower()}_{quant_config.bits}_{quant_config.version.name.lower()}"

        elif quant_config.quant_method == QuantizationMethod.GPTQ:
            if quant_config.use_exllama:
                q = f"{quant_config.quant_method.name.lower()}_{quant_config.bits}_exllama_version_{quant_config.exllama_config['version'].name.lower()}"
            else:
                q = f"{quant_config.quant_method.name.lower()}_{quant_config.bits}"

        elif quant_config.quant_method == QuantizationMethod.BITS_AND_BYTES:
            q = f"{quant_config.quant_method.name.lower()}_{'4' + '_' + quant_config.bnb_4bit_quant_type if quant_config.load_in_4bit else '8'}"

        model = model_loader(quant_config)

    print(f"\n\n{'=' * 35} Quant: {q} {'=' * 35}")
    prompt = "What is the meaning of life?"
    time.sleep(5)
    model.eval()
    torch.cuda.synchronize()
    # warmup
    inputs = tokenizer("Warming up", return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(inputs, max_new_tokens=128)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # benchmark
    cmd = f"nvidia-smi --format=csv,nounits --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total -lms 100 > {METHOD_EXPORTS}/gpu_bench_mem_usage_{q if q else 'baseline'}.csv"
    process = subprocess.Popen(cmd, shell=True)
    tick = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(inputs, max_new_tokens=128)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    tock = time.time()
    process.kill()
    time.sleep(5)
    os.system("killall nvidia-smi")

    print("Output:", text)
    op_tokens = 128 
    # op_tokens = len(tokenizer.tokenize(text))
    tokens_per_sec = op_tokens / (tock - tick)
    print(f"{'=' * 40} Stats {'=' * 40}\nGenerated {op_tokens} tokens in {tock-tick} seconds ({tokens_per_sec} tokens/s)\n{'=' * 80}\n\n")

    
        