import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_model(model_path, quant_path, quant_config):
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Quantize a causal language model.")

    # Add command line arguments
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained language model")
    parser.add_argument("--quant_path", type=str, help="Path to save the quantized model")
    parser.add_argument("--zero_point", action="store_true", help="Enable zero point quantization")
    parser.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--w_bit", type=int, default=4, help="Number of bits for weight quantization")
    parser.add_argument("--version", type=str, default="GEMM", help="Quantization version")

    # Parse command line arguments
    args = parser.parse_args()

    # Construct quantization config
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    # Call the function to quantize the model
    quantize_model(args.model_path, args.quant_path, quant_config)
