import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, AutoTokenizer
import shutil
import os

def convert_apriel_to_llama(
    apriel_checkpoint="ServiceNow-AI/Apriel-5B-Base",
    output_dir="converted-llama-5b",
    torch_dtype=torch.bfloat16
):
    """
    Convert an Apriel-5B-Base model to Llama format with the same number of parameters,
    saving in bf16 and copying the Apriel tokenizer.

    Args:
        apriel_checkpoint (str): Path or Hugging Face model ID for the Apriel model.
        output_dir (str): Directory to save the converted Llama model and tokenizer.
        torch_dtype (torch.dtype): Dtype for saving the model (default: torch.bfloat16).
    """
    # Step 1: Load Apriel model and configuration
    print("Loading Apriel model and configuration...")
    apriel_config = AutoConfig.from_pretrained(apriel_checkpoint, trust_remote_code=True)
    apriel_model = AutoModelForCausalLM.from_pretrained(
        apriel_checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto"  # Optimize memory usage
    )

    # Step 2: Load Apriel tokenizer
    print("Loading Apriel tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(apriel_checkpoint, trust_remote_code=True)

    # Step 3: Create Llama configuration
    print("Creating Llama configuration...")
    llama_config = LlamaConfig(
        # Core architecture parameters
        vocab_size=apriel_config.vocab_size,                    # 131072
        hidden_size=apriel_config.hidden_size,                  # 4096
        intermediate_size=apriel_config.intermediate_size,      # 8192
        num_hidden_layers=apriel_config.num_hidden_layers,      # 28
        num_attention_heads=apriel_config.num_attention_heads,  # 24
        num_key_value_heads=apriel_config.num_key_value_heads,  # 8
        head_dim=apriel_config.head_dim,                        # 128
        
        # Normalization and initialization
        rms_norm_eps=apriel_config.rms_norm_eps,                # 1e-05
        initializer_range=apriel_config.initializer_range,      # 0.02
        
        # Attention and MLP settings
        attention_dropout=apriel_config.attention_dropout,      # 0.0
        hidden_act=apriel_config.hidden_act,                    # "silu"
        attention_bias=apriel_config.attention_bias,            # False
        mlp_bias=apriel_config.mlp_bias,                        # False
        
        # Positional embeddings
        max_position_embeddings=apriel_config.max_position_embeddings,  # 16384
        rope_theta=apriel_config.rope_theta,                    # 1000000.0
        rope_scaling={
            "type": apriel_config.rope_scaling["rope_type"],    # "yarn"
            "factor": apriel_config.rope_scaling["factor"],     # 32.0
            "original_max_position_embeddings": apriel_config.rope_scaling["original_max_position_embeddings"],  # 4096
            "beta_fast": apriel_config.rope_scaling["beta_fast"],  # 32.0
            "beta_slow": apriel_config.rope_scaling["beta_slow"],  # 1.0
            "attention_factor": apriel_config.rope_scaling["attention_factor"]  # null
        },
        
        # Model behavior
        bos_token_id=apriel_config.bos_token_id,                # 1
        eos_token_id=apriel_config.eos_token_id,                # 2
        tie_word_embeddings=apriel_config.tie_word_embeddings,  # False
        use_cache=apriel_config.use_cache,                      # True
        
        # Llama-specific settings
        _attn_implementation="eager",  # Safe default (Apriel supports eager/sdpa/flash_attention_2)
        torch_dtype="bfloat16"         # Explicitly set for config
    )

    # Step 4: Initialize Llama model
    print("Initializing Llama model...")
    llama_model = LlamaForCausalLM(config=llama_config).to(dtype=torch_dtype)

    # Step 5: Transfer weights
    print("Transferring weights from Apriel to Llama...")
    apriel_state_dict = apriel_model.state_dict()
    llama_state_dict = llama_model.state_dict()

    # Map weights (keys should match due to identical architecture)
    for key in apriel_state_dict:
        if key in llama_state_dict:
            llama_state_dict[key].copy_(apriel_state_dict[key])
        else:
            print(f"Warning: Key {key} not found in Llama model, skipping...")

    # Load the updated state dict into Llama model
    llama_model.load_state_dict(llama_state_dict, strict=True)

    # Step 6: Finalize model initialization
    print("Finalizing Llama model initialization...")
    llama_model.post_init()

    # Step 7: Save the converted model in bf16
    print(f"Saving converted Llama model to {output_dir} in bf16...")
    llama_model.save_pretrained(
        output_dir,
        torch_dtype=torch_dtype,
        safe_serialization=True  # Use safetensors for efficiency
    )
    llama_config.save_pretrained(output_dir)

    # Step 8: Copy Apriel tokenizer to output directory
    print(f"Copying Apriel tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    
    # Copy any additional tokenizer files (e.g., tokenizer.json, special_tokens_map.json)
    tokenizer_files = [
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "added_tokens.json"
    ]
    for fname in tokenizer_files:
        src_path = os.path.join(apriel_checkpoint, fname) if os.path.isdir(apriel_checkpoint) else fname
        dst_path = os.path.join(output_dir, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        elif fname in tokenizer.pretrained_vocab_files_map.get("tokenizer_file", {}):
            # Download from Hugging Face if not local
            tokenizer._download_and_save_file(
                tokenizer.pretrained_vocab_files_map["tokenizer_file"][apriel_checkpoint],
                dst_path
            )

    # Step 9: Clean up
    del apriel_model
    del llama_model
    del tokenizer
    torch.cuda.empty_cache()

    print(f"Conversion complete! Converted model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Run the conversion
    convert_apriel_to_llama(
        apriel_checkpoint="ServiceNow-AI/Apriel-5B-Base",
        output_dir="converted-llama-5b",
        torch_dtype=torch.bfloat16
    )
