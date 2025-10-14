"""
Test configuration for LLaVA with LlamaRealConfig and ReweightAttentionConfig
"""

test_llava_config = {
    "architectures": [
        "LlavaForConditionalGeneration"
    ],
    "ignore_index": -100,
    "image_token_index": 32000,
    "model_type": "llava",
    "pad_token_id": 32001,
    "projector_hidden_act": "gelu",
    "text_config": {
        "model_type": "llama_real",
        "text_config": {
            "_name_or_path": "lmsys/vicuna-7b-v1.5",
            "architectures": [
                "LlamaForCausalLM"
            ],
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "rms_norm_eps": 1e-05,
            "torch_dtype": "float16",
            "vocab_size": 32064
        },
        "additional_attention_module_config": {
            "model_type": "reweight_attention",
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "rank_dim": 16
        }
    },
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.36.0.dev0",
    "vision_config": {
        "hidden_size": 1024,
        "image_size": 336,
        "intermediate_size": 4096,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 14,
        "projection_dim": 768,
        "vocab_size": 32000
    },
    "vision_feature_layer": -2,
    "vision_feature_select_strategy": "default",
    "vocab_size": 32064
}


if __name__ == "__main__":
    import json
    from transformers import LlavaConfig

    # Register custom configs
    from src.models.llama_real.configuration_llama_real import LLamaRealConfig
    from src.models.reweighting_module.configuration_module import ReweightAttentionConfig

    print("Test LLaVA Configuration:")
    print(json.dumps(test_llava_config, indent=2))

    print("\n" + "="*80 + "\n")

    # Try to instantiate the config
    try:
        config = LlavaConfig(**test_llava_config)
        print("✓ Configuration successfully instantiated!")
        print(f"Type: {type(config)}")
        print(f"Text config type: {type(config.text_config)}")
        if hasattr(config.text_config, 'additional_attention_module_config'):
            print(f"Additional attention module type: {type(config.text_config.additional_attention_module_config)}")
    except Exception as e:
        print(f"✗ Failed to instantiate configuration: {e}")
