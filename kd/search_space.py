import json
import logging

logger = logging.getLogger(__name__)

# def get_search_space_auto(teacher_model):
#     """ Automatically generates search space based on teacher config and predefined ratios. """
#     teacher_config = teacher_model.config
#     logger.info("Generating search space automatically based on teacher config...")

#     # Define ratios
#     layer_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0] # Added 0.75
#     head_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0]  # Added 0.75
#     intermediate_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0]

#     # --- Attention Heads ---
#     attn_heads = getattr(teacher_config, 'num_attention_heads', 12) # Default 12 if not found
#     num_attention_heads = [max(1, int(attn_heads * ratio)) for ratio in head_ratios]
#     num_attention_heads = sorted(list(set(num_attention_heads)))

#     # --- Hidden Layers ---
#     hidden_layers = getattr(teacher_config, 'num_hidden_layers', 12) # Default 12 if not found
#     num_hidden_layers = [max(1, int(hidden_layers * ratio)) for ratio in layer_ratios]
#     num_hidden_layers = sorted(list(set(num_hidden_layers)))

#     # --- Hidden Size ---
#     # User's logic: base on top 2 head counts * 64
#     # Let's make it slightly more robust: allow hidden sizes divisible by *any* head option
#     h_size_options = set()
#     default_head_dim = 64 # Common default
#     teacher_head_dim = getattr(teacher_config, 'hidden_size', 768) // attn_heads if attn_heads > 0 else default_head_dim
    
#     for h in num_attention_heads:
#         # Option 1: Keep teacher's head dimension
#         h_size_options.add(h * teacher_head_dim)
#         # Option 2: Use default head dimension (e.g., 64)
#         h_size_options.add(h * default_head_dim)

#     # Filter for reasonable sizes (e.g., >= 128)
#     hidden_size = sorted([hs for hs in h_size_options if hs >= 128])
#     if not hidden_size: # Fallback if filtering removed everything
#         hidden_size = [max(128, num_attention_heads[-1] * default_head_dim)]


#     # --- Intermediate Size ---
#     h_size = getattr(teacher_config, 'hidden_size', 768) # Default 768
#     i_size = getattr(teacher_config, 'intermediate_size', h_size * 4)
#     intermediate_size = [max(128, int(i_size * ratio)) for ratio in intermediate_ratios]
#     intermediate_size = sorted(list(set(intermediate_size)))

#     # --- Activation Function ---
#     hidden_act = ['gelu', 'relu', 'silu'] # Standard options

#     search_space = {
#         "num_hidden_layers": num_hidden_layers,
#         "num_attention_heads": num_attention_heads,
#         "hidden_size": hidden_size,
#         "intermediate_size": intermediate_size,
#         "hidden_act": hidden_act
#     }
#     logger.info(f"Generated Search Space: {json.dumps(search_space, indent=2)}")
#     return search_space

def get_search_space_auto(teacher_model):
    return {
        "num_hidden_layers": [3, 4, 6, 10, 12],
        "num_attention_heads": [2, 3, 4, 6, 12],
        "hidden_size": [384, 768],
        "intermediate_size": [384, 512, 576, 768, 1024, 1536, 2048, 3072],
        "hidden_act": ['gelu', 'relu', 'silu']
    }
