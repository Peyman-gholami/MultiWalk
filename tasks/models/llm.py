from transformers import AutoModelForCausalLM
from torch import nn

def LLM(model_name):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                )
    return model


def get_resnet_separation_point(model: nn.Module) -> int:
    """
    Given an LLM model with 12 layers, return the separation point index
    in the flattened parameter list such that separation_point + 1 is the index of the first
    parameter belonging to the 11th layer (0-indexed layer 10).
    
    This separates the model into:
    - First part: layers 0-9 (first 10 layers)
    - Second part: layers 10-11 (last 2 layers)
    
    This uses parameter identity rather than names, so it works regardless of wrappers that
    may alter parameter names (e.g., DataParallel).
    """
    layers = None
    
    # Try different common transformer architectures
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT model structure: model.decoder.layers
        layers = model.model.decoder.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style: transformer.h
        layers = model.transformer.h
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        # Some models: transformer.layers
        layers = model.transformer.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Some models: model.layers
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        # Direct layers attribute
        layers = model.layers
    
    if layers is None:
        raise ValueError("Model does not have expected transformer layers structure")
    
    # Check if we have at least 12 layers
    if len(layers) < 12:
        raise ValueError(f"Model has {len(layers)} layers, but expected at least 12")
    
    # Get parameters of the 11th layer (0-indexed layer 10)
    layer_10_param_ids = {id(p) for p in layers[10].parameters()}
    
    # Find the first parameter of layer 10 in the model's parameter list
    first_layer_10_idx = None
    for idx, p in enumerate(model.parameters()):
        if id(p) in layer_10_param_ids:
            first_layer_10_idx = idx
            break
    
    if first_layer_10_idx is None:
        raise ValueError("Could not locate parameters of layer 10 within model.parameters() order")
    
    separation_point = max(0, first_layer_10_idx - 1)
    return separation_point