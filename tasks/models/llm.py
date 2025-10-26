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
    # Check if model has transformer layers
    if not hasattr(model, 'transformer') and not hasattr(model, 'model'):
        raise ValueError("Model does not have expected transformer structure")
    
    # Get the transformer part of the model
    transformer = getattr(model, 'transformer', None) or getattr(model, 'model', None)
    if not transformer:
        raise ValueError("Could not find transformer component in model")
    
    # Check if it has layers attribute
    if not hasattr(transformer, 'h') and not hasattr(transformer, 'layers'):
        raise ValueError("Transformer does not have expected layers structure")
    
    # Get the layers
    layers = getattr(transformer, 'h', None) or getattr(transformer, 'layers', None)
    if not layers:
        raise ValueError("Could not find layers in transformer")
    
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