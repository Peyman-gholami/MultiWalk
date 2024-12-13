from transformers import AutoModelForCausalLM

def LLM(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map = device,
                )
    return model