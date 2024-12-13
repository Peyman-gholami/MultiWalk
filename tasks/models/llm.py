from transformers import AutoModelForCausalLM

def LLM(model_name):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map = "cpu",
                )
    return model