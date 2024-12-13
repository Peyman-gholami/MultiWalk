from transformers import AutoModelForCausalLM

def LLM(model_name):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map='auto',
                )
    return model