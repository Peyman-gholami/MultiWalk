from transformers import AutoModelForCausalLM

def LLM(model_name):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                )
    for param in model.parameters():
        param.requires_grad = True
    return model