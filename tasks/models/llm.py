from transformers import AutoModelForCausalLM

def LLM():
    model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    # load_in_8bit=True,
                    # device_map='auto',
                )
    return model