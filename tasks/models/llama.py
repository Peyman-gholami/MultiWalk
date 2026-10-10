import torch
from torch import nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def Llama(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=16, lora_alpha=32, lora_dropout=0,
        target_modules=['q_proj', 'v_proj'], bias='none'))
    # The existing tasks compute gradients with torch.autograd.grad.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return _LoRAModel(model)


class _LoRAModel(nn.Module):
    """Expose adapters through the parameter API used by existing FL algorithms."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def named_parameters(self, *args, **kwargs):
        return ((name, p) for name, p in self.model.named_parameters(*args, **kwargs)
                if p.requires_grad)

    def forward(self, **inputs):
        return self.model(**inputs)
