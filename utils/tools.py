import torch
import numpy as np

def load_from_shared(parameters, shared_parameters, device):
    for param, shared_param in zip(parameters, shared_parameters):
        shared_param_data = np.frombuffer(shared_param.get_obj(), dtype=np.float32).reshape(param.shape)
        param.data = torch.from_numpy(shared_param_data).to(device)
    return parameters

def while_till(comm_process_started, ):
    while comm_process_started.value == self.parent.size:  # wait for communication process to start
        logging.info(f"[Computation] wait at Rank {rank}")
        time.sleep(0.5)