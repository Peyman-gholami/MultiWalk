import argparse
import os
from distributed_training import DecentralizedTraining
import subprocess
import json
from torch.multiprocessing import set_start_method


# "task": "ImageNet",
# "model_name": "ResNet_EvorNorm18",
# "task": "ImageNet",
# "model_name": "ResNet_EvoNorm18",
# "task": "Cifar",
# "model_name": "VGG-11",

def load_graph_as_dict(config_file):
    # Load the configuration from the JSON file
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    # Create an empty adjacency dictionary
    adjacency_dict = {i: [] for i in range(config_data['num_nodes'])}

    # Populate the adjacency dictionary
    for edge in config_data['edges']:
        node1, node2 = edge
        adjacency_dict[node1].append(node2)
        adjacency_dict[node2].append(node1)

    return adjacency_dict, config_data.get('top_nodes', []), config_data.get('bottom_nodes', [])

def send_log_to_remote_server(log_file_path, remote_user, remote_address, remote_directory):
    remote_path = f"{remote_user}@{remote_address}:{remote_directory}"
    command = ["scp", log_file_path, remote_path]
    subprocess.run(command, check=True)


# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
# # Environment variables set by MPI
# LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
# WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
# WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

MASTER_ADDR = os.environ['MASTER_ADDR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, choices=['erdos_renyi', 'cycle', 'complete'], default='erdos_renyi', help='Graph topology')
    parser.add_argument('--tau', type=int, default=5, help='Number of SGD steps per node')
    parser.add_argument('--train_time', type=int, default=5, help='Time in minutes to run')
    parser.add_argument('--ports', type=int, nargs='+', default=[29500, 29501], help='List of ports for the groups')
    parser.add_argument('--group_names', type=str, nargs='+', default=['group1', 'group2'], help='List of group names')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for asynchronous gossip')
    parser.add_argument('--global_learning_rate', type=float, default=1.0, help='Global earning rate for federated settings')
    parser.add_argument('--algorithm', type=str, choices=['random_walk', 'async_gossip', 'async_gossip_general', 'split_random_walk', 'fedavg', 'scaffold', 'huscaffold', 'hscaffold', 'sgfocus'], required=True,
                        help='Algorithm to run')
    parser.add_argument('--split_random_walk_ratio', type=int, default=1, help='Split random walk ratio')
    parser.add_argument('--participation_rate', type=float, default=0.1, help='Fraction of clients participating in each FedAVG round')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--task', type=str, choices=['Cifar', 'MNLI'], default="Cifar", help='Task name')
    parser.add_argument('--model_name', type=str, default="ResNet20", help='Model name')
    parser.add_argument('--data_split_method', type=str, choices=['random', 'dirichlet'], default="dirichlet", help='Data split method')
    parser.add_argument('--non_iid_alpha', type=float, default=1.0, help='Non-IID alpha value')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per worker')
    parser.add_argument('--base_optimizer', type=str, default="SGD", help='Base optimizer')
    parser.add_argument('--lr_warmup_time', type=float, default=2 , help='Learning rate warmup time in minutes')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--evaluate_interval', type=int, default=10, help='Evaluation interval in seconds')
    parser.add_argument('--eval_gpu', type=int, default=0, help='GPU to use for evaluation')
    parser.add_argument('--train_eval_frac', type=float, default=.5,
                        help='Fraction of data to use for training and evaluation')
    parser.add_argument('--no_test_set_eval', action='store_true',
                        help='If provided, set no_test_set_eval to True. Default is False.')
    parser.add_argument('--no_lr_schedule', action='store_true',
                        help='If provided, set no_lr_schedule to True. Default is False.')
    parser.add_argument('--aggregator_failure_times', type=float, nargs='+', default=[],
                        help='List of times (in minutes) when aggregator node fails and needs to be replaced')

    args = parser.parse_args()

    rw_starting_ranks = list(range(len(args.ports)))
    size = WORLD_SIZE
    master_address = MASTER_ADDR
    local_rank = LOCAL_RANK
    rank = WORLD_RANK
    specific_keys = ['graph', 'train_time', 'learning_rate', 'algorithm', 'task', 'data_split_method', 'non_iid_alpha','base_optimizer', 'tau', 'split_random_walk_ratio']  # Replace these with your specific keys
    log_name = f'full_dup_size={size}_rank={rank}_rw={len(args.group_names)}' + '_'.join(
        [f'{key}={value}' for key, value in vars(args).items() if key in specific_keys])
    if args.algorithm == 'async_gossip':
        output_file = f'./configs/bipartite_{args.graph}_graph_{size}_nodes.json'
    else:
        output_file = f'./configs/{args.graph}_graph_{size}_nodes.json'
    neighbors, top_nodes, bottom_nodes = load_graph_as_dict(output_file)


    config = {
        "seed": args.seed,
        "task": args.task,
        "model_name": args.model_name,
        "split_random_walk_ratio": args.split_random_walk_ratio,
        "participation_rate": args.participation_rate,
        "data_split_method": args.data_split_method,
        "non_iid_alpha": args.non_iid_alpha,
        "batch_size": args.batch_size,
        "base_optimizer": args.base_optimizer,
        "learning_rate": args.learning_rate,
        "global_learning_rate": args.global_learning_rate,
        "lr_warmup_time": args.lr_warmup_time,
        "lr_schedule_milestones": [] if args.no_lr_schedule else [(args.train_time*60*.75, .1), (args.train_time*60*.9, .1)],
        "momentum": args.momentum,
        "weight_decay": args.weight_decay
    }

    if args.task == 'MNLI' or ('opt' in args.model_name.lower()):
        set_start_method('spawn', force=True)


    training = DecentralizedTraining(
        size=size,
        local_rank = local_rank,
        tau=args.tau,
        train_time=args.train_time,
        neighbors=neighbors,
        top_nodes=top_nodes,
        rw_starting_ranks=rw_starting_ranks,
        master_address=master_address,
        ports=args.ports,
        group_names=args.group_names,
        algorithm=args.algorithm,
        config=config,
        evaluate_interval=args.evaluate_interval,# seconds
        eval_gpu=args.eval_gpu,
        train_eval_frac=args.train_eval_frac,
        no_test_set_eval=args.no_test_set_eval,
        log_name = log_name,
        aggregator_failure_times=args.aggregator_failure_times,
    )
    training.run(rank)
    # Send log file to remote server
    log_file_path = f'./log/{log_name}'
    # send_log_to_remote_server(log_file_path, 'exouser', master_address, '~/rw_implement/log')

    # mp.set_start_method('spawn')
    # processes = []
    # for rank in range(size):
    #     p = mp.Process(target=training.run, args=(rank,))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
