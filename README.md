# A Tale of Two Learning Algorithms: Multiple Stream Random Walk and Asynchronous Gossip



Abstract: 
In decentralized learning, both gossip-based and random walk-based approaches are widely studied. However, a comprehensive understanding of their performance and differences across various settings remains incomplete. In this paper, we aim to bridge this gap. We first introduce a new algorithm called Asynchronous Multi-walk (MW), a novel asynchronous decentralized learning method that improves the convergence rate of random walk approaches by introducing multiple walks within the network. By incorporating more walks, MW achieves faster convergence, albeit with increased communication and computation overhead. Then we provide a convergence analysis for MW and Asynchronous Gossip with respect to iterations (computation), clock time, and communication.

Our results show that, in terms of iterations, MW outperforms Asynchronous Gossip in graph topologies with larger diameters (such as cycles) and at the same time reduces communication overhead. Conversely, for graph topologies with smaller diameters (like complete graphs), this no longer holds, and Asynchronous Gossip can be faster. When considering clock time, since multiple iterations in Asynchronous Gossip or MW with multiple walks are executed simultaneously, we observe a linear speed-up with the number of nodes in Asynchronous Gossip and the number of walks in MW. When evaluating convergence with respect to communication overhead, we have shown that MW with one walk is superior to both MW with multiple walks and Asynchronous Gossip. This highlights the effectiveness of each algorithm in different scenarios.

## Assumed environment
This project uses the official PyTorch Docker image to ensure consistency and compatibility with the required dependencies.

The specific Docker image used is:
```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```
### Requirements

This project relies on several Python libraries to work with large language models. Below are the required libraries and their latest versions:

#### Python Libraries

- `transformers==4.47.1` - For working with pretrained language models.
- `datasets==2.14.5` - To manage and preprocess datasets for machine learning tasks.
- `accelerate==0.22.0` - For optimizing training and inference on multi-GPU setups.
- `bitsandbytes==0.45.0` - For efficient low-precision (8-bit/4-bit) optimizations.

#### Installation

To install these specific versions, use the following command:

```bash
pip install transformers==4.47.1 datasets==2.14.5 accelerate==0.22.0 bitsandbytes==0.45.0
````
## Lunch decentralized multi-node learning
There are different ways to “launch” algorithms in a distributed fashion across multiple nodes. Here are two ways you can do it:

### MPI
1- Passwordless SSH setup: Use **setup_ssh_leader.sh** to set up a paswordless ssh among nodes and update nodes' information with user@IP address (like root@192.168.1.101)
and password.
```bash
MEMBER_NODES=(
    ["member1"]="password1"
    ["member2"]="password2"
    ["member3"]="password3"
) 
```
2- Lunch the training: Use **MPI command** in the master node and after updating ```MASTER_ADDR```and other variables related to the training. You alos need to add all the nodes to **hostfile**. 

### Kubernetes
You can set up the job for both image classification and LLM task using ***cifar_job.yaml*** and ***mnli_job.yaml***, respectively. Please do the required adjustments like the number of nodes, requested resources for each node, and the training variables.
## Code Organization

- The starting point for our code execution is ***main.py***.
- You need to start ***main.py*** on different nodes. This could be through MPI, Kubernetes, or other ways as described above.
- It can run different experiments based on its global ```config``` variable.
- All the hyperparameters and configs used in our experiments are listed in the paper and  ***cifar_job.yaml*** and ***mnli_job.yaml*** for different tasks.
- The algorithms are implemented in ***distributed_training.py***.

