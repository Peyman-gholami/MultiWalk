# FedSUM-CR

Implements Algorithm 3, Section 3.3 of [You and Pu (2025), FedSUM Family](https://arxiv.org/html/2512.18275v1#S3.SS3).

Select it with `--algorithm fedsum_cr`. Rank 0 aggregates on CPU; ranks 1 through
`WORLD_SIZE - 1` train. Tasks, data partitions, models, GPU assignment, evaluation,
participation patterns and logging use the existing interfaces.

## Learning rates and state

`--learning_rate` is the paper's local learning rate `eta_l`,
`--global_learning_rate` is `eta_g`, `--tau` is `K`, and `N = WORLD_SIZE - 1`.
The local SGD step is **eta_l / N**, not eta_l. The server step is
**eta_g * eta_l * K / N** times its persistent control sum.

Each client initializes `h = 0`, `z = x_initial`, `a = -1`. On selected round `t`:

1. Receive `x` and reconstruct `c = N * (z - x) / (eta_g * eta_l * K * (t - a)) - h`.
2. Run `K` minibatches through the same `configure_base_optimizer` and
   `parent.local_sgd` path as HScaffold. The optimizer step uses `eta_l / N`,
   followed by a separate `x_local -= (eta_l / N) * c` correction. With plain
   SGD this is `x_local -= (eta_l / N) * (gradient + c)`, as in Algorithm 3.
3. Compute `h_new = N * (x - x_local) / (eta_l * K) - c`.
4. Send `delta = h_new - h`; retain `h = h_new`, `z = x`, `a = t`.

The server initializes `y = 0`, then updates `y += sum(delta)` and
`x -= (eta_g * eta_l * K / N) * y`. It sums deltas without averaging over
selected clients, retains controls across inactive rounds, and still updates
the model from stored `y` if nobody participates. Inactive clients retain all
three state variables. Round numbers advance even on skipped rounds.

Only a model-sized tensor is downloaded and a model-sized delta is uploaded for
each selected client, plus the existing integer participation notifications.
The server streams deltas into `y`; it does not store a model for every client.

## Optimizer options

For Algorithm 3's plain SGD updates, pass:

```sh
--algorithm fedsum_cr --base_optimizer SGD --momentum 0 \
--lr_warmup_time 0 --no_lr_schedule
```

Like HScaffold, FedSUM-CR also accepts `--momentum 0.9` with SGD (the repository's
Nesterov implementation), or `--base_optimizer Adam`. Optimizer state is initialized
once per client and retained across rounds, including inactive rounds. The control
correction is applied after the optimizer step, so it does not enter momentum or
Adam's moment buffers. These are experimental extensions of Algorithm 3; with
either extension, `h` represents the mean optimizer direction rather than the
mean raw gradient.

Warmup and learning-rate schedules remain unsupported because the correction
reconstructs server history using constant rates. Weight decay is provided by the
existing task gradient, as for the other algorithms. The C10, C100, SVHN and Dolly
YAMLs use momentum 0.9 for FedSUM-CR, matching HScaffold. Set momentum to zero
when running the paper's plain-SGD version.

## Verification

`tests/test_fedsum_cr.py` checks the updates against an independent NumPy
reference that reconstructs corrections from server history and directly computes
SGD, Nesterov and Adam directions. It covers multiple local steps, stochastic minibatch gradients,
1/3/10 clients, delayed first participation, skipped and empty rounds, cached-state
isolation, task-state updates, persistent optimizer state, the plain-SGD
full-participation gradient-descent identity, and invalid settings. Four-rank
CPU/Gloo tests with momentum zero and 0.9 exercise the real server, clients,
existing evaluator, shared arrays, termination, and communication accounting.

`tests/test_task_partitioning.py` also verifies that the server is excluded from
FedSUM-CR training partitions for all existing tasks.
