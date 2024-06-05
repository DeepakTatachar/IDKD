import math
import torch
from torch.optim.optimizer import Optimizer, required
from utils.qgm_optimizer import TensorBuffer, get_data
from utils.graph_manager import GraphManager
from utils.timer import Timer
import numpy as np


class CentralizedSGD(Optimizer):
    def __init__(
        self,
        params,        
        graph_manager:GraphManager,
        logger,
        timer:Timer=None,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        device='cpu',
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CentralizedSGD, self).__init__(params, defaults)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.graph_manager = graph_manager
        self.logger = logger
        self.timer = timer
        self.device = device

        # store the whole training arguments.
        self.conf = conf

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # init the momentum buffer.
        params, _ = get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        self.momentum_buffer = torch.zeros_like(flatten_params.buffer)

        # init the dictionary.
        # self.conf.tracking_dict = collections.defaultdict(list)

        # init for the evaluator.
        self.cosine_sim_fn = torch.nn.CosineSimilarity()

    def __setstate__(self, state):
        super(CentralizedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # first sync gradients and then apply the aggregated graidents.

        params, _ = get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        grads, _ = get_data(
            self.param_groups, self.param_names, is_get_grad=True
        )
        flatten_params = TensorBuffer(params)
        flatten_grads = TensorBuffer(grads)

        # aggregate the gradients.
        with self.timer("Comm and Agg Grad"):
            np_param = flatten_grads.buffer.cpu().numpy()
            neighbors_weights = self.graph_manager.w[self.graph_manager.rank]
            neighbors_rank = np.where(neighbors_weights > 0)[0]
            recv_params = self.graph_manager.comm_neigh(np_param)

            weighted_avg = np.zeros_like(np_param)
            for neigh_idx, neigh_param in enumerate(recv_params):
                n_rank = neighbors_rank[neigh_idx]
                weighted_avg += neighbors_weights[n_rank] * neigh_param

            flatten_grads.buffer = torch.from_numpy(weighted_avg).to(self.device)

        # self.logger.info(f"Param {flatten_params.buffer[0]}")
        # self.logger.info(f"Grad {flatten_grads.buffer[0]}")

        # add weight decay.
        flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

        if self.momentum != 0:
            # apply momentum via the slow momentum buffer.
            momentum_buffer = self.momentum_buffer
            momentum_buffer.mul_(self.momentum).add_(
                flatten_grads.buffer, alpha=1 - self.dampening
            )
            if self.nesterov:
                to_be_applied = flatten_grads.buffer.add(
                    momentum_buffer, alpha=self.momentum
                )
            else:
                to_be_applied = momentum_buffer
        else:
            to_be_applied = flatten_grads.buffer

        # apply on the model params.
        flatten_params.buffer.add_(to_be_applied, alpha=-self.param_groups[0]["lr"])
        flatten_params.unpack(params)

        return