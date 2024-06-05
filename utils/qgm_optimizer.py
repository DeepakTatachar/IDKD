import collections

import torch
from torch.optim.optimizer import Optimizer, required
from utils.str2bool import str2bool
import numpy as np
from utils.graph_manager import GraphManager
from utils.timer import Timer

def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    current_device = tensors[0].device
    target_device = tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu"
    vec = torch.empty(pointers[-1], device=target_device)

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = (
            tensor.data.view(-1).to(device=target_device)
            if current_device != target_device
            else tensor.data.view(-1)
        )
    return vec

class TensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors=None, use_cuda=True):
        if tensors is None:
            return

        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def clone(self):
        clone_buff = TensorBuffer()
        clone_buff._start_idx = self._start_idx
        clone_buff._end_idx = self._end_idx
        clone_buff._tensors_len = self._tensors_len
        clone_buff._tensors_sizes = self._tensors_sizes
        clone_buff.buffer = self.buffer.clone().detach()
        return clone_buff

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors, param_names=None, param_to_be_ignored=None):
        param_names = (
            [(None, None)] * len(tensors) if param_names is None else param_names
        )
        for (_, param_name), tensor, entry in zip(param_names, tensors, self):
            if (
                param_name is None
                or param_to_be_ignored is None
                or (
                    param_to_be_ignored is not None
                    and param_to_be_ignored not in param_name
                )
            ):
                tensor.data[:] = entry

    def add(self, val):
        clone = self.clone()
        if isinstance(val, torch.Tensor):
            val = val.to(clone.buffer.device)

        clone.buffer += val
        return clone

def _get_data(param_groups, idx, is_get_grad):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    else:
        if param_groups[idx]["params"][0].requires_grad:
            return param_groups[idx]["params"][0]
        else:
            return None

def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True):
    data, shapes = [], []

    for idx, param_name in param_names:
        _data = _get_data(param_groups, idx, is_get_grad)

        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes

def dict_parser(values):
    local_dict = {}
    if values is None:
        return local_dict
    for kv in values.split(","):
        k, v = kv.split("=")
        try:
            local_dict[k] = float(v)
        except ValueError:
            try:
                local_dict[k] = str2bool(v)
            except ValueError:
                local_dict[k] = v
    return local_dict

class QGM_SGD(Optimizer):
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
        super(QGM_SGD, self).__init__(params, defaults)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.graph_manager = graph_manager
        self.logger = logger
        self.device = device
        self.timer = timer

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
        self.virtual_seq_buffer = torch.zeros_like(flatten_params.buffer)

        # # init the conf for slow buffer.
        # self.conf.slow_buffer_conf_ = dict_parser(conf.slow_buffer_conf)

        # # init the dictionary.
        # self.conf.tracking_dict = collections.defaultdict(list)

    def __setstate__(self, state):
        super(QGM_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # first get and flatten all params.
        params, _ = get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        grads, _ = get_data(
            self.param_groups, self.param_names, is_get_grad=True
        )

        flatten_params = TensorBuffer(params)
        flatten_grads = TensorBuffer(grads)
        tmp_flatten_params = flatten_params.buffer.clone()

        # add weight decay.
        flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

        if self.momentum != 0:
            # apply momentum via the slow momentum buffer.
            momentum_buffer = self.momentum_buffer.clone()
            momentum_buffer.mul_(self.momentum).add_(
                flatten_grads.buffer, alpha=1 - self.dampening
            )
            if self.nesterov:
                to_be_applied = flatten_grads.buffer.add(
                    momentum_buffer, alpha=self.momentum
                )
            else:
                to_be_applied = momentum_buffer

        # apply on the model params (and we may clip the update).
        flatten_params.buffer.add_(-to_be_applied * self.param_groups[0]["lr"])

        with self.timer("Comm"):
            np_param = flatten_params.buffer.cpu().numpy()
            neighbors_weights = self.graph_manager.w[self.graph_manager.rank]
            neighbors_rank = np.where(neighbors_weights > 0)[0]
            recv_params = self.graph_manager.comm_neigh(np_param)

        weighted_avg = np.zeros_like(np_param)
        for neigh_idx, neigh_param in enumerate(recv_params):
            n_rank = neighbors_rank[neigh_idx]
            weighted_avg += neighbors_weights[n_rank] * neigh_param

        flatten_params.buffer = torch.from_numpy(weighted_avg).to(self.device)
        flatten_params.unpack(params)

        # update the progress buffer on the virtual sequence.
        self.virtual_seq_buffer.add_(
            (tmp_flatten_params - flatten_params.buffer)
            / self.param_groups[0]["lr"]
        )

        # update the gossip buffer.
        local_step = get_slow_buffer_step(
            None, -1
        )
        assert local_step > 0
        if local_step > 0:
            opp_mu = (
                0.1
            )
            self.momentum_buffer.mul_(1 - opp_mu).add_(
                self.virtual_seq_buffer / local_step, alpha=opp_mu
            )
            self.virtual_seq_buffer = torch.zeros_like(self.virtual_seq_buffer)

        return

def get_slow_buffer_step(slow_buffer_conf, current_epoch) -> int:
    if slow_buffer_conf is None or "local_step" not in slow_buffer_conf:
        return 1
    else:
        return int(slow_buffer_conf["local_step"])