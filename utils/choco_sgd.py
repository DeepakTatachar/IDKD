from copy import deepcopy
import threading
import torch
from torch.optim.optimizer import Optimizer, required
from utils.qgm_optimizer import TensorBuffer, get_data
from utils.compressors import SignCompressor, SparsificationCompressor, QuantizationCompressor
from utils.choco_comm import ChocoComm, get_aggregator
from utils.graph_manager import GraphManager
import torch.multiprocessing as mp

class ChocoSGD(Optimizer):
    def __init__(
        self,
        params,
        graph_manager:GraphManager,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        logger=None,
        conf=None,
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
        super(ChocoSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.graph_manager = graph_manager
        self.logger = logger

        # define the aggregator.
        self.rank = self.graph_manager.rank
        self.neighbors_info = self.graph_manager.get_neighbours_rank_weights()
        self.world_aggregator = get_aggregator(self.graph_manager)

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        self.neighbor_hat_params, self.params_shapes = self._init_neighbor_hat_params(
            graph_manager, self.param_groups, self.param_names
        )

        self.aggregator = get_aggregator(graph_manager)

        self.compressor = CHOCOCompressor(
            aggregator=self.aggregator,
            comm_op=conf.comm_op,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            choco_comm=ChocoComm(graph_manager)
        )

        self.n_bits = torch.FloatTensor([0])

    def __setstate__(self, state):
        super(ChocoSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def _init_neighbor_hat_params(self, graph_manager, param_groups, param_names):
        params, params_shapes = get_data(
            param_groups, param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        return (
            {
                graph_manager.rank: deepcopy(flatten_params),
                "memory": deepcopy(flatten_params),
            },
            params_shapes,
        )

    def step(self, closure=None, **kargs):
        # Apply the gradients with the weight decay and momentum.
        apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )


        # recover current params and hat_params
        params, flatten_params, flatten_hat_params = recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.graph_manager.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
        )

        # get updated flatten params.
        update_params_from_neighbor(
            neighbor_hat_params=self.neighbor_hat_params,
            flatten_params=flatten_params,
            consensus_stepsize=self.conf.consensus_stepsize,
            self_rank=self.graph_manager.rank,
        )

        # update the local model using neighborhood info.
        flatten_params.unpack(params)

        # start compress/sync.
        sync_buffer = {
            "original_shapes": self.params_shapes,
            "flatten_params": flatten_params,
            "flatten_hat_params": flatten_hat_params,
        }

        self.compressor.pipeline(
            sync_buffer=sync_buffer,
            neighbor_hat_params=self.neighbor_hat_params,
            neighbors_info=self.neighbors_info,
        )

        self.n_bits.data[0] = sync_buffer["n_bits"]

        # Plain DP-SGD comment after update_params_call to this line
        # flatten_params.buffer = self.aggregator._agg(
        #             flatten_params.buffer, op="weighted"
        #         )

        # flatten_params.unpack(params)

        return self.n_bits.item()

    def __del__(self):
        pass


"""the entry for CHOCOCompressor."""

def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()

class CHOCOCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = CHOCOSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = CHOCOQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = CHOCOSignCompressor(**kargs)
        else:
            raise NotImplementedError

    def pipeline(self, *args, **kargs):
        return self.compressor_fn.pipeline(*args, **kargs)

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def sync(self, *args, **kargs):
        return self.compressor_fn.sync(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)


"""Detailed CHOCOCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class CHOCOSparsificationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        choco_comm:ChocoComm,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.comm = choco_comm
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        self.compress(sync_buffer)
        self.sync(sync_buffer)
        self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)


    def compress(self, sync_buffer):
        selected_values, selected_indices = [], []

        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _selected_values, _selected_indices = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.compress_ratio,
                self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        # get selected shapes.
        selected_shapes = [len(_value) for _value in selected_values]

        # flatten selected values/indices.
        flatten_selected_values = TensorBuffer(selected_values)
        flatten_selected_indices = TensorBuffer(selected_indices)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
            flatten_selected_indices.buffer
        )

        # update shared dict.
        sync_buffer["selected_shapes"] = selected_shapes
        sync_buffer["flatten_selected_values"] = flatten_selected_values
        sync_buffer["flatten_selected_indices"] = flatten_selected_indices
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # get the flatten values and prepare the sync.
        message_to_send = torch.cat(
            [
                sync_buffer["flatten_selected_values"].buffer,
                sync_buffer["flatten_selected_indices"].buffer,
            ]
        )

        synced_message = self.aggregator_fn._agg(
            message_to_send, op="get_raw_sync_data"
        )

        # update sync_buffer.

        sync_buffer["synced_message"] = synced_message
        sync_buffer["sycned_message_size"] = len(message_to_send)

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):
        # uncompress and update.
        message_size = int(sync_buffer["sycned_message_size"] / 2)

        for rank, weight in neighbors_info.items():
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]
            hat_params_memory = neighbor_hat_params["memory"]

            # recover values/indices to the correct device.
            q_values, q_indices = self._uncompress_helper(
                hat_params,
                rank,
                sync_buffer["synced_message"],
                message_size,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # update neighbor_hat_params
            if rank in neighbor_hat_params:
                hat_params.buffer[q_indices] += q_values
            hat_params_memory.buffer[q_indices] += weight * q_values

    def _uncompress_helper(
        self,
        _hat_params,
        _rank,
        synced_message,
        sycned_message_size,
        selected_shapes,
        original_shapes,
    ):
        # recover the message and the corresponding device.
        _message = self.comm.recover_device(
            synced_message[_rank], device=_hat_params.buffer.device
        )
        values = _message[:sycned_message_size]
        indices = _message[sycned_message_size:]

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes, original_shapes
        )
        return q_values, q_indices


class CHOCOQuantizationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        choco_comm:ChocoComm,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.comm = choco_comm
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        self.compress(sync_buffer)
        self.sync(sync_buffer)
        self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)


    def compress(self, sync_buffer):
        quantized_values = []

        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _quantized_values = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.quantize_level,
                self.is_biased,
            )
            quantized_values.append(_quantized_values)

        # flatten selected values/indices.
        flatten_updates = TensorBuffer(quantized_values)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 32

        # update shared dict.
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare the sync.
        to_sync_message = sync_buffer["flatten_updates"].buffer

        # sync.
        synced_message = self.aggregator_fn._agg(
            to_sync_message, op="get_raw_sync_data", force_wait=False
        )

        sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):

        for rank, weight in neighbors_info.items():
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]
            hat_params_memory = neighbor_hat_params["memory"]

            # recover correct values/indices.
            q_values = self.comm.recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )

            # update neighbor_hat_params
            if rank in neighbor_hat_params:
                hat_params.buffer += q_values
            hat_params_memory.buffer += weight * q_values


class CHOCOSignCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        choco_comm:ChocoComm,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.comm = choco_comm
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        self.compress(sync_buffer)
        self.sync(sync_buffer)
        self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)


    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        norms, updates = [], []
        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _update = half_param - hat_param
            updates += [_update]
            norms += [_update.norm(p=1)]

        # flatten selected values/indices.
        flatten_norms = TensorBuffer(norms)
        flatten_directions = TensorBuffer(updates)
        signs, sign_size = self.compressor_fn.compress(flatten_directions.buffer)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_norms.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["flatten_norms"] = flatten_norms
        sync_buffer["flatten_directions"] = flatten_directions
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_flatten_norms = sync_buffer["flatten_norms"].buffer
        to_sync_signs = sync_buffer["signs"]

        synced_flatten_norms = self.aggregator_fn._agg(
            to_sync_flatten_norms, op="get_raw_sync_data", force_wait=False
        )
        synced_signs = self.aggregator_fn._agg(
            to_sync_signs, op="get_raw_sync_data", force_wait=False
        )

        sync_buffer["synced_flatten_norms"] = synced_flatten_norms
        sync_buffer["synced_signs"] = synced_signs

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):
        # uncompress and update.
        for rank, weight in neighbors_info.items():
            # get hat_params of the current rank.
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]

            # recover the message and the corresponding device.
            sync_buffer["flatten_norms"].buffer = self.comm.recover_device(
                sync_buffer["synced_flatten_norms"][rank],
                device=hat_params.buffer.device,
            )
            sync_buffer["flatten_directions"].buffer = self.compressor_fn.uncompress(
                self.comm.recover_device(
                    sync_buffer["synced_signs"][rank], device=hat_params.buffer.device
                ),
                sync_buffer["sign_size"],
            )

            # update neighbor_hat_params
            for hat_param, hat_param_memory, norm, sign in zip(
                hat_params,
                neighbor_hat_params["memory"],
                sync_buffer["flatten_norms"],
                sync_buffer["flatten_directions"],
            ):
                _update = norm / sign.nelement() * sign
                if rank in neighbor_hat_params:
                    hat_param.add_(_update)
                hat_param_memory.add_(_update, alpha=weight)


def apply_gradient(param_groups, state, apply_grad_to_model=True):
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad.data

            # get param_state
            param_state = state[p]

            # add weight decay.
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            if apply_grad_to_model:
                p.data.add_(d_p, alpha=-group["lr"])
            else:
                p.grad.data = d_p


def recover_params(
    param_groups, param_names, rank=None, neighbor_hat_params=None, get_hat_params=True
):
    # get flattened params.
    params, _ = get_data(param_groups, param_names, is_get_grad=False)
    flatten_params = TensorBuffer(params)

    if get_hat_params:
        assert neighbor_hat_params is not None and rank is not None
        # recover the hat_params.
        flatten_hat_params = TensorBuffer(params)
        flatten_hat_params.buffer.data[:] = neighbor_hat_params[rank].buffer
        return params, flatten_params, flatten_hat_params
    else:
        return params, flatten_params


def update_params_from_neighbor(
    neighbor_hat_params, flatten_params, consensus_stepsize, self_rank
):
    flatten_params.buffer += consensus_stepsize * (
        neighbor_hat_params["memory"].buffer - neighbor_hat_params[self_rank].buffer
    )


"""utilities for parallel choco."""


class HelperThread(threading.Thread):
    def __init__(self, name, func, *args, **kargs):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func

        # task-related.
        self.args = args
        self.kargs = kargs

    def run(self):
        self.func(**self.kargs)


def join_thread(thread):
    if thread is None:
        return False
    thread.join()
    return True
