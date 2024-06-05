# -*- coding: utf-8 -*-
import torch
import mpi4py
from utils.graph_manager import GraphManager

class ChocoComm():
    """some auxiliary functions for communication."""
    def __init__(self, graph_manager) -> None:
        self.graph_manager = graph_manager        

    def global_average(self, sum, count, on_cuda=True):
        def helper(array):
            array = torch.FloatTensor(array)
            array = array.cuda() if on_cuda else array
            self.graph_manager.all_reduce(array, op=mpi4py.MPI.SUM)
            all_sum, all_count = array
            if all_count == 0:
                return 0
            else:
                return all_sum / all_count

        avg = helper([sum, count])
        return avg

    def elementwise_min(self, tensor):
        self.graph_manager.all_reduce(tensor, op=mpi4py.MPI.MIN)
        return tensor

    def broadcast(self, tensor, src):
        return self.graph_manager.broadcast(tensor, src=src)

    """some aggregation functions."""


    def _get_data(self,param_groups, idx, is_get_grad):
        # Define the function to get the data.
        # when we create the param_group, each group only has one param.
        if is_get_grad:
            return param_groups[idx]["params"][0].grad
        else:
            return param_groups[idx]["params"][0]


    def _get_shape(self,param_groups, idx):
        return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


    def get_data(self, param_groups, param_names, is_get_grad=True):
        data, shapes = [], []
        for idx, _ in param_names:
            _data = self._get_data(param_groups, idx, is_get_grad)
            if _data is not None:
                data.append(_data)
                shapes.append(self._get_shape(param_groups, idx))
        return data, shapes


    def flatten(self, tensors, shapes=None, use_cuda=True):
        # init and recover the shapes vec.
        pointers = [0]
        if shapes is not None:
            for shape in shapes:
                pointers.append(pointers[-1] + shape[1])
        else:
            for tensor in tensors:
                pointers.append(pointers[-1] + tensor.nelement())

        # flattening.
        vec = torch.empty(
            pointers[-1],
            device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
        )

        for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
            vec[start_idx:end_idx] = tensor.data.view(-1)
        return vec


    def unflatten(self, tensors, synced_tensors, shapes):
        pointer = 0

        for tensor, shape in zip(tensors, shapes):
            param_size, nelement = shape
            tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
            pointer += nelement


    """auxiliary."""


    def recover_device(self, data, device=None):
        if device is not None:
            return data.to(device)
        else:
            return data


"""main aggregators."""


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def _agg(self, graph_manager:GraphManager):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op):
        """Aggregate models by model weight.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            param.data = self._agg(param.data, op=op)

    def agg_grad(self, model, op):
        """Aggregate models gradients.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op)
            param.grad.data = grad

class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, graph_manager:GraphManager):
        # init
        self.graph_manager = graph_manager

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        neighs = self.graph_manager.get_neighbours()
        neighbors_info = self.graph_manager.get_neighbours_rank_weights()
        # Create some tensors to host the values from neighborhood.
        local_data = {i: torch.empty_like(data) for i in neighs}
        local_data[self.graph_manager.rank] = data

        recv_data = self.graph_manager.comm_neigh_diff_size(data.cpu().numpy())
        for neigh_idx, neigh_data in zip(neighs, recv_data):
            local_data[neigh_idx] = torch.Tensor(neigh_data)

        # Aggregate local_data
        if op == "avg":
            output = sum(local_data.values()) / (self.world_size + 1)
        elif op == "weighted":
            output = sum(
                [
                    tensor * neighbors_info[rank]
                    for rank, tensor in local_data.items()
                ]
            )
        elif op == "get_raw_sync_data":
            output = local_data
        else:
            raise NotImplementedError("op {} is not supported yet.".format(op))
        return output



def get_aggregator(graph_manager):
    return DecentralizedAggregation(graph_manager)

