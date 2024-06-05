import numpy as np
from enum import Enum
from sklearn.preprocessing import normalize
from utils.backend import Backend
from utils.cpu_mpi import CPU_MPI
import torch
import math
import functools
import networkx

class GraphType(Enum):
    FullyConnected = 1
    FullyConnectedSkipSelf = 2
    Ring = 3
    Social_15 = 4
    Social_32 = 5
    Chain = 6

class GraphManager():
    def __init__(
        self, 
        backend:Backend=None, 
        w=None):

        if backend == None:
            backend = CPU_MPI(logger=None)
        self.backend = backend
        self.rank = self.backend.Get_rank()
        self.name = self.backend.Get_name()
        self.num_nodes = self.backend.Get_size()
        if w != None:
            n, m = w.shape
            assert n == self.num_nodes, "Adjacency matrix size different from number of nodes"
            assert m == self.num_nodes, "Adjacency matrix size different from number of nodes"
            self.w = w
        else:
            self.set_graph_type(GraphType.FullyConnected)

        self.comm_bytes = 0
        self.set_inactive_neighbours(None)
        self.device = -1

    def degree(self) -> int:
        return len(self.get_active_neighs())

    @property
    def max_degree(self) -> int:
        return max([self.degree() for _ in range(self.num_nodes)])

    def set_device(self, device):
        self.device = device
        self.backend.device = device

    def set_graph_type(self, graph_type:GraphType):
        if graph_type == GraphType.FullyConnected:
            self.w = self.w = np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes
        
        elif graph_type == GraphType.FullyConnectedSkipSelf:
            self.w = (1 - np.eye(self.num_nodes)) / (self.num_nodes - 1)

        elif graph_type == GraphType.Ring:
            self.w = np.eye(self.num_nodes)

            for curr_node in range(self.num_nodes):
                next_node = (curr_node + 1) % self.num_nodes
                prev_node = (curr_node - 1) % self.num_nodes
                self.w[curr_node, prev_node] = 1
                self.w[curr_node, next_node] = 1

            self.w = normalize(self.w, norm='l1', axis=1)
        elif graph_type == GraphType.Social_15:
            graph = networkx.florentine_families_graph()
            adjacency_matrix = networkx.adjacency_matrix(graph).toarray().astype(
                        np.float
                    ) + np.eye(15)
            sum_row = np.repeat([np.sum(adjacency_matrix, axis=0)], adjacency_matrix.shape[0], axis=0).T
            adjacency_matrix = adjacency_matrix / sum_row
            self.w = adjacency_matrix
        elif graph_type == GraphType.Social_32:
            graph = networkx.davis_southern_women_graph()
            adjacency_matrix = networkx.adjacency_matrix(graph).toarray().astype(
                        np.float
                    ) + np.eye(15)
            sum_row = np.repeat([np.sum(adjacency_matrix, axis=0)], adjacency_matrix.shape[0], axis=0).T
            adjacency_matrix = adjacency_matrix / sum_row
        elif graph_type == GraphType.Chain:
            self.w = np.eye(self.num_nodes)
            for curr_node in range(1, self.num_nodes - 1):
                next_node = (curr_node + 1) % self.num_nodes
                prev_node = (curr_node - 1) % self.num_nodes
                self.w[curr_node, prev_node] = 1
                self.w[curr_node, next_node] = 1

            self.w[0, 1] = 1
            if self.num_nodes >= 2:
                self.w[self.num_nodes - 1, self.num_nodes - 2] = 1

        else:
            raise ValueError("Set w directly")
        
    def set_logger(self, logger):
        self.logger = logger
        self.backend.logger = logger
    
    def partition_get_subset_idx(self, dataset, alpha):
        if self.rank == 0:
            # Partition the dataset
            indxs, label_distribution = self._create_indices(dataset, alpha)
            for node in range(1, self.num_nodes):
                self.backend.send(len(indxs[node]), dest=node, tag=node + 4)
                self.backend.Send(indxs[node], dest=node, tag=node + 5)
                self.backend.Send(label_distribution, dest=node, tag=node + 6)
            
            index = indxs[0]
        else:
            index_len = self.backend.recv(source=0, tag=self.rank + 4)
            index = np.empty(int(index_len), dtype=np.int64)
            index = self.backend.Recv(index, source=0, tag=self.rank + 5)
            label_distribution = np.empty((self.num_nodes, dataset.num_classes), dtype=np.int64)
            label_distribution = self.backend.Recv(label_distribution, source=0, tag=self.rank + 6)

        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        self.logger.info(f"Label Distribution\n{label_distribution}")
        return index, label_distribution

    def _get_bytes(self, array):
        if isinstance(array, np.ndarray):
            return np.prod(array.shape) * array.itemsize
        
        if isinstance(array, torch.Tensor):
            return torch.prod(torch.Tensor(list(array.shape))) * array.element_size()

        raise NotImplementedError

    def _create_indices(self, dataset, alpha):
        targets = []
        for _, (_, labels) in enumerate(dataset.train_loader):
            labels = labels.numpy().tolist()
            targets += labels

        list_of_indices = self.build_non_iid_by_dirichlet(
            random_state=np.random,
            indices2targets=np.array(list(enumerate(targets))),
            non_iid_alpha=alpha,
            num_classes=dataset.num_classes,
            num_indices=dataset.train_length,
            n_workers=self.num_nodes,
        )

        indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        partitions = []
        from_index = 0
        partition_sizes = [1 / self.num_nodes] * self.num_nodes
        for partition_size in partition_sizes:
            to_index = from_index + int(partition_size * dataset.train_length)
            partitions.append(np.array(indices[from_index:to_index]))
            from_index = to_index

        # indices2targets shuffled in place so do not use 
        label_distribution = []
        for node_idx in partitions:
            node_dist = [0] * dataset.num_classes
            for idx in node_idx:
                target = targets[idx]
                node_dist[target] += 1

            label_distribution.append(node_dist)

        label_distribution = np.array(label_distribution)
        return partitions, label_distribution

    def build_non_iid_by_dirichlet(
        self,
        random_state:np.random, 
        indices2targets, 
        non_iid_alpha, 
        num_classes, 
        num_indices, 
        n_workers
    ):
        n_auxi_workers = 10

        # random shuffle targets indices.
        random_state.shuffle(indices2targets)

        # partition indices.
        from_index = 0
        splitted_targets = []
        num_splits = math.ceil(n_workers / n_auxi_workers)
        split_n_workers = [
            n_auxi_workers
            if idx < num_splits - 1
            else n_workers - n_auxi_workers * (num_splits - 1)
            for idx in range(num_splits)
        ]

        split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
        for idx, ratio in enumerate(split_ratios):
            to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
            splitted_targets.append(
                indices2targets[
                    from_index : (num_indices if idx == num_splits - 1 else to_index)
                ]
            )
            from_index = to_index

        idx_batch = []
        for _targets in splitted_targets:
            # rebuild _targets.
            _targets = np.array(_targets)
            _targets_size = len(_targets)

            # use auxi_workers for this subset targets.
            _n_workers = min(n_auxi_workers, n_workers)
            n_workers = n_workers - n_auxi_workers

            # get the corresponding idx_batch.
            min_size = 0
            while min_size < int(0.50 * _targets_size / _n_workers):
                _idx_batch = [[] for _ in range(_n_workers)]
                for _class in range(num_classes):
                    # get the corresponding indices in the original 'targets' list.
                    idx_class = np.where(_targets[:, 1] == _class)[0]
                    idx_class = _targets[idx_class, 0]

                    # sampling.
                    try:
                        proportions = random_state.dirichlet(
                            np.repeat(non_iid_alpha, _n_workers)
                        )
                        # balance
                        proportions = np.array(
                            [
                                p * (len(idx_j) < _targets_size / _n_workers)
                                for p, idx_j in zip(proportions, _idx_batch)
                            ]
                        )
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                            :-1
                        ]
                        _idx_batch = [
                            idx_j + idx.tolist()
                            for idx_j, idx in zip(
                                _idx_batch, np.split(idx_class, proportions)
                            )
                        ]
                        sizes = [len(idx_j) for idx_j in _idx_batch]
                        min_size = min([_size for _size in sizes])
                    except ZeroDivisionError:
                        pass
            idx_batch += _idx_batch

        return idx_batch

    def set_inactive_neighbours(self, inactive_neighs):
        self.inactive_neighs = inactive_neighs
    
    def get_active_neighs(self):
        neighs = np.where(self.w[self.rank] > 0)[0]

        if self.inactive_neighs is not None and len(self.inactive_neighs) > 0:
            neighs = np.delete(neighs, self.inactive_neighs)

        return neighs
        
    def comm_neigh(self, data, inactive_neighs=None):
        assert isinstance(data, np.ndarray) or isinstance(data, torch.Tensor), "Data to send must be a numpy array"

        self.set_inactive_neighbours(inactive_neighs)
        neighs = self.get_active_neighs()

        send_reqs = []
        recv_data_list = []
        for neigh in neighs:
            # Non blocking send data to neighbors
            self.comm_bytes += self._get_bytes(data)
            send_req = self.isend_array(data, dest=neigh)
            send_reqs.append(send_req)

        for neigh in neighs:
            # Blocking receive from neighbors
            recv_data = self.recv_array(src=neigh, shape=data.shape, dtype=data.dtype)
            recv_data_list.append(recv_data)

        # Make sure sends succeeded before return
        self.backend.WaitAll(send_reqs)
        return recv_data_list

    def comm_neigh_diff_size(self, data):
        assert isinstance(data, np.ndarray), "Data to send must be a numpy array"

        neighs = self.get_active_neighs()

        send_reqs = []
        recv_data_list = []
        for neigh in neighs:
            # Non blocking send data to neighbors
            send_req = self.backend.isend(len(data.shape), dest=neigh, tag=neigh + 25)
            send_reqs.append(send_req)

            shape_arr = np.array(data.shape).astype(np.int64)
            send_req = self.isend_array(shape_arr, dest=neigh, tag=neigh + 26)
            send_reqs.append(send_req)
            
            send_req = self.isend_array(data, dest=neigh)
            send_reqs.append(send_req)

            self.comm_bytes += self._get_bytes(data) + self._get_bytes(shape_arr) + 4 # 4 bytes for isend(len)

        for neigh in neighs:
            # Blocking receive from neighbors
            shape_len = self.backend.recv(source=neigh, tag=self.rank + 25)
            data_shape = self.recv_array(src=neigh, shape=(shape_len), dtype=np.int64, tag=self.rank + 26)
            recv_data = self.recv_array(src=neigh, shape=data_shape, dtype=data.dtype)
            recv_data_list.append(recv_data)

        # Make sure sends succeeded before return
        self.backend.WaitAll(send_reqs)
        return recv_data_list

    def get_neighbours(self):
        neighs = np.where(self.w[self.rank] > 0)[0]
        return neighs

    def get_neighbours_weights(self):
        return self.w[self.rank]

    def get_neighbours_rank_weights(self):
        neigh_ranks = self.get_neighbours()
        res = {}
        for neigh_rank in neigh_ranks:
            res[neigh_rank] = self.w[self.rank][neigh_rank]

        return res

    def isend_array(self, array, dest, tag=None):
        if tag is None:
            tag = self.rank + 3
        return self.backend.Isend(array, dest=dest, tag=tag)

    def barrier(self):
        self.backend.Barrier()

    '''
    Blocking receive from source
    '''
    def recv_array(self, src, shape, dtype, tag=None):
        if tag is None:
            tag = src + 3

        if dtype == np.int64 or dtype == np.float32:
            recv_buf = np.empty(shape=shape, dtype=dtype)
            recv_buf = torch.from_numpy(recv_buf).to(device=self.device)
        elif dtype == torch.int64 or dtype == torch.float32:
            recv_buf = np.empty(shape=shape)
            recv_buf = torch.from_numpy(recv_buf).to(dtype).to(self.device)
        else:
            raise NotImplementedError

        if dtype == np.int64:
            recv_buf = recv_buf.to(torch.int64)

        recv_buf = self.backend.Recv(recv_buf, src, tag=tag)
        return recv_buf
    
    '''
    External comm bytes updates
    '''
    def update_comm_bytes(self, external_comm_bytes):
        self.comm_bytes += external_comm_bytes
