'''
Comments: Adopted from official implementation https://github.com/epfml/relaysgd/blob/main/utils/communication.py and integrated into the current code
'''

import torch
import random
from utils.graph_manager import GraphManager
from typing import Dict, Tuple
from utils.qgm_optimizer import TensorBuffer
from logging import Logger

class RelayMechanism:
    def __init__(
        self,
        logger: Logger,
        graph_manager: GraphManager,
        overlap: bool = False,
        normalization_mode: str = "world_size",
        initial_state=None,
        message_drop_prob=0,
        tag: int = 2,
    ):
        self._graph_manager = graph_manager
        self._initial_state = initial_state
        self._received_messages = {}
        self._received_counts = {}
        self._tag = tag
        self._overlap = overlap
        self._normalization_mode = normalization_mode
        self._messages_per_worker = graph_manager.max_degree
        self._message_drop_prob = message_drop_prob

        self._send_handle = None
        self._last_sent_tensor = None
        self.logger = logger

    def send(self, tensor: torch.Tensor):
        """Send `tensor` to neighbors, relaying previously received messages"""
        if self._overlap and self._last_sent_tensor is not None:
            self._received_counts = self.recv_messages(
                example_message=torch.tensor(1), topology=self._graph_manager, tag=self._tag
            )

            self._received_messages = self.recv_messages(
                example_message=self._last_sent_tensor,
                graph_manager=self._graph_manager,
                tag=self._tag,
            )

            # Randomly delete some messages for simulations of robustness
            for key in list(self._received_counts.keys()):
                if random.uniform(0, 1) < self._message_drop_prob:
                    del self._received_counts[key]
                    del self._received_messages[key]

        count_handle, delta_bytes_1 = self.isend_to_neighbors(
            torch.tensor(1), 
            relay=self._received_counts, 
            tag=self._tag
        )

        msg_handle, delta_bytes_2 = self.isend_to_neighbors(
            tensor, 
            relay=self._received_messages, 
            tag=self._tag
        )

        self._send_handle = (count_handle, msg_handle)
        self._last_sent_tensor = tensor

        bytes_sent = (delta_bytes_1 + delta_bytes_2) * self._messages_per_worker / self._graph_manager.max_degree
        self._graph_manager.update_comm_bytes(bytes_sent)

    def set_initial_state(self, tensor: torch.Tensor):
        """Set the fallback for when not enough messages arrive"""
        self._initial_state.data = tensor.clone()

    def receive_average(self):
        """Form an average from received message and the last sent tensor"""
        assert self._last_sent_tensor is not None

        count_tensor = torch.tensor(1)
        avg_tensor = self._last_sent_tensor.clone()
        if isinstance(avg_tensor, TensorBuffer):
            avg_tensor = avg_tensor.buffer

        if not self._overlap:
            self._received_counts = self.recv_sum_into_tensor(
                tensor=count_tensor, 
                tag=self._tag
            )

            self._received_messages = self.recv_sum_into_tensor(
                tensor=avg_tensor,
                tag=self._tag
            )

            # Randomly delete some messages for simulations of robustness
            for key in list(self._received_counts.keys()):
                if random.uniform(0, 1) < self._message_drop_prob:
                    count_tensor.sub_(self._received_counts[key])
                    del self._received_counts[key]
                    avg_tensor.sub_(self._received_messages[key])
                    del self._received_messages[key]

        else:
            for count in self._received_counts.values():
                count_tensor.add_(count)
            for message in self._received_messages.values():
                avg_tensor.add_(message)

        if self._normalization_mode == "world_size":
            num_workers = self._graph_manager.num_nodes

            # Supplement missing values by 'initial state'.
            # This will be 0 for RelaySum/Grad and x0 for RelaySum/Model.
            if count_tensor < num_workers and self._initial_state is not None:
                avg_tensor.add_(self._initial_state, alpha=num_workers - count_tensor)

            avg_tensor.div_(num_workers)

        elif self._normalization_mode == "counts":
            avg_tensor.div_(count_tensor)

        else:
            raise ValueError(f"Unknown normalization mode {self._normalization_mode}")

        return avg_tensor
    
    def isend_to_neighbors(
        self,
        data, 
        relay: Dict[int, torch.Tensor] = {}, 
        tag=2):
    
        handles = []
        neighbors = self._graph_manager.get_active_neighs()
        self_rank = self._graph_manager.rank
        for neighbor in neighbors:
            # Adjacency matrix has self to self connections ignore this
            if neighbor == self_rank:
                continue

            total_relay = sum(
                msg for msg_origin, msg in relay.items() if msg_origin != neighbor
            )

            if isinstance(data, TensorBuffer):
                handle = self._graph_manager.isend_array(data.add(total_relay).buffer, neighbor, tag=tag)
            else:   
                handle = self._graph_manager.isend_array(data + total_relay, neighbor, tag=tag)
            handles.append(handle)

        if isinstance(data, TensorBuffer):
            bytes_sent = self._graph_manager._get_bytes(data.buffer) * self._graph_manager.max_degree  # worst case
        else:
            bytes_sent = self._graph_manager._get_bytes(data) * self._graph_manager.max_degree  # worst case
        self._graph_manager.update_comm_bytes(bytes_sent)

        return handles, bytes_sent

    def recv_messages(
        self,
        example_message: torch.Tensor,
        tag=2
    ) -> Dict[int, torch.Tensor]:
        """Receive one message from each neighbor"""
        received_messages = {}
        neighbors = self._graph_manager.get_active_neighs()
        self_rank = self._graph_manager.rank
        for neighbor in neighbors:
            # Adjacency matrix has self to self connections ignore this
            if neighbor == self_rank:
                continue

            msg = self._graph_manager.recv_array(src=neighbor, shape=example_message.shape, tag=tag, dtype=example_message.dtype)
            received_messages[neighbor] = torch.from_numpy(msg)
        return received_messages

    def recv_sum_into_tensor(
        self,
        tensor: torch.Tensor,
        tag = 2, 
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Add a message from all neighbors to `tensor` (in place)"""
        received_messages = self.recv_messages(tensor, tag=tag)

        for msg in received_messages.values():
            tensor.add_(msg.to(tensor.device))

        return received_messages

class MultiTopologyRelayMechanism:
    """
    Divide communication evenly over multiple topologies
    """
    def __init__(self, graph_managers, logger, **kwargs):
        self.logger = logger
        if not isinstance(graph_managers, list):
            graph_managers = [graph_managers]
        
        initial_state = kwargs.pop("initial_state", None)
        if initial_state is None:
            initial_states = [None for _ in graph_managers]
        else:
            initial_states = [initial_state[i::len(graph_managers)] for i in range(len(graph_managers))]

        self._relays = [RelayMechanism(graph_manager=gm, initial_state=s, logger=logger, **kwargs) for gm, s in zip(graph_managers, initial_states)]

        # Make sure the number of bytes sent is tracked in a representative way.
        messages_per_worker = 0
        
        avg_num_neighbors = graph_managers[0].w.sum() / graph_managers[0].num_nodes
        messages_per_worker = max(avg_num_neighbors, messages_per_worker)
        self.logger.info(f"Messages per worker: {messages_per_worker}")
        for r in self._relays:
            r._messages_per_worker = messages_per_worker

    def send(self, tensor: torch.Tensor):
        if len(self._relays) == 1:
            return self._relays[0].send(tensor)
        else:
            self._last_sent = tensor
            for i, relay in enumerate(self._relays):
                relay.send(tensor[i::len(self._relays)])

    def set_initial_state(self, tensor: torch.Tensor):
        if len(self._relays) == 1:
            return self._relays[0].set_initial_state(tensor)
        else:
            self._last_sent = tensor
            for i, relay in enumerate(self._relays):
                relay.set_initial_state(tensor[i::len(self._relays)])

    def receive_average(self):
        if len(self._relays) == 1:
            return self._relays[0].receive_average()
        else:
            avg_tensor = torch.empty_like(self._last_sent)
            for i, relay in enumerate(self._relays):
                avg_tensor[i::len(self._relays)] = relay.receive_average()
            return avg_tensor
