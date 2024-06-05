from utils.backend import Backend
import numpy as np
import torch
from mpi4py import *

class CPU_MPI(Backend):
    def __init__(self, logger):
        super().__init__(logger)

    def send(self, value, dest:int, tag:int):
        return self.comm.send(value, dest, tag)

    def Isend(self, arr, dest:int, tag:int):
        if not isinstance(arr, np.ndarray):
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()

        return self.comm.Isend(arr, dest, tag)

    def isend(self, value, dest:int, tag:int):
        return self.comm.isend(value, dest, tag)

    def recv(self, source:int, tag:int):
        return self.comm.recv(source=source, tag=tag)

    def Send(self, obj, dest:int, tag:int):
        if not isinstance(obj, np.ndarray):
            if isinstance(obj, torch.Tensor):
                obj = obj.cpu().numpy()
        
        return self.comm.Send(obj, dest=dest, tag=tag)

    def Recv(self, obj_buff, source:int, tag:int):
        if not isinstance(obj_buff, np.ndarray):
            if isinstance(obj_buff, torch.Tensor):
                obj_buff = obj_buff.cpu().numpy()
        
        self.comm.Recv(obj_buff, source=source, tag=tag)
        return obj_buff

    def WaitAll(self, wait_reqs:list):
        MPI.Request.waitall(wait_reqs)
    
    def Barrier(self):
        self.comm.Barrier()