from mpi4py import MPI
from logging import Logger
import mpi4py

class Backend():
    def __init__(self, logger:Logger):
        mpi4py.rc.recv_mprobe = False
        self.comm = MPI.COMM_WORLD
        self.rank = self.Get_rank()
        self.world_size = self.Get_size()
        self.logger = logger

    def Get_rank(self):
        return self.comm.Get_rank()

    def Get_name(self):
        return self.comm.Get_name()    

    def Get_size(self):
        return self.comm.Get_size()

    def send(self, value, dest:int, tag:int):
        raise NotImplementedError
    
    def Isend(self, value, dest:int, tag:int):
        raise NotImplementedError

    def isend(self, value, dest:int, tag:int):
        raise NotImplementedError

    def recv(self, value, source:int, tag:int):
        raise NotImplementedError

    def Send(self, obj, dest:int, tag:int):
        raise NotImplementedError

    def Recv(self, obj_buff, source:int, tag:int):
        raise NotImplementedError

    def WaitAll(self, wait_reqs:list):
        raise NotImplementedError

    def Barrier(self):
        raise NotImplementedError
