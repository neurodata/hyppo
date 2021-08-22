
from abc import abstractmethod

class IndependentJob(object):
    def __init__(self, aggregator, walltime=24 * 60 * 60, memory=2, nodes=1):
        self.aggregator = aggregator
        self.walltime = walltime
        self.memory = memory
        self.nodes = nodes
    
    @abstractmethod
    def compute(self):
        raise NotImplementedError()
    
    def get_walltime_mem_nodes(self):
        return self.walltime, self.memory, self.nodes
