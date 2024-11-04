"""
File for storing objects of NODE, LINK and OD_Pair.
"""


from math import inf


class NODE:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.upstream_link: list[LINK] = list()
        self.downstream_link: list[LINK] = list()
        self.p: NODE = self
        self.u: float = inf
        self.visited = False  # For SPFA algorithm

    def __repr__(self):
        return f'NODE {self.node_id}, U: {self.u}'


class LINK:
    def __init__(self, link_id, tail=None, head=None, capacity=None,
                 length=None, free_flow_time=None, alpha=None, beta=None):
        self.link_id: int = link_id
        self.tail: NODE = tail
        self.head: NODE = head
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = free_flow_time
        self.alpha: float = alpha
        self.beta: float = beta
        self.flow: float = 0
        self.auxiliary_flow: float = 0
        self.cost: float = 0
        if link_id != 0:
            self.update_cost()

    def __repr__(self):
        return f'LINK {self.link_id} complete, cost = {self.cost}, flow = {self.flow}'

    def update_cost(self):
        self.cost = self.fft * (1 + self.alpha * (self.flow / self.capacity) ** self.beta)

    def cost_under_certain_flow(self, temp):
        return self.fft * (1 + self.alpha * (temp / self.capacity) ** self.beta)


class ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: NODE = origin
        self.destination: NODE = destination
        self.demand: float = demand

    def __repr__(self):
        return f'ODPair {self.origin.node_id}->{self.destination.node_id}={self.demand}'


if __name__ == "__main__":
    pass
