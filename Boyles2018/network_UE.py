import re
import numpy as np
from collections import defaultdict


class _Node:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.message_id: list[int] = list()
        self.__upstream_link: list[_Link] = list()
        self.__downstream_link: list[_Link] = list()
        self.__downstream_node: list[_Node] = list()
        self.__upstream_node: list[_Node] = list()
        self.__message_vectors: list[list[_State]] = list()
        self.__message_prob: list[float] = list()
        self.__ETT: float = 0  # expected travel time

    def __repr__(self):
        return f'Node {self.node_id}'

    def generate_up_downstream_node(self):
        self.__downstream_node = [link.head for link in self.__downstream_link]
        self.__upstream_node = [link.tail for link in self.__upstream_link]

    def create_message_set(self):
        def backtrack(fs, ps, index):
            if len(fs) == len(self.__downstream_link):
                self.__message_vectors.append(fs[:])
                self.__message_prob.append(round(ps, 6))
                return
            for state in self.__downstream_link[index].link_state:
                fs.append(state)
                ps *= state.get_prob()
                backtrack(fs, ps, index+1)
                fs.pop()
                ps /= state.get_prob()
        backtrack(fs=[], ps=1, index=0)
        self.message_id = [i for i in range(len(self.__message_vectors))]

    def get_specific_message(self, mv_id):
        return self.__message_vectors[mv_id], self.__message_prob[mv_id]

    def get_ETT(self):
        return self.__ETT

    def get_downstream_link(self):
        return self.__downstream_link

    def get_downstream_node(self):
        return self.__downstream_node

    def get_upstream_node(self):
        return self.__upstream_node

    def get_message_vector(self):
        return self.__message_vectors

    def get_message_prob(self):
        return self.__message_prob

    def get_upstream_link(self):
        return self.__upstream_link

    def set_ETT(self, val):
        self.__ETT = val

    def set_upstream_link(self, val):
        self.__upstream_link.append(val)

    def set_downstream_link(self, val):
        self.__downstream_link.append(val)


class _Link:
    def __init__(self, link_id=None, tail=None, head=None,
                 capacity=None, length=None, fft=None,
                 b=None, power=None):
        self.link_id: int = link_id
        self.tail: _Node = tail
        self.head: _Node = head
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = fft
        self.b: float = b
        self.power: float = power
        self.number_of_states: int = 2
        self.link_state: list[_State] = list()

    def __repr__(self):
        return f'Link {self.link_id}, from node {self.tail.node_id} to node {self.head.node_id}'


class _State:
    def __init__(self, state_id, mother_link, prob, capacity, power, b, fft, tail, head):
        self.state_id = state_id
        self.__mother_link: _Link = mother_link
        self.__prob: float = prob
        self.__capacity: float = capacity
        self.__b: float = b
        self.__power: float = power
        self.__fft: float = fft
        self.__flow: float = 0
        self.__cost: float = 0
        self.__tail: _Node = tail
        self.__head: _Node = head
        self.__rho: float = 0
        self.__aux_flow: float = 0

    def __repr__(self):
        return f'link-state: {self.__mother_link.link_id}-{self.state_id}'

    def get_fft(self):
        return self.__fft

    def get_b(self):
        return self.__b

    def get_power(self):
        return self.__power

    def get_capacity(self):
        return self.__capacity

    def get_prob(self):
        return self.__prob

    def get_mother_link(self):
        return self.__mother_link

    def get_flow(self):
        return self.__flow

    def get_cost(self):
        return self.__cost

    def get_time(self):
        return self.__cost

    def get_tail(self):
        return self.__tail

    def get_head(self):
        return self.__head

    def get_rho(self):
        return self.__rho

    def get_aux_flow(self):
        return self.__aux_flow

    def set_aux_flow(self, val):
        self.__aux_flow = val

    def set_rho(self, val):
        self.__rho = val

    def set_flow(self, val):
        self.__flow = val

    def add_aux_flow(self, val):
        self.__aux_flow += val

    def add_rho(self, val):
        self.__rho += val

    def update_cost(self):
        self.__cost = self.__fft * (1 + self.__b * (self.__flow / self.__capacity) ** self.__power)

    def get_specific_cost(self, val):
        return self.__fft * (1 + self.__b * (val / self.__capacity) ** self.__power)


class _ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: _Node = origin
        self.destination: _Node = destination
        self.demand: float = demand


class _Policy:
    def __init__(self, destination):
        self.destination: _Node = destination
        self.__mapping: defaultdict[_Node: dict[int: _Node]] = defaultdict(dict)

    def __repr__(self):
        line1 = f'\n***Policy information:***\n'
        line2 = f'This policy correspond to destination {self.destination.node_id}\n'
        line3 = f'{self.__mapping}\n'
        return line1+line2+line3

    def set_mapping(self, cur_node, message, next_node):
        self.__mapping[cur_node][message] = next_node

    def get_mapping(self, cur_node, message):
        return self.__mapping[cur_node][message]

    def print_whole_mapping(self):
        for key, val in self.__mapping.items():
            print(f'{key}: {val}')


class Network:
    def __init__(self, network_name):
        self.network_name: str = network_name
        self.link_states: list[_State] = list()  # all the link states in the network.
        self.number_of_link_states: int = 0
        self.number_of_nodes: int = 0
        self.number_of_links: int = 0
        self.LINK: list[_Link] = [_Link(0)]
        self.NODE: list[_Node] = list()
        self.ODPAIR: list[_ODPair] = list()
        self.DEST: list[_Node] = list()
        self.ORIGIN: list[_Node] = list()
        self.POLICY: dict[_Node: _Policy] = dict()
        self.TM: dict[_Node: np.array] = dict()
        self.vec_y: defaultdict[_Node: float] = defaultdict(float)
        self.gap: list[float] = list()
        self.gap_time: list[float] = list()
        self.__main()

    def __main(self):
        self.__read_network()
        self.__read_trip()
        self.__initialize_node()
        self.__generate_policy()
        self.__generate_TM()
        self.update_state_cost()

    def __read_network(self):
        with open(f'{self.network_name}/{self.network_name}_net.tntp') as f1:
            # read the network data
            lines = list()
            pattern = re.compile(r'[~\w.]+')
            for line in f1.readlines():
                match = pattern.findall(line)
                if len(match) > 0:
                    lines.append(match)
            self.number_of_nodes = int(lines[1][-1])
            self.number_of_links = int(lines[3][-1])
            for i in range(len(lines)):
                if '~' in lines[i]:
                    lines = lines[i+1:]
                    break
            # create Link and Node instance
            self.NODE = [_Node(i) for i in range(self.number_of_nodes+1)]
            for index, line in enumerate(lines):
                tail, head = self.NODE[int(line[0])], self.NODE[int(line[1])]
                capacity, length, fft = float(line[2]), float(line[3]), float(line[4])
                b, power = float(line[5]), float(line[6])
                temp = _Link(index+1, tail, head, capacity, length, fft, b, power)
                self.LINK.append(temp)
                tail.set_downstream_link(temp)
                head.set_upstream_link(temp)
            # create link state
            self.number_of_link_states = self.number_of_links * 2
            for link in self.LINK[1:]:
                temp = _State(1, link, 0.9, link.capacity*0.9,
                              link.power, link.b, link.fft, link.tail, link.head)
                temp1 = _State(2, link, 0.1, link.capacity*0.1*0.5,
                               link.power, link.b, link.fft, link.tail, link.head)
                self.link_states.extend((temp, temp1))
                link.link_state.extend((temp, temp1))

    def __read_trip(self):
        with open(f'{self.network_name}/{self.network_name}_trips.tntp') as f1:
            lines = list()
            pattern = re.compile(r'[\w.]+')
            for line in f1.readlines():
                match = pattern.findall(line)
                if len(match) > 0:
                    lines.append(match)
            lines = lines[3:]
            for line in lines:
                if 'Origin' in line:
                    ori = self.NODE[int(line[-1])]
                    if ori not in self.ORIGIN:
                        self.ORIGIN.append(ori)
                    continue
                for i in range(len(line)//2):
                    dest = self.NODE[int(line[2*i])]
                    dem = float(line[2*i+1])
                    self.ODPAIR.append(_ODPair(ori, dest, dem))
                    if dest not in self.DEST:
                        self.DEST.append(dest)

    def __generate_policy(self):
        for dest in self.DEST:
            temp = _Policy(dest)
            self.POLICY[dest] = temp

    def __initialize_node(self):
        for node in self.NODE[1:]:
            node.generate_up_downstream_node()
            node.create_message_set()

    def update_state_cost(self):
        for state in self.link_states:
            state.update_cost()

    def __generate_TM(self):
        for dest in self.DEST:
            self.TM[dest] = np.zeros(shape=(self.number_of_link_states, self.number_of_link_states))

    def generate_rho(self, dest: _Node):
        for state in self.link_states:
            state.set_rho(0)
        temp_policy = self.POLICY[dest]
        for cur_node in self.NODE[1:]:
            # This is a must. RP indicates the next node to the dest itself.
            # No rho of states starting from the dest can be obtained.
            if cur_node == dest:
                continue
            for mv_id in cur_node.message_id:
                mv, prob = cur_node.get_specific_message(mv_id)
                next_node = temp_policy.get_mapping(cur_node, mv_id)
                for state in mv:
                    if state.get_head() == next_node:
                        state.add_rho(prob)
                        break
                else:
                    raise ValueError("No corresponding state found when generating rho.")

    def update_TM(self, dest: _Node):
        # create transition matrix
        self.TM[dest] = np.zeros(shape=(self.number_of_link_states, self.number_of_link_states))
        for cur_node in self.NODE[1:]:
            dl = cur_node.get_downstream_link()
            # find the corresponding rows
            for i, row in enumerate(self.link_states):
                if row.get_head() == cur_node:
                    # find the corresponding columns
                    for link in dl:
                        for state in link.link_state:
                            rho = state.get_rho()
                            j = self.link_states.index(state)
                            self.TM[dest][i][j] = rho

    def generate_y(self, dest):
        self.vec_y: defaultdict[_Node: float] = defaultdict(float)
        for od in self.ODPAIR:
            if od.destination == dest:
                self.vec_y[od.origin] += od.demand

    def generate_b(self):
        vec_b = np.zeros(self.number_of_link_states)
        for i, state in enumerate(self.link_states):
            tail = state.get_tail()
            vec_b[i] = state.get_rho() * self.vec_y[tail]
        return vec_b

    def current_TETT(self):
        return sum([state.get_flow() * state.get_cost() for state in self.link_states])


if __name__ == "__main__":
    sf = Network("SiouxFalls")
