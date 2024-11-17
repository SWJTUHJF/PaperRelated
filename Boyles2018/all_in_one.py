import math
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict
from datetime import datetime


# Part 1: read the network and define the classes
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
        self.__time: float = 0
        self.__toll: float = 0  # toll is the money that is charged, time is the travel time, cost is the summation.
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

    def get_toll(self):
        return self.__toll

    def get_cost(self):
        return self.__cost

    def get_time(self):
        return self.__time

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
        self.__update_time()
        self.__update_toll()
        self.__cost = self.__time + self.__toll

    def get_specific_cost(self, val):
        _time = self.__fft * (1 + self.__b * (val / self.__capacity) ** self.__power)
        _toll = self.__fft * self.__b * self.__power * (val / self.__capacity) ** self.__power
        return _time + _toll

    def __update_time(self):
        self.__time = self.__fft * (1 + self.__b * (self.__flow / self.__capacity) ** self.__power)

    def __update_toll(self):
        self.__toll = self.__fft * self.__b * self.__power * (self.__flow / self.__capacity) ** self.__power


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
        return sum([state.get_flow() * state.get_time() for state in self.link_states])


# Part 2: find the optimal routing policy
def TD_OSP(G: Network, dest):
    # Step 1: Initialize Labels
    for node in G.NODE[1:]:
        if node == dest:
            continue
        node.set_ETT(math.inf)
    SEL = deque()
    SEL.extend(dest.get_upstream_node())
    # Step 2
    while len(SEL):
        node_i = SEL.popleft()
        xi, lbd = [1], [math.inf]
        for link in node_i.get_downstream_link():
            node_j = link.head
            xi_prime, lbd_prime = list(), list()
            for state in link.link_state:
                for k in range(len(xi)):
                    xi_prime.append(xi[k] * state.get_prob())
                    if state.get_cost() + node_j.get_ETT() < lbd[k]:
                        lbd_prime.append(state.get_cost() + node_j.get_ETT())
                    else:
                        lbd_prime.append(lbd[k])
            xi_prime, lbd_prime = reduce(xi_prime, lbd_prime)
            xi, lbd = xi_prime, lbd_prime
        temp = np.dot(xi, lbd)
        if temp < node_i.get_ETT():
            node_i.set_ETT(temp)
            SEL.extend(node_i.get_upstream_node())
    # Step 3: Choose Optimal Policy
    for node in G.NODE[1:]:
        for mv_id in node.message_id:
            mv, mv_prob = node.get_specific_message(mv_id)
            if node == dest:
                G.POLICY[dest].set_mapping(node, mv_id, node)
            else:
                next_node = -1
                min_val = math.inf
                for state in mv:
                    if state.get_cost() + state.get_head().get_ETT() < min_val:
                        min_val = state.get_cost() + state.get_head().get_ETT()
                        next_node = state.get_head()
                G.POLICY[dest].set_mapping(node, mv_id, next_node)


def reduce(xi, lbd):
    dd = defaultdict(float)
    for i in range(len(xi)):
        ele = lbd[i]
        prob = xi[i]
        dd[ele] += prob
    lbd = list(dd.keys())
    xi = list(dd.values())
    return xi, lbd


# Part 3: Frank-Wolfe algorithm
class FW:
    def __init__(self, net, set_gap):
        self.net: Network = net
        self.set_gap: float = set_gap
        self.real_gap: float = math.inf
        self.bisection_gap: float = 0.0001
        self.gap_list: list[float] = list()
        self.main()

    # algorithm implementation
    def main(self):
        iter_times = 0
        self.initialize()
        while abs(self.real_gap) > self.set_gap:
            s = time.perf_counter()
            if iter_times % 5 == 0 and iter_times != 0:
                print(f'{iter_times} iteration: current gap = {self.real_gap:.5f}')
            self.all_or_nothing()
            self.check_convergence()
            if iter_times == 0:
                step = 1
                self.real_gap = math.inf
            else:
                step = self.optimal_step_size()
            for state in self.net.link_states:
                val = (1 - step) * state.get_flow() + step * state.get_aux_flow()
                state.set_flow(val)
                state.update_cost()
            iter_times += 1
            e = time.perf_counter()
            self.net.gap.append(self.real_gap)
            self.net.gap_time.append(e - s)

    def initialize(self):
        for state in self.net.link_states:
            state.set_flow(0)
        self.net.update_state_cost()

    def all_or_nothing(self):
        for state in self.net.link_states:
            state.set_aux_flow(0)
        for dest in self.net.DEST:
            TD_OSP(self.net, dest)
            self.net.generate_rho(dest)
            self.net.update_TM(dest)
            self.net.generate_y(dest)
            mtx_i = np.identity(self.net.number_of_link_states)
            mtx_inv_A = np.linalg.inv(mtx_i - self.net.TM[dest].T)
            temp = mtx_inv_A @ self.net.generate_b()
            for i in range(len(self.net.link_states)):
                self.net.link_states[i].add_aux_flow(temp[i])

    def check_convergence(self):
        numerator = sum([state.get_cost() * state.get_flow() for state in self.net.link_states])
        denominator = sum([state.get_cost() * state.get_aux_flow() for state in self.net.link_states])
        self.real_gap = numerator / denominator - 1
        self.gap_list.append(self.real_gap)

    def optimal_step_size(self):
        def func(step):
            return sum([state.get_specific_cost((1 - step) * state.get_flow() + step * state.get_aux_flow()) *
                        (state.get_aux_flow() - state.get_flow()) for state in self.net.link_states])
        left, right, mid = 0, 1, 0.5
        iter_times, max_iter_times = 1, 500
        while abs(func(mid)) > self.bisection_gap:
            if iter_times == max_iter_times:
                raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
            elif abs(func(mid)) <= self.bisection_gap:
                return mid
            elif func(mid) * func(right) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
            iter_times += 1
        return mid


# Part4: Output results
class OutputResult:
    def __init__(self, name, gap, model):
        self.name = name
        self.gap = gap
        self.model = model
        if self.model == "SO":
            self.net = Network(self.name)
        else:
            raise ValueError("A wrong model is chosen.")
        start = time.perf_counter()
        self.result = FW(net=self.net, set_gap=gap)
        end = time.perf_counter()
        self.run_time = end - start
        print(f"Running time for {model}R is {self.run_time:.2f}s")

    # 输出总系统出行时间
    def total_system_travel_time(self):
        res = 0
        for state in self.net.link_states:
            res += state.get_time() * state.get_flow()
        print(f'Total system travel time: {res}')
        return res

    # 输出各link-state的参数
    def state_info(self):
        for state in self.net.link_states:
            print(f'State {state.get_mother_link().link_id}-{state.state_id}: cost = {state.get_cost():.1f},'
                  f'time = {state.get_time():.1f}, flow = {state.get_flow():.1f},'
                  f'total travel time = {state.get_cost() * state.get_flow():.2f}')

    def plot_gap(self, save=False, path=None):
        x = np.cumsum(self.net.gap_time)
        y = np.array(self.net.gap)
        fig, ax = plt.subplots()
        ax.plot(x, y, linestyle='-', color='g', label=f'0-{self.model}R')
        ax.set_yticks([10**i for i in range(-4, 3)])
        ax.set_yscale('log')
        ax.set_xlabel('Time(in seconds)')
        ax.set_ylabel('Relative gap')
        ax.legend()
        ax.grid(True)
        if save:
            plt.savefig(f'{path}/relative_gap.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 输出各个点的o和d需求量
    def node_origin_demand(self):
        ori_node = defaultdict(float)
        dest_node = defaultdict(float)
        for od in self.net.ODPAIR:
            o, d, dem = od.origin.node_id, od.destination.node_id, od.demand
            ori_node[o] += dem
            dest_node[d] += dem
        print(f'{ori_node}')
        print(f'{dest_node}')

    def log_out_all_info(self):
        dir_name = f"{self.model}R_{datetime.now().strftime('%y%m%d_%H%M%S')}_LogInfo"
        os.mkdir(f'Results/{dir_name}')
        file_path = f'Results/{dir_name}/Data.txt'
        with open(file_path, 'w') as f:
            f.write(f"Data for experiment {self.model}R\n"
                    f"network: {self.name}\n"
                    f"iteration gap: {self.gap}\n"
                    f"Total expected travel time: {self.total_system_travel_time():.2f}\n"
                    f"Total running time: {self.run_time:.2f}s\n")
        self.plot_gap(save=True, path=f'Results/{dir_name}')
        print(f"Results are stored at Results/{dir_name}")


if __name__ == "__main__":
    info = OutputResult(name="SiouxFalls", gap=0.0001, model="SO")
    info.log_out_all_info()
