import math
import time
import numpy as np
from network_SO import Network
from TDOSP import TD_OSP as osp


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
            osp(self.net, dest)
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


if __name__ == "__main__":
    sf = Network("SiouxFalls")
    result = FW(net=sf, set_gap=0.01)
