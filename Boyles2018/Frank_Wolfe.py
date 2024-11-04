import math
import numpy as np
from network import Network
from TDOSP import TD_OSP as osp


class FW:
    def __init__(self, net, set_gap):
        self.net: Network = net
        self.set_gap: float = set_gap
        self.real_gap: float = math.inf
        self.bisection_gap: float = 0.001
        self.main()

    # algorithm implementation
    def main(self):
        iter_times = 0
        self.initialize()
        while self.real_gap > self.set_gap:
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
            mtx_i = np.identity(self.net.number_of_link_states)
            mtx_inv_A = np.linalg.inv(mtx_i - self.net.TM[dest].T)
            temp = mtx_inv_A @ self.net.generate_b()
            for i in range(len(self.net.link_states)):
                self.net.link_states[i].add_aux_flow(temp[i])

    def check_convergence(self):
        numerator = 0
        denominator = 0
        for state in self.net.link_states:
            numerator += state.get_cost() * state.get_flow()
        for od in self.net.ODPAIR:
            ori = od.origin
            dem = od.demand
            denominator += dem * ori.get_ETT()
        self.real_gap = numerator/denominator - 1

    def optimal_step_size(self):
        def func(step):
            res = 0
            for state in self.net.link_states:
                val = (1 - step) * state.get_flow() + step * state.get_aux_flow()
                cost = state.get_specific_cost(val) * (state.get_aux_flow() - state.get_flow())
                res += cost
            return res
        left = 0
        right = 1
        mid = (right - left) / 2
        max_iter_times = 500
        iter_times = 1
        while right - left > self.bisection_gap:
            iter_times += 1
            if iter_times == max_iter_times:
                raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
            if abs(func(mid)) <= self.bisection_gap:
                return mid
            elif func(mid) * func(right) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
        return mid


if __name__ == "__main__":
    sf = Network("SiouxFalls")
    result = FW(net=sf, set_gap=0.0001)
