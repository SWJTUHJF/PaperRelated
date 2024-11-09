import os
import time
import matplotlib.pyplot as plt
import numpy as np
from Frank_Wolfe import FW
from network_SO import Network as SO
from network_UE import Network as UE
from collections import defaultdict
from datetime import datetime


class OutputResult:
    def __init__(self, name, gap, model):
        self.name = name
        self.gap = gap
        self.model = model
        if self.model == "SO":
            self.net = SO(self.name)
        elif self.model == "UE":
            self.net = UE(self.name)
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
    info = OutputResult(name="SiouxFalls", gap=0.0001, model="UE")
    info.log_out_all_info()
