from Frank_Wolfe import FW
from network import Network
from collections import defaultdict


class OutputResult:
    def __init__(self, net, gap):
        print("Reading Network...")
        self.net = Network(net)
        print("Conducting traffic assignment...")
        self.result = FW(net=self.net, set_gap=gap)

    def total_system_travel_time(self):
        res = 0
        for state in self.net.link_states:
            res += state.get_time() * state.get_flow()
        print(f'Total system travel time: {res}')

    def state_info(self):
        for state in self.net.link_states:
            print(f'State {state.get_mother_link().link_id}-{state.state_id}: cost = {state.get_cost():.1f},'
                  f'toll = {state.get_toll():.1f}, time = {state.get_time():.1f}, flow = {state.get_flow():.1f},'
                  f'total travel time = {state.get_cost() * state.get_flow():.2f}')

    def node_origin_demand(self):
        ori_node = defaultdict(float)
        dest_node = defaultdict(float)
        for od in self.net.ODPAIR:
            o, d, dem = od.origin.node_id, od.destination.node_id, od.demand
            ori_node[o] += dem
            dest_node[d] += dem
        print(f'{ori_node}')
        print(f'{dest_node}')

    def states_params(self):
        for state in self.net.link_states:
            print(f'{state}: fft={state.get_fft()}, b={state.get_b()}, capacity={state.get_capacity():.1f},'
                  f' power={state.get_power()}, flow={state.get_flow():.1f}, time={state.get_time()}')


if __name__ == "__main__":
    info = OutputResult(net="SiouxFalls", gap=0.0001)
    info.total_system_travel_time()
    # info.state_info()
    # info.states_params()
