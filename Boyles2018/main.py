from Frank_Wolfe import FW
from network import Network


class OutputResult:
    def __init__(self, net, gap):
        print("Reading Network...")
        self.net = Network(net)
        print("Conducting traffic assignment...")
        self.result = FW(net=self.net, set_gap=gap)

    def total_system_travel_time(self):
        res = 0
        for state in self.net.link_states:
            res += state.get_cost()
        return res

    def print_state_info(self):
        for state in self.net.link_states:
            print(f'State {state.get_mother_link().link_id}-{state.state_id}: cost = {state.get_cost():.1f},'
                  f'toll = {state.get_toll():.1f}, time = {state.get_time():.1f}, flow = {state.get_flow():.1f}')


if __name__ == "__main__":
    info = OutputResult(net="SiouxFalls", gap=0.0001)
    print(info.total_system_travel_time())