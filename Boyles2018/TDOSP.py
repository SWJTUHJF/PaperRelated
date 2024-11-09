import math
import numpy as np
from network_SO import Network
from collections import deque, defaultdict


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


if __name__ == "__main__":
    sf = Network("SiouxFalls")
    n = sf.NODE[24]
    TD_OSP(sf, n)
    sf.generate_rho(n)
    sf.update_TM(n)
    print(sf.generate_b())
