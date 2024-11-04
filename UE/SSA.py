"""
This file is for the shortest path problem algorithms. All the algorithms below are all-to-one type.
GLC is actually bellman-ford, LC is like SPFA, LS is dijkstra and has the highest efficiency.
In the future considering test Bidirectional Dijkstra.
"""


import time
from collections import deque
from math import inf
from init import main


def initialize(nodes, o_id):
    # Initialize parameters
    for node in nodes:
        node.p = node
        node.u = inf
        node.visited = False
    nodes[o_id].u = 0
    nodes[o_id].p = -1
    nodes[o_id].visited = True


# Based on the current P label, obtain the shortest path.
def obtain_shortest_path(nodes, d_id: int):
    shortest_path_links = list()
    current_node = nodes[d_id]
    while current_node.p != -1:
        for link in current_node.upstream_link:
            if link.tail == current_node.p:
                shortest_path_links.append(link)
                break
        else:
            raise ValueError('Something is wrong with shortest path.')
        current_node = current_node.p
    return shortest_path_links[::-1]


# This is actually Bellman-Ford algorithm
def GLC(nodes, links, o_id: int, d_id: int):
    initialize(nodes, o_id)
    # Main loop
    for _ in nodes[1:]:
        updated = False
        for link in links[1:]:
            tail_node = link.tail
            head_node = link.head
            if tail_node.u != inf and head_node.u > tail_node.u + link.cost:
                head_node.u = tail_node.u + link.cost
                head_node.p = tail_node
                updated = True
        if not updated:
            break
    results = obtain_shortest_path(nodes, d_id)
    return results


# This resembles SPFA but the append criterion is different
def LC(nodes, links, o_id: int, d_id: int):
    initialize(nodes, o_id)
    SEL = deque()
    SEL.append(nodes[o_id])
    while len(SEL) != 0:
        tail_node = SEL.popleft()
        for link in tail_node.downstream_link:
            head_node = link.head
            if head_node.u > tail_node.u + link.cost:
                head_node.u = tail_node.u + link.cost
                head_node.p = tail_node
                if head_node not in SEL:
                    SEL.append(head_node)
    results = obtain_shortest_path(nodes, d_id)
    return results


# This is dijkstra in essence
def LS(nodes, links, o_id: int, d_id: int):
    initialize(nodes, o_id)
    SEL = [nodes[o_id]]
    while len(SEL) != 0:
        SEL.sort(key=lambda node: node.u)
        tail_node = SEL[0]
        if tail_node.node_id == d_id:
            break
        del SEL[0]
        for link in tail_node.downstream_link:
            head_node = link.head
            if head_node.u > tail_node.u + link.cost:
                head_node.u = tail_node.u + link.cost
                head_node.p = tail_node
                if head_node not in SEL:
                    SEL.append(head_node)
    results = obtain_shortest_path(nodes, d_id)
    return results


def SPFA(nodes, links, o_id: int, d_id: int):
    initialize(nodes, o_id)
    que = deque([nodes[o_id]])
    while que:
        tail_node = que.popleft()
        tail_node.visited = False
        for link in tail_node.downstream_link:
            destination = link.head
            cost = link.cost
            if tail_node.u != inf and tail_node.u + cost < destination.u:
                destination.u = tail_node.u + cost
                destination.p = tail_node
                if not destination.visited:
                    que.append(destination)
                    destination.visited = True
    results = obtain_shortest_path(nodes, d_id)
    return results


def algorithm_dict(name):
    types = {"GLC": GLC, "LC": LC, "LS": LS, "SPFA": SPFA}
    try:
        return types[name]
    except IndexError:
        print("A wrong SSP algorithm has been chosen. Automatically change to SPFA.")
        return types["SPFA"]


def running_time_analysis():
    start = time.perf_counter()
    algorithm_dict(an)(NODES, LINKS, o_id, d_id)
    end = time.perf_counter()
    print(f'running time for {an} is {end - start:.7f}')
    res = obtain_shortest_path(NODES, d_id)
    res_nodes = [o_id]
    total_cost = 0
    for link in res:
        total_cost += link.cost
        res_nodes.append(link.head.node_id)
    print(res_nodes)
    print(f'Total cost is {total_cost:.3f}')
    print("_"*20)


if __name__ == "__main__":
    network_name, o_id, d_id = 'cr', 1, 27
    NODES, LINKS, OD_PAIRS = main(network_name)
    for an in ["SPFA", "LC", "LS", "GLC"]:
        running_time_analysis()
