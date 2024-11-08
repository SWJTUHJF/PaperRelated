"""
Basic steps for Frank-Wolfe algorithm for UE.
Step 1: Set all the links' flow to zero and perform the all-or-nothing method, get the feasible start X.
Step 2: Update the cost of each link.
Step 3: Based on the obtained cost of each link, perform the all-or-nothing method once. This is the descend direction.
Step 4: Calculate the optimal move size by using bisection method.
Step 5: Check convergence.
"""


from time import perf_counter
from init import main
from SSA import algorithm_dict


def update_costs(links):
    for link in links[1:]:
        link.update_cost()


def all_or_nothing(nodes, links, od_pairs, ssp_name, initialize=False):
    if initialize:
        print('Initializing...')
    # Everytime before performing the all-or-nothing method, the auxiliary of each flow must be reset
    for link in links:
        link.auxiliary_flow = 0
    for od in od_pairs:
        origin = od.origin  # A node object
        destination = od.destination
        demand = od.demand
        shortest_links = algorithm_dict(ssp_name)(nodes, links, origin.node_id, destination.node_id)
        if initialize:
            for link in shortest_links:
                link.flow += demand
        for link in shortest_links:
            link.auxiliary_flow += demand


# Function for step 4
def derivative_f(alpha, links):
    res = 0
    for link in links[1:]:
        first_part = link.auxiliary_flow - link.flow
        second_part = link.cost_under_certain_flow(link.flow + alpha * (link.auxiliary_flow - link.flow))
        res += first_part * second_part
    return res


def bisection(links, accuracy):
    left = 0
    right = 1
    middle = (right + left) / 2
    max_iter_times = 500
    iter_times = 1
    while right - left > accuracy:
        iter_times += 1
        if iter_times == max_iter_times:
            raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
        if abs(derivative_f(middle, links)) <= accuracy:
            return middle
        elif derivative_f(middle, links) * derivative_f(right, links) > 0:
            right = middle
        else:
            left = middle
        middle = (right + left) / 2
    return middle


def update_flow(links, step):
    for link in links[1:]:
        link.flow = link.flow + step * (link.auxiliary_flow - link.flow)


def check_convergence(links, accuracy):
    numerator = 0
    denominator = 0
    for link in links[1:]:
        denominator += link.auxiliary_flow * link.cost
        numerator += link.flow * link.cost
    if abs(numerator / denominator - 1) < accuracy:
        return True
    else:
        return False


def conduct_FW(network_name, SSA, accuracy1, accuracy2):
    # Read the networks
    NODES, LINKS, OD_PAIRS = main(network_name)
    # Initialize
    start = perf_counter()
    all_or_nothing(NODES, LINKS, OD_PAIRS, SSA, initialize=True)
    end = perf_counter()
    print(f'Running time for initialization is {end-start:.5f}s')
    # Main loop
    max_iter_times = 500
    iter_times = 1
    while iter_times <= max_iter_times:
        print("****************************************")
        for link in LINKS:
            print(f"link {link.link_id}: flow = {link.flow}")
        update_costs(LINKS)
        all_or_nothing(NODES, LINKS, OD_PAIRS, SSA)
        step = bisection(LINKS, accuracy1)
        update_flow(LINKS, step)
        if check_convergence(LINKS, accuracy2):
            print('Success!')
            # print(f'Total system travel time is {TSTT(LINKS):.2f}')
            return LINKS
        else:
            if iter_times % 20 == 0:
                print(f'{iter_times} iteration...')
                # print(f'Total system travel time is {TSTT(LINKS):.2f}')
        iter_times += 1

    else:
        print("Reach maximum iteration times in main loop part but still fail to converge.")
        return LINKS


def TSTT(links):
    res = 0
    for link in links[1:]:
        res += link.cost * link.flow
    return res


if __name__ == "__main__":
    name = 'br12'
    algorithm = 'LC'
    bisection_accuracy = 0.001  # accuracy 1
    convergence_accuracy = 0.001  # accuracy 2
    start = perf_counter()
    LINKS = conduct_FW(name, algorithm, bisection_accuracy, convergence_accuracy)
    end = perf_counter()
    print(f'Total running time is {end - start:.5f} seconds')

