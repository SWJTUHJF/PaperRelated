"""
Network(highest scale first): cr, cs, sf, br12, toy.
Shortest Path Algorithm(highest efficiency first): LS, SPFA, LC, GLC
"""


from datetime import datetime
import time
import pandas as pd
from Frank_Wolfe import conduct_FW


name = 'cr'
algorithm = 'LS'
bisection_accuracy = 0.001  # accuracy 1
convergence_accuracy = 0.001 # accuracy 2


def print_info(LINKS):
    TTST = sum(link.cost * link.flow for link in LINKS[1:])
    print(f"Total system travel time is {TTST}.")
    ID = [i.link_id for i in LINKS[1:]]
    cost = [i.cost for i in LINKS[1:]]
    flow = [i.flow for i in LINKS[1:]]
    res = pd.DataFrame({"Link ID": ID, "Link flow": cost, "Link cost": flow})
    path = f'FlowInfo/{name.upper()}_{algorithm}_{datetime.now().strftime("%y%m%d_%H%M%S")}.xlsx'
    res.to_excel(path)
    print(f'Flow information has been recorded at {path}')


if __name__ == "__main__":
    start = time.perf_counter()
    print(f"*****Conducting traffic assignment in network {name.upper()}*****")
    LINKS = conduct_FW(name, algorithm, bisection_accuracy, convergence_accuracy)
    end = time.perf_counter()
    print(f'Total running time: {end-start:.5f} seconds.')
    print(f"Print the flow information?[Y/N]")
    if input() == 'y' or 'Y':
        print_info(LINKS)
