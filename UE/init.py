"""
File for network loading.
"""


import re
import time
from UEClass import NODE, LINK, ODPair


def read_network(path):
    with open(path, 'r', encoding='UTF-8') as f1:
        # Process the text file
        lines = f1.readlines()
        pattern = re.compile(r'[\w.~]+')
        data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
        number_of_nodes = int(data[1][-1])
        data = data[6:]
        # Create NODE and LINK object
        nodes = [NODE(i) for i in range(number_of_nodes+1)]  # Be CAREFUL that position 0 represents nothing
        links = [LINK(0)]
        for index, line in enumerate(data):
            temp = LINK(index+1, nodes[int(line[0])], nodes[int(line[1])], float(line[2]),
                        float(line[3]), float(line[4]), float(line[5]), float(line[6]))
            links.append(temp)
            nodes[int(line[0])].downstream_link.append(temp)
            nodes[int(line[1])].upstream_link.append(temp)
        return nodes, links


def read_OD(path, nodes):
    with open(path, 'r', encoding='UTF-8') as f1:
        # Process the text file
        lines = f1.readlines()
        pattern = re.compile(r'[0-9.]+|Origin')
        data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
        total_flow = float(data[1][0])
        for i in range(len(data)):
            if 'Origin' in data[i]:
                data = data[i:]
                break
        # Create NODE and LINK object
        od_pairs = list()
        for line in data:
            if "Origin" in line:
                origin = nodes[int(line[-1])]
            else:
                for i in range(len(line)//2):
                    destination = nodes[int(line[2*i])]
                    demand = float(line[2*i+1])
                    if demand != 0:
                        od_pairs.append(ODPair(origin, destination, demand))
        # Check the correctness of OD flows
        temp = total_flow
        for od in od_pairs:
            total_flow -= od.demand
        if abs(total_flow)/temp > 0.01:
            raise ValueError("Data in the file does not match with the total OD flow.")
        return od_pairs


def main(network_name):
    nodes, links = read_network(f'network\\{network_name}\\{network_name}_net.txt')
    od_pairs = read_OD(f'network\\{network_name}\\{network_name}_trp.txt', nodes)
    return nodes, links, od_pairs


if __name__ == "__main__":
    start = time.perf_counter()
    NODES, LINKS, OD_PAIRS = main('cr')
    end = time.perf_counter()
    print(len(OD_PAIRS))
    print(f'running time: {end-start}s')
