import networkx as nx

def build_P1_graph():
    DG = nx.DiGraph()
    node_attrs = {
        'start':dict(h=0),
        'A':dict(h=3),
        'B':dict(h=2),
        'C':dict(h=3),
        'D':dict(h=3),
        'E':dict(h=1),
        'F':dict(h=3),
        'G':dict(h=2),
        'H':dict(h=1),
        'I':dict(h=2),
        'J':dict(h=3),
        'K':dict(h=2),
        'L':dict(h=3),
        'goal':dict(h=0)
    }
    edges = [
        ('start','A',1),
        ('A','D',1),
        ('A','E',1),
        ('A','F',3),
        ('E','goal',3),
        ('start','B',1),
        ('B','G',4),
        ('G','goal',3),
        ('B','H',1),
        ('B','I',2),
        ('I','goal',3),
        ('start','C',1),
        ('C','J',1),
        ('C','K',1),
        ('K','goal',2),
        ('C','L',1)
    ]
    DG.add_weighted_edges_from(edges)
    nx.set_node_attributes(DG,node_attrs)
    return DG


def a_star(DG, heuristic=True, start='start', goal='goal'):
    queue = [(0,start,0,None)] # (Priority, Name, Distance from start, predecessor)
    n_iter = 0
    while queue:
        n_iter += 1
        new_queue = []
        p = min(queue, key= lambda x:x[0])
        for child in list(DG[p[1]]):
            weight = DG[p[1]].get(child)['weight']
            
            if heuristic:
                h = DG.nodes[child]['h']
            else:
                h = 0

            priority = p[2] + weight + h
            dist_from_start = p[2] + weight
            new_queue.append((priority,child, dist_from_start, p[1]))

        queue.extend(new_queue)
        queue = sorted(queue[1:], key=lambda x:x[0])

        if queue[0][1] == goal:
            child = queue[0][3]
            node_list = [goal, child]
            path_distance = DG[child][goal]['weight']
            while child != 'start':
                new_child = list(DG.predecessors(child))[0]
                path_distance += DG[new_child][child]['weight']
                child = new_child
                node_list.append(child)

            final_sequence = list(reversed(node_list))
            break

    return final_sequence, n_iter, path_distance
