import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import networkx as nx
import timeit
from tqdm import tqdm

class Obstacle:

    def __init__(self, xy, sx=1, sy=1, c=True):
        if c: # If centerpoint given, convert to lower-left coordinate position
            self.xy = np.array(xy) - np.array([0.5*sx,0.5*sy])
            self.c = xy
        else:
            self.xy = np.array(xy)
            self.c = self.xy + np.array([0.5*sx,0.5*sy])

        self.sx = sx
        self.sy = sy

        min_x = self.xy[0]
        min_y = self.xy[1]
        max_x = self.xy[0] + self.sx
        max_y = self.xy[1] + self.sy

        self.segments = [
            [(min_x,min_y),(max_x,min_y)],
            [(max_x,min_y),(max_x,max_y)],
            [(max_x,max_y),(min_x,max_y)],
            [(min_x,max_y),(min_x,min_y)]
            ]

    def is_colliding(self, q):
        min_x = self.xy[0]
        min_y = self.xy[1]
        max_x = self.xy[0] + self.sx
        max_y = self.xy[1] + self.sy

        x, y = q

        return (min_x <= x <= max_x) and (min_y <= y <= max_y)
    

    @staticmethod
    def _check_line_intersect(p11,p12,p21,p22):
        """
        Check if line given by points p11,p12 intersects line given by points p21,p22
        """
        def lin_params(x1,y1,x2,y2):
            """
            Given to points, return associated slope and intercept
            """
            m = (y2-y1)/(x2-x1)
            b = -m*x1+y1
            return m,b
        
        x1,y1 = p11
        x2,y2 = p12
        x3,y3 = p21
        x4,y4 = p22

        Segment1 = {(x1, y1), (x2, y2)}
        Segment2 = {(x3, y3), (x4, y4)}
        I1 = [min(x1,x2), max(x1,x2)]
        I2 = [min(x3,x4), max(x3,x4)]
        Ia = [max( min(x1,x2), min(x3,x4) ),min( max(x1,x2), max(x3,x4) )]

        if max(x1,x2) < min(x3,x4):
            return False
        
        m1,b1 = lin_params(x1, y1, x2, y2)
        
        if x3 == x4:
            yk = m1*x3 + b1
            if (min(y3,y4) <= yk <= max(y3,y4)) and (min(y1,y2) <= yk <= max(y1,y2)):
                return True
            else:
                return False

        m2,b2 = lin_params(x3,y3,x4,y4)

        if m1 == m2:
            return False


        Xa = (b2 - b1) / (m1 - m2)

        if ( (Xa < max( min(x1,x2), min(x3,x4) )) or
            (Xa > min( max(x1,x2), max(x3,x4) )) ):
            return False 
        else:
            return True

    def is_segment_intersecting(self,p1,p2):
        for p21,p22 in self.segments:
            if self._check_line_intersect(p1,p2,p21,p22):
                return True
        
        return False

class RRTSolver:
    def __init__(self, qstart, qgoal, obstacles, xlim, ylim, pgoal=0.05, eps=0.25, max_iter=5000, step_size=0.5):
        self.qstart = np.array(qstart)
        self.qgoal = np.array(qgoal)
        self.obstacles = obstacles
        self.pgoal = pgoal
        self.eps = eps
        self.max_iter = max_iter
        self.xlim = xlim
        self.ylim = ylim
        self.step_size = step_size
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(
            [(1,dict(pos=(self.qstart), heuristic=0))]
            )
        self._c = 1
        self.path = None
        self.distance = None

        self.solution_found = False

    def is_segment_intersecting(self, p1, p2):
        intersects = False
        for obstacle in self.obstacles:
            if obstacle.is_segment_intersecting(p1,p2):
                intersects = True
                break
        return intersects

    @staticmethod
    def get_distance(p1,p2):
        return np.linalg.norm(p1-p2)

    def dist_to_goal(self, pt):
        return np.linalg.norm(self.qgoal-pt)

    def is_solution(self,pt):
        return  self.dist_to_goal(pt) <= self.eps

    def gen_samples(self):
        """
        steps is number of subdivisions to give path
        """
        if np.random.rand() <= self.pgoal:
            qrand = self.qgoal
        else:
            x_range = self.xlim[1] - self.xlim[0]
            y_range = self.ylim[1] - self.ylim[0]
            qrand = np.random.rand(2)*np.array([x_range, y_range]) + np.array([self.xlim[0],self.ylim[0]])
        
        # Find closest point on current tree
        min_dist = float('inf')
        min_dist_node = None
        qnear = None

        for node in self.Graph.nodes:
            dist = np.linalg.norm(qrand-self.Graph.nodes[node]['pos'])
            if dist < min_dist:
                min_dist = dist
                qnear = self.Graph.nodes[node]['pos']
                min_dist_node = node

        assert (min_dist < float('inf')) and (qnear is not None)

        # Find angle between points:
        dx,dy = qrand - qnear
        theta = np.arctan2(dy,dx)
        qnew = qnear
        segment_length = self.step_size
        
        start = True
        end = False
        while not end:
            # Accidentally made connectRRT
            # Didn't want things to break so just terminate while loop after 1 iteration
            end = True
            # Gen new point
            if self.get_distance(qnew, qrand) < self.step_size:
                x_new, y_new = qrand[0], qrand[1]
                segment_length = self.get_distance(qnew, qrand)
                end = True
            else:
                x_new = qnew[0] + self.step_size*np.cos(theta)
                y_new = qnew[1] + self.step_size*np.sin(theta)
            
            if start:
                prev_node = min_dist_node
            else:
                prev_node = self._c
            
            start = False

            # If segment not intersecting, add node, edge to graph
            if not self.is_segment_intersecting(qnew,np.array([x_new, y_new])):
                qnew = np.array([x_new, y_new])
                if self.is_solution(qnew):
                    self.solution_found = True
                    self.Graph.add_node(0, pos=qnew, heuristic=0)
                    self.Graph.add_edge(prev_node,0,weight=segment_length)
                    self._c += 1
                    break
                else:
                    self.Graph.add_node(self._c+1, pos=qnew, heuristic=self.dist_to_goal(qnew))
                    self.Graph.add_edge(prev_node,self._c+1,weight=segment_length)
                    self._c += 1
                    
            else: # Segment intersects obstacle
                break

    @staticmethod
    def a_star(G, start=1, goal=0, use_heuristic=True):
        queue = [(0, start, 0, None)] # (priority, node_num, reach_cost, parent)
        
        prev_queued = {} # KEY : node ,       VALUE: (reach_cost, heuristic)
        explored = {} # KEY : explored node, VALUE: parent of explored node
        
        while queue:
            # Retrieve node with lowest priority score (top of queue)
            priority, node, reach_cost, parent = queue.pop(0)
            
            if node == goal:
                path = [node]
                total_dist = 0
                next_node = parent
                # Backtrack through nodes (stored in explored list) to find path to goal
                while next_node != None:
                    path.append(next_node)
                    total_dist += G[path[-1]][path[-2]]['weight']
                    next_node = explored[next_node]
                    
                return list(reversed(path)), total_dist
            
            if node in explored.keys():
                if explored[node] == None:
                    continue
                
                cost, heuristic = prev_queued[node]
                if cost < reach_cost:
                    continue
            
            # Update explored dict to include parent-child relationship of current node
            explored[node] = parent 
            
            # Iterate over children of current node
            for child, w in G[node].items(): 
                new_cost = reach_cost + w['weight'] # Reach cost of child node
                if child in prev_queued.keys():
                    qcost, heuristic = prev_queued[child]
                    if qcost <= new_cost:
                        continue
                else:
                    if use_heuristic:
                        heuristic = G.nodes[child]['heuristic']
                    else:
                        heuristic = 0
                    
                prev_queued[child] = new_cost, heuristic
                new_priority = new_cost + heuristic
                queue.append((new_priority, child, new_cost, node))
            
            queue.sort(key = lambda x: x[0]) # sort queue by priority
        
        # If goal never found, return None for both path and distance
        return None,None
    
    def search(self):
        count = 0
        while (not self.solution_found) and (count < self.max_iter):
            self.gen_samples()
            count += 1

        if self.solution_found:
            self.path, self.distance = self.a_star(self.Graph)
            assert self.path is not None and self.distance is not None

    def plot(self,figsize=(20,10), labels=False, node_size=25, path_only=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if self.path is not None:
            colors = ['green' if node in self.path else 'blue' for node in self.Graph.nodes]
            title = f"Path Length: {round(self.distance, 3)}"
        else:
            colors = 'blue'
            title = None
        
        ax.set_title(title, fontsize=20)

        if path_only and self.path is not None:
            nx.draw(self.Graph,
                {i:self.Graph.nodes[i]['pos'] if i in self.path else None for i in range(len(self.Graph.nodes))},
                node_size=node_size,
                nodelist=self.path,
                edgelist = [(self.path[i],self.path[i+1]) for i in range(len(self.path)-1)],
                node_color = 'green',
                ax = ax
            )
        else:
            
            nx.draw(self.Graph, 
                {i:self.Graph.nodes[i]['pos'] for i in range(self._c)}, 
                with_labels=labels, 
                ax=ax, 
                node_size=node_size,
                node_color = colors
            )
        ax.set_aspect('equal')
        plt.draw()
        
        ax.add_patch(Circle(self.qgoal,self.eps,fill=False, ec='green'))
        
        for obs in self.obstacles:
            ax.add_patch(Rectangle(obs.xy,obs.sx, obs.sy, fc='red'))
        
        return fig, ax



def load_env(id):
    if id.lower() == 'a':
        q0 = (0,0)
        qgoal = (10,0)
        obs1 = Obstacle((4,1),c=True)
        obs2 = Obstacle((7,-1),c=True)
        obstacles = [obs1,obs2]
        xlim=[-1,11]
        ylim=[-3,3]
        return q0, qgoal, obstacles, xlim, ylim

    elif id.lower() == 'b1':
        q0 = (0,0)
        qgoal = (10,10)
        obs1 = Obstacle((1,1),1,4,c=False)
        obs2 = Obstacle((3,4),1,8,c=False)
        obs3 = Obstacle((3,12),9,1,c=False)
        obs4 = Obstacle((12,5),1,8,c=False)
        obs5 = Obstacle((6,5),6,1,c=False)
        obstacles = [obs1,obs2,obs3,obs4,obs5]
        xlim=[-1,13]
        ylim =[-1,13]
        return q0, qgoal, obstacles, xlim, ylim

    elif id.lower() == 'b2':
        q0 = (0,0)
        qgoal = (35,0)
        obs1 = Obstacle((-6,-6),25-(-6),1, c=False)
        obs2 = Obstacle((-6,5),30-(-6),1, c=False)
        obs3 = Obstacle((-6,-5),1,10, c=False)
        obs4 = Obstacle((4,-5),1,6, c=False)
        obs5 = Obstacle((9,0),1,5, c=False)
        obs6 = Obstacle((14,-5),1,6, c=False)
        obs7 = Obstacle((19,0),1,5, c=False)
        obs8 = Obstacle((24,-5),1,6, c=False)
        obs9 = Obstacle((29,0),1,5, c=False)
        obstacles = [obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8,obs9]
        xlim=[-6,36]
        ylim =[-6,6]
        return q0, qgoal, obstacles, xlim, ylim
    else:
        raise ValueError("Available Environments are [A,B1,B2]")
