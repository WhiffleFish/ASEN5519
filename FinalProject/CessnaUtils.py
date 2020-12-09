# from networkx.generators import random_clustered
import numpy as np
import logging 
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import networkx as nx
import tqdm
from tqdm import trange
from pyfme.aircrafts import Cessna172
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.wind import NoWind
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment import Environment
from pyfme.utils.trimmer import steady_state_trim
from pyfme.models.state.position import EarthPosition
from pyfme.models import EulerFlatEarth
from pyfme.utils.input_generator import Constant
from pyfme.simulator import Simulation
from IPython.display import clear_output


class CessnaModel:
    aircraft = Cessna172()
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    def __init__(self, x0, y0, h0, v0, psi0=0.5):
        self.x0 = x0
        self.y0 = y0
        self.h0 = h0
        self.v0 = v0
        self.pos = EarthPosition(x=x0, y=y0, height=h0)
        controls0 = {'delta_elevator': 0, 'delta_aileron': 0, 'delta_rudder': 0, 'delta_t': 0.5}

        # Trim Aircraft as IC
        trimmed_state, trimmed_controls = steady_state_trim(
            self.aircraft,
            self.environment,
            self.pos,
            psi0,
            v0,
            controls0
        )
        self.environment.update(trimmed_state)
        self.state = trimmed_state
        self.system = EulerFlatEarth(t0=0, full_state=trimmed_state)
        self.solution_found = False

    def rand_action(self):
        lim_delta_e = self.aircraft.control_limits['delta_elevator']
        e_range = lim_delta_e[1] - lim_delta_e[0]
        lim_delta_a = self.aircraft.control_limits['delta_aileron']
        a_range = lim_delta_a[1] - lim_delta_a[0]
        lim_delta_r = self.aircraft.control_limits['delta_rudder']
        r_range = lim_delta_r[1] - lim_delta_r[0]
        lim_delta_t = self.aircraft.control_limits['delta_t']
        t_range = lim_delta_t[1] - lim_delta_t[0]

        u = np.random.rand(4)*np.array([e_range,a_range, r_range, t_range])

        return u + np.array([lim_delta_e[0],lim_delta_a[0], lim_delta_r[0], lim_delta_t[0]])


    def simulate(self, Tprop, controls, init_state=None):
        if init_state is None:
            init_state = self.state
        
        if not isinstance(controls, dict):
            controls = self.format_controls(*controls)

        system = EulerFlatEarth(t0=0, full_state=init_state)
        sim = Simulation(self.aircraft, system, self.environment, controls)
        result = sim.propagate(Tprop)
        clear_output()
        return result, sim # Changed from just returning result

    @staticmethod
    def positions_from_sim(results):
        '''results are sim position results after propagation'''
        return results[['x_earth','height']].to_numpy()

    def format_controls(self, delta_e, delta_a, delta_r, delta_t):
        '''
        Formats controls into pyFME compatible dict
        '''
        lim_delta_e = self.aircraft.control_limits['delta_elevator']
        lim_delta_a = self.aircraft.control_limits['delta_aileron']
        lim_delta_r = self.aircraft.control_limits['delta_rudder']
        lim_delta_t = self.aircraft.control_limits['delta_t']

        # Ensure control bounds aren't exceeded
        delta_e = max(min(lim_delta_e[1], delta_e),lim_delta_e[0])
        delta_a = max(min(lim_delta_a[1], delta_a),lim_delta_a[0])
        delta_r = max(min(lim_delta_r[1], delta_r),lim_delta_r[0])
        delta_t = max(min(lim_delta_t[1], delta_t),lim_delta_t[0])

        return {
            'delta_elevator': Constant(delta_e),
            'delta_aileron': Constant(delta_a),
            'delta_rudder': Constant(delta_r),
            'delta_t': Constant(delta_t)
            }
    
    @staticmethod
    def pos_from_state(state):
        return np.array([state.position.x_earth, state.position.height])
        

class FullSimRRT:
    def __init__(self, model, obstacles, eps, xlim, ylim, qgoal, max_iter=10):
        self.model = model
        self.x0 = model.x0
        self.y0 = model.y0
        self.h0 = model.h0
        self.v0 = model.v0
        self.obstacles = obstacles
        self.eps = eps
        self.xlim = xlim
        self.ylim = ylim
        self.qgoal = qgoal

        self.Graph = nx.DiGraph()
        self.Graph.add_nodes_from([(1,dict(
            pos=(model.pos_from_state(model.state)),
            state=model.state,
            ))]
            )
        self._c = 1 # Node counter
        self.solution_found = False


    @staticmethod
    def get_distance(p1,p2):
        return np.linalg.norm(p1-p2)


    def dist_to_goal(self, pt):
        return np.linalg.norm(self.qgoal-pt)


    def is_solution(self,pt):
        return  self.dist_to_goal(pt) <= self.eps


    def sample_pos(self):
        while True:
            qrand = np.random.rand(2)
            qrand[0] = (self.xlim[1]- self.xlim[0])*qrand[0] + self.xlim[0]
            qrand[1] = (self.ylim[1]- self.ylim[0])*qrand[1] + self.ylim[0]

            if not self.obstacles or all(not obs.is_colliding(qrand) for obs in self.obstacles):
                return qrand


    def find_closest_node(self, qrand):
        # Use Angle off as distance metric as well
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

        return qnear, min_dist_node


    def is_inbounds(self,pt):
        return (self.xlim[0] <= pt[0] <= self.xlim[1]) and (self.ylim[0] <= pt[1] <= self.ylim[1])


    def check_path(self, arr):
        '''
        Check for collision or solution
        Input array has columns [x_earth, height]
        Returns array up to end point , goal_found_flag
        '''
        for i in range(arr.shape[0]-1):
            #Check Solution
            if self.is_solution(arr[i+1]):
                return arr[:i+2], True
            
            # Check Bounds
            if not self.is_inbounds(arr[i+1]):
                return None, False

            # Check Collision
            for obs in self.obstacles:
                if obs.is_segment_intersecting(arr[i],arr[i+1]):
                    return None, False
        
        return arr, False

    
    def get_solution_trajectory(self, end_node):
        child = end_node
        node_hist = [end_node]
        parent = list(self.Graph.predecessors(end_node))[0]
        path = self.model.positions_from_sim(self.Graph[parent][child]['state_hist'])
        ctrl = self.Graph[parent][child]['action']
        child = parent
        node_hist.append(parent)

        X_hist = path
        u_hist = ctrl
        while child != 1:
            parent = list(self.Graph.predecessors(child))[0]
            path = self.model.positions_from_sim(self.Graph[parent][child]['state_hist'])
            ctrl = self.Graph[parent][child]['action']
            X_hist = np.vstack([path,X_hist])
            u_hist = np.vstack([ctrl,u_hist])
            child = parent
            node_hist.append(parent)

        return reversed(node_hist), X_hist, u_hist

    def single_sample(self, Tprop, N_tries):
        qrand = self.sample_pos()
        qnear, min_dist_node = self.find_closest_node(qrand)
        
        goal_found = False

        best_action = None
        best_endstate = None
        best_result = None
        best_endpos = None
        best_dist = float('inf')


        for i in range(N_tries):
            a = self.model.rand_action()
            try:
                result, sim = self.model.simulate(Tprop, a, self.Graph.nodes[min_dist_node]['state'])
            except ValueError:
                continue
            path = self.model.positions_from_sim(result)
            endstate = sim.system.full_state
            arr, flag = self.check_path(path)
            if arr is None:
                continue
            
            end_pos = path[-1]
            dist = self.get_distance(end_pos, qrand)
            
            if flag:
                goal_found = True
                best_action = a
                best_endstate = endstate
                best_result = result
                best_endpos = end_pos
                best_dist = dist
                continue

            if dist < best_dist:
                best_action = a
                best_endstate = endstate
                best_result = result
                best_endpos = end_pos
                best_dist = dist
        
        return best_result, best_endpos, best_endstate, best_action, goal_found, min_dist_node


    def gen_samples(self, Tprop, N_tries, N=None, status=True):
        """
        Tprop - Dynamics Propagation Duration
        N_tries - Number of tries to get close to qrand
        N - Number of sampling iterations
        """
        if N is None:
            N = self.max_iter

        for i in trange(N, disable=not status):
            best_result, best_endpos, best_endstate, best_action, goal_found, prev_node = self.single_sample(Tprop,N_tries)
            if goal_found:
                self.Graph.add_node(
                    self._c + 1,
                    pos = best_endpos,
                    state = best_endstate
                )
                self.Graph.add_edge(prev_node, self._c+1,
                    state_hist = best_result,
                    action = best_action
                )
                self._c += 1
                self.solution_found = True
                node_hist, X_hist, u_hist = self.get_solution_trajectory(self._c)
                self.sol_node_hist = node_hist
                self.sol_X_hist = X_hist
                self.sol_u_hist = u_hist
                break
            elif best_result is None:
                continue

            self.Graph.add_node(
                    self._c + 1,
                    pos = best_endpos,
                    state = best_endstate
                )
            self.Graph.add_edge(prev_node, self._c+1,
                    state_hist = best_result,
                    action = best_action
                )
            self._c += 1

        
    def plot_path(self, figsize= (12,7), solution=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if solution:
            ax.plot(self.sol_X_hist[:,0],self.sol_X_hist[:,1])
        else:
            for i,j in self.Graph.edges:
                X = self.model.positions_from_sim(self.Graph[i][j]['state_hist'])
                ax.plot(X[:,0], X[:,1], c='blue')
        
        ax.add_patch(Circle(self.qgoal, self.eps, fill=False, ec='green'))
        for obs in self.obstacles:
            ax.add_patch(Rectangle(obs.xy,obs.sx, obs.sy, fc='red'))

        return fig, ax
