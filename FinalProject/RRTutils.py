import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, Rectangle
from numpy import cos, matmul, sin, zeros
from scipy.integrate import odeint
from scipy.linalg import expm
from tqdm import tqdm, trange


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

class RECUV:
    def __init__(self, dt=3, N=100, vel = 1):
        self.dt = dt
        self.N = N
        self.vel = vel
        self.A = np.array(
            [[-0.3206, 0.3514, 0, -9.8048, 0, 0],
            [-0.9338,-6.8632,19.8386,-0.3187, 0, -0.0004],
            [0,-2.5316,-3.9277, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0,-21,0, 0]]
        )
        
        self.B = np.array(
            [[0.0037, 11.2479],
            [-0.1567, 0],
            [-0.9332, 0],
            [0, 0],
            [0, 0],
            [0, 0]]
        )
        
        self.C = np.array(
            [[0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1]]
        )
        
        self.D = zeros((2,2))

        # Grammian integration steps independent from dynamics propagation steps
        self.update_grammian(dt,500) 

    def update_grammian(self, dt, N):
        def gram_integrand(A,B,t):
            return expm(-A*t) @ B @ B.T @ expm(-A.T*t)
        
        x = np.linspace(0,dt,N)
        self.G = np.trapz(np.array([gram_integrand(self.A,self.B,i) for i in x]), x=x,axis=0)


    def get_path(self,x0,x1):
        assert (x0.shape == (6,1))
        assert (x1.shape == (6,1))

        zeta = matmul(expm(-self.A*self.dt),x1) - x0
        r = np.linalg.inv(self.G) @ zeta

        def ufunc(A,B,r,t):
            return B.T @ expm(-A.T*t) @ r

        t_arr = np.linspace(0, self.dt, self.N)

        us = np.array([np.squeeze(ufunc(self.A,self.B, r, i)) for i in t_arr])
        
        def model(x, t, A, B, r, ufunc):
            return np.squeeze(A @ x.reshape(-1,1) + B @ ufunc(A, B, r, t))

        y = odeint(model,np.squeeze(x0),t_arr, args=(self.A,self.B, r, ufunc))

        return y, us


    def get_rand_path(self,x0, u1lim, u2lim,Tprop=None):
        '''
        Random control zero-order-hold (ZOH) dynamics propagation
        '''
        if Tprop is None:
            Tprop = self.dt
        
        t_arr = np.linspace(0,Tprop,self.N)
        
        def model(x, A, B, u):
            return np.squeeze(A @ x.reshape(-1,1) + B @ u)
        
        
        u = np.random.rand(2,1)
        u[0] = (u1lim[1]- u1lim[0])*u[0] + u1lim[0]
        u[1] = (u2lim[1]- u2lim[0])*u[0] + u2lim[0]

        X = odeint(model,np.squeeze(x0),t_arr, args=(self.A,self.B, u))
        return X, u


class KinodynamicRRT2D:

    def __init__(self, model, x0, qgoal, obstacles, eps, xlim, ylim, step_size, pgoal = 0.05, max_iter=10):
        self.rand_config = True # Sample from config space randomly
        
        self.model = model

        # x = [u w q theta x z]
        self.x0 = x0
        self.qstart = np.squeeze(x0[-2:])
        self.qgoal = np.array(qgoal)
        self.obstacles = obstacles
        self.eps = eps
        self.xlim = xlim
        self.ylim = ylim
        self.step_size = step_size
        self.max_iter = max_iter
        self.pgoal = pgoal

        self.Graph = nx.DiGraph()

        self.Graph.add_nodes_from([(1,dict(
            pos=(self.qstart),
            state=x0,
            coll = False
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

    # Move to RRT solve
    def sanitize_collision(self, X, u):
        '''
        Truncates path to where collision occurs, if one does occur
        '''
        x_arr, z_arr = X[:,-2],X[:,-1]
        intersect = False

        for i in range(len(x_arr)-1):

            if self.is_solution((x_arr[i+1],z_arr[i+1])):
                    return X[:i+2,:],u[:i+2,:], True, False
            # Out of Bounds
            elif not (self.xlim[0] <= x_arr[i+1] <= self.xlim[1] and self.ylim[0] <= z_arr[i+1] <= self.ylim[1]):
                return X[:i+1,:],u[:i+1,:], False, True

            for obstacle in self.obstacles:
                if obstacle.is_segment_intersecting((x_arr[i],z_arr[i]),(x_arr[i+1],z_arr[i+1])):
                    intersect = True
                    return X[:i+1,:],u[:i+1,:], False, True
                
        # return states, actions, goal_found_flag, collision_flag
        return X, u, False, False


    def find_closest_node(self,qrand):
        # Find closest point on current tree
        min_dist = float('inf')
        min_dist_node = None
        qnear = None

        for node in self.Graph.nodes:
            dist = np.linalg.norm(qrand-self.Graph.nodes[node]['pos'])
            if dist < min_dist and not self.Graph.nodes[node]['coll']:
                min_dist = dist
                qnear = self.Graph.nodes[node]['pos']
                min_dist_node = node

        assert (min_dist < float('inf')) and (qnear is not None)

        return qnear, min_dist_node

    def single_rand_sample(self, Tprop, u1lim=[-np.pi/3,np.pi/3], u2lim=[-np.pi/3,np.pi/3]):
        # Find random node in Tree
        node = np.random.choice(self.Graph.nodes)
        x0 = self.Graph.nodes[node]['state']
        # Propagate Random Dynamics
        X, u = self.model.get_rand_path(x0, u1lim, u2lim)

        X, u, sol_flag, col_flag = self.sanitize_collision(X,u)

        return node, X, u, sol_flag, col_flag


    def single_det_sample(self):
        '''
        Stochastically sample state space; deterministic path to new sample
        '''
        if np.random.rand() <= self.pgoal:
            qrand = self.qgoal
        else:
            while True:
                x_range = self.xlim[1] - self.xlim[0]
                y_range = self.ylim[1] - self.ylim[0]
                qrand = np.random.rand(2)*np.array([x_range, y_range]) + np.array([self.xlim[0],self.ylim[0]])
                if not self.obstacles or not all([obs.is_colliding(qrand) for obs in self.obstacles]):
                    break

        qnear, min_dist_node = self.find_closest_node(qrand)
        
        # Find angle between points:
        dx,dy = qrand - qnear
        theta = np.arctan2(dy,dx)
        qnew = qnear
        segment_length = self.step_size

        if self.get_distance(qnew, qrand) < self.step_size:
            x_new, y_new = qrand[0], qrand[1]
            segment_length = self.get_distance(qnew, qrand)
            end = True
        else:
            x_new = qnew[0] + self.step_size*cos(theta)
            y_new = qnew[1] + self.step_size*sin(theta)
        if self.rand_config:
            theta = np.random.rand()*2 - 1 
            u = np.random.rand()*2 - 1
            v = np.random.rand()*2 - 1
            q = np.random.rand()*2 - 1
            x1 = np.array([u,v,q,theta,x_new,y_new]).reshape(6,1)
        else:
            theta_bound = np.pi/4
            theta_p = max(min(theta,theta_bound),-theta_bound) # theta_p = theta for prev results
            u = self.model.vel*cos(theta_p)
            v = self.model.vel*sin(theta_p)
            x1 = np.array([u,v,0,theta_p,x_new,y_new]).reshape(6,1)

        x0 = self.Graph.nodes[min_dist_node]['state'].reshape(6,1)
        
        X, u = self.model.get_path(x0,x1)
        X, u, sol_flag, col_flag = self.sanitize_collision(X,u)
        return  min_dist_node, X, u, sol_flag, col_flag


    def get_solution_trajectory(self, end_node):
        
        child = end_node
        node_hist = [end_node]
        parent = list(self.Graph.predecessors(end_node))[0]
        path = self.Graph[parent][child]['state_hist']
        ctrl = self.Graph[parent][child]['ctrl_hist']
        child = parent
        node_hist.append(parent)

        X_hist = path
        u_hist = ctrl
        while child != 1:
            parent = list(self.Graph.predecessors(child))[0]
            path = self.Graph[parent][child]['state_hist']
            ctrl = self.Graph[parent][child]['ctrl_hist']
            X_hist = np.vstack([path,X_hist])
            u_hist = np.vstack([ctrl,u_hist])
            child = parent
            node_hist.append(parent)

        return reversed(node_hist), X_hist, u_hist


    def gen_samples(self,N=None):
        if N is None:
            N = self.max_iter
        # Iterate until goal is found or max_iter reached

        for i in trange(N):
            prev_node, X, u, goal_flag, col_flag = self.single_det_sample()
            self.Graph.add_node(
                self._c+1,
                pos = X[-1,-2:],
                state = X[-1,:],
                coll = col_flag
            )
            self.Graph.add_edge(prev_node, self._c+1,
                state_hist = X,
                ctrl_hist = u
            )
            self._c += 1
            if goal_flag:
                self.solution_found = True
                # Get solution trajectory
                node_hist, X_hist, u_hist = self.get_solution_trajectory(self._c)
                self.sol_node_hist = node_hist
                self.sol_X_hist = X_hist
                self.sol_u_hist = u_hist
                break

            
    def plot_path(self, figsize=(12,7), solution=True):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if solution:
            ax.plot(self.sol_X_hist[:,-2],self.sol_X_hist[:,-1])
        else:
            for i,j in self.Graph.edges:
                xhist = self.Graph[i][j]['state_hist'][:,-2]
                zhist = self.Graph[i][j]['state_hist'][:,-1]
                ax.plot(xhist,zhist,c='blue')
        
        ax.add_patch(Circle(self.qgoal, self.eps, fill=False, ec='green'))

        for obs in self.obstacles:
            ax.add_patch(Rectangle(obs.xy,obs.sx, obs.sy, fc='red'))
        
        return fig, ax
