import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Obstacle(object):
    def __init__(self, xy, sx=1, sy=1):
        self.xy = np.array(xy)
        self.sx = sx
        self.sy = sy


class WaveFront(object):
    def __init__(self, q0, qgoal, obstacles, pad=2, dx=0.25, obs_grid=None, x=None, y=None):
        self.q0 = np.array(q0)
        self.qgoal = np.array(qgoal)
        self.obstacles = obstacles
        self.dx = dx

        self.xlim, self.ylim = self.get_border_vals(q0,qgoal,obstacles, pad)

        self.x = np.arange(self.xlim[0],self.xlim[1]+dx,dx)
        self.y = np.arange(self.ylim[0],self.ylim[1]+dx,dx)
        self.X, self.Y = np.meshgrid(self.x,self.y)

        self.obs_grid = self.get_obstacle_grid(self.X, self.Y, obstacles)
        self.update()

        
    @staticmethod
    def get_border_vals(q0, qgoal, obstacles, pad):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for obs in obstacles:
            if obs.xy[0] < min_x:
                min_x = obs.xy[0]
            if obs.xy[0]+obs.sx > max_x:
                max_x = obs.xy[0]+obs.sx
            if obs.xy[1] < min_y:
                min_y = obs.xy[1]
            if obs.xy[1]+obs.sy > max_y:
                max_y = obs.xy[1]+obs.sy

        min_pts = np.minimum(q0, qgoal)
        max_pts = np.maximum(q0, qgoal)
        if min_pts[0] < min_x:
            min_x = min_pts[0]
        if min_pts[1] < min_y:
            min_y = min_pts[1]
        if max_pts[0] > max_x:
            max_x = max_pts[0]
        if max_pts[1] > max_y:
            max_y = max_pts[1]

        xlim = [min_x-pad, max_x+pad]
        ylim = [min_y-pad, max_y+pad]

        return xlim, ylim

    @staticmethod
    def get_obstacle_grid(X,Y,obstacles):

        obs_grid = np.zeros(X.shape, dtype=np.int32)
        for obs in obstacles:
            X_bool = (obs.xy[0]<=X) & (X<= (obs.xy[0]+obs.sx))
            Y_bool = (obs.xy[1]<=Y) & (Y<= (obs.xy[1]+obs.sy))
            obs_grid[X_bool*Y_bool] = 1

        return obs_grid

    def fill_grid(self):
        self.WF_grid = self.obs_grid.copy()
        self.WF_grid[tuple(self.qgoal)]
        self.WF_grid = self.wave_fill(self.WF_grid.copy(), [tuple(self.qgoal)])
        
    def update(self):
        def is_open(G,p):
                return ([0,0] <= p).all() and (p < G.shape).all() and G[tuple(p)] == 0

        def wave_fill(G, pts, n=2):
            new_pts = []
            for p in pts:
                for k in np.array([[0,1],[1,0],[-1,0],[0,-1]])+np.array(p):
                    if is_open(G,k):
                        new_pts.append(tuple(k))
                        G[tuple(k)] = n+1
            
            if new_pts: # Recurse
                return wave_fill(G,new_pts,n+1)
            else:
                return G

        def path_from_fill(G,q):
            path = [q]
            while G[q] != 2:
                for k in np.array([[0,1],[1,0],[-1,0],[0,-1]]) + np.array(q):
                    k = tuple(k)
                    if G[k] == G[q]-1:
                        path.append(k)
                        q = k
                        break

            return path

        # Instantiate WaveFront value grid
        self.WF_grid = self.obs_grid.copy()
        # Set goal grid value to 2
        self.WF_grid[(self.X == self.qgoal[0]) & (self.Y == self.qgoal[1])] = 2
        # Starting coordinate (qg_gp) for WaveFront is qgoal grid position with max starting board value
        qg_gp = np.unravel_index(np.argmax(self.WF_grid, axis=None), self.WF_grid.shape)
        # Apply WaveFill to WF_grid
        self.WF_grid = wave_fill(self.WF_grid, [qg_gp])

        # Get q0 grid position and descend WaveFront grid to get to qgoal
        q0_gp = tuple(np.argwhere((self.X == self.q0[0]) & (self.Y == self.q0[1]))[0])
        path = path_from_fill(self.WF_grid,q0_gp)

        # Convert path values from array indices to coordinate positions
        self.path = np.array([np.array([self.x[pt[1]],self.y[pt[0]]]) for pt in path])
        self.path_length = (len(self.path)-1)*self.dx


    def plot_path(self,figsize=(12,7)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.obs_grid, origin='lower', extent=[*self.xlim,*self.ylim], cmap='gray_r')
        ax.plot(self.path[:,0],self.path[:,1], 'g--')
        ax.add_patch(Circle(self.qgoal,0.5,fill=False, ec='red'))
        ax.set_title("WaveFront Path")
        return fig, ax

    def plot_wavefront(self,figsize=(12,7)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.WF_grid, origin='lower', extent=[*self.xlim,*self.ylim])
        ax.set_title("WaveFront Value Heatmap")
        return fig, ax
        