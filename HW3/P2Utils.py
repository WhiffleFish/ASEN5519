import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class Obstacle(object):
    def __init__(self, xy, sx=1, sy=1, c=True, eta=0.1, Q_star=5):
        if c: # If centerpoint given, convert to lower-left coordinate position
            self.xy = np.array(xy) - np.array([0.5*sx,0.5*sy])
        else:
            self.xy = xy
        
        self.sx = sx
        self.sy = sy
        self.eta = eta
        self.Q_star = Q_star

    def dist_to_pt(self, q):
        x, y = q

        min_x = self.xy[0]
        min_y = self.xy[1]
        max_x = self.xy[0] + self.sx
        max_y = self.xy[1] + self.sy

        # If point touching box boundaries or inside box, return distance = 0
        if (min_x <= x <= max_x) and (min_y <= y <= max_y):
            return 0

        dx = max(min_x - x, 0 , x - max_x)
        dy = max(min_y - y, 0 , y - max_y)

        return np.linalg.norm([dx,dy])

    def closest_point_on_obs(self, q):

        x, y = q

        min_x = self.xy[0]
        min_y = self.xy[1]
        max_x = self.xy[0] + self.sx
        max_y = self.xy[1] + self.sy

        # If point touching box boundaries or inside box, return distance = 0
        if (min_x <= x <= max_x) and (min_y <= y <= max_y):
            return 0

        if x > max_x:
            if y > max_y:
                return np.array([max_x,max_y])
            elif (min_y <= y <= max_y):
                return np.array([max_x,y])
            else:
                return np.array([max_x,min_y])
        elif (min_x <= x <= max_x):
            if y > max_y:
                return np.array([x,max_y])
            elif (min_y <= y <= max_y):
                return np.array([x,y])
            else:
                return np.array([x,min_y])
        else:
            if y > max_y:
                return np.array([min_x,max_y])
            elif (min_y <= y <= max_y):
                return np.array([min_x,y])
            else:
                return np.array([min_x,min_y])

    def Urep(self, q):
        dq = self.dist_to_pt(q)
        if 0 < dq <= self.Q_star:
            return 0.5*self.eta*np.square((1/self.Q_star) - (1/dq))
        else:
            return 0

    def Urep_grad(self, q):
        dq = self.dist_to_pt(q)
        if 0 < dq <= self.Q_star:
            qobs = self.closest_point_on_obs(q)
            return self.eta*((1/self.Q_star) - (1/dq))*(q-qobs)/np.square(dq)
        else:
            return np.array([0,0])

    def Urep_grad_grid(self, xlim, ylim, N):
        X,Y = np.meshgrid(np.linspace(*xlim,N),np.linspace(*ylim,N))
        U = np.zeros((len(X),len(Y)))
        V = np.zeros((len(X),len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                U[j,i],V[j,i] = self.Urep_grad((X[0,i],Y[j,0]))

        return X,Y,U,V


class RectGradDescent(object):
    def __init__(self, q0, qgoal, obstacles, d_star=5, zeta=0.1, eps=0.25, N=50):
        self.q0 = np.array(q0,dtype=np.float64)
        self.qgoal = np.array(qgoal)
        self.obstacles = obstacles
        self.d_star = d_star
        self.zeta = zeta
        self.eps = eps
        self.N = N

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

        self.xlim = [min_x-2, max_x+2]
        self.ylim = [min_y-2, max_y+2]

        self.update_vectorfield()
        self.update_path()


    def update_vectorfield(self):
        X,Y,U,V = self.Uatt_grad_grid(self.xlim, self.ylim, self.N)
        for obs in self.obstacles:
            _,_,Ui,Vi = obs.Urep_grad_grid(self.xlim, self.ylim, self.N)
            U += Ui
            V += Vi

        self.v_field = (X, Y, U, V)

    @staticmethod
    def distance(q1, q2):
        """
        Distance between points
        """
        return np.linalg.norm(q1-q2)

    def Uatt(self, q):
        """
        Attraction Potential
        """
        dist = self.distance(q, self.qgoal)
        if dist <= self.d_star:
            return 0.5*self.zeta*np.square(dist)
        else:
            return self.d_star*self.zeta*dist-0.5*self.zeta*self.d_star**2

    def Uatt_grad(self, q):
        dist = self.distance(q, self.qgoal)
        if dist <= self.d_star:
            return self.zeta*(q-self.qgoal)
        else:
            return self.d_star*self.zeta*(q-self.qgoal)/dist

    def Uatt_grad_grid(self, xlim, ylim, N):
        X, Y = np.meshgrid(np.linspace(*self.xlim, N), np.linspace(*self.ylim, N))
        U = np.zeros((len(X), len(Y)))
        V = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(X)):
                U[j,i],V[j,i] = self.Uatt_grad((X[0,i], Y[j,0]))

        return X, Y, U, V

    def Urep(self, obs, q):
        dq = obs.dist_to_pt(q)
        if 0 < dq <= obs.Q_star:
            return 0.5*obs.eta*np.square((1/obs.Q_star) - (1/dq))
        else:
            return 0

    def update_path(self):

        q = self.q0.copy()
        path = self.q0.copy()

        length = 0
        i = 0 # timeout
        while (self.distance(q,self.qgoal) > self.eps) and (i<=50):
            i += 1
            U_grad = self.Uatt_grad(q)
            for obs in self.obstacles:
                U_grad += obs.Urep_grad(q)
            
            q -= U_grad
            length += np.linalg.norm(U_grad)
            path = np.vstack((path,q))

        self.path, self.path_length = path, length


    def plot_v_field(self, figsize=(15,10), path=False):
        self.update_vectorfield()

        fig, ax = plt.subplots(figsize=figsize)
        X, Y, U, V = self.v_field
        
        _, _, Ua, Va = self.Uatt_grad_grid(self.xlim, self.ylim, self.N)
        thresh = np.sqrt(np.square(Ua) + np.square(Va)).max()*1.2
        weight_map = thresh/np.sqrt(np.square(U) + np.square(V))
        weight_map[weight_map > 1] = 1

        ax.quiver(X,Y,-U*weight_map,-V*weight_map, headwidth=2, headlength=2)
        for obs in self.obstacles:
            ax.add_patch(Rectangle(obs.xy,obs.sx, obs.sy))
        
        ax.add_patch(Circle(self.qgoal,self.eps, fill=False, ec='red'))
        ax.set_aspect(1)
        
        if path:
            self.update_path()
            ax.plot(*self.path.T, 'g--')
        return fig, ax


    def plot_path(self, figsize=(15,10)):
        self.update_path()
        fig, ax = plt.subplots(figsize=figsize)

        for obs in self.obstacles:
            ax.add_patch(Rectangle(obs.xy,obs.sx, obs.sy))
        
        ax.add_patch(Circle(self.qgoal,self.eps, fill=False, ec='red'))
        ax.set_aspect(1)
        
        ax.plot(*self.path.T, 'g--')

        return fig, ax
