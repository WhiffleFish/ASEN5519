import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np


def distance(q1,q2):
    """
    Distance between points
    """
    return np.linalg.norm(q1-q2)


def dist_to_obs(q, obs, s=1):
    """
    Distance to square obstacle
    """
    x, y = q

    min_x = obs[0] - 0.5*s
    min_y = obs[1] - 0.5*s
    max_x = obs[0] + 0.5*s
    max_y = obs[1] + 0.5*s

    # If point touching box boundaries or inside box, return distance = 0
    if (min_x <= x <= max_x) and (min_y <= y <= max_y):
        return 0

    dx = max(min_x - x, 0 , x - max_x)
    dy = max(min_y - y, 0 , y - max_y)

    return np.linalg.norm([dx,dy])


def closest_point_on_obs(q, obs, sx=1, sy=1):
    x, y = q

    min_x = obs[0] - 0.5*sx
    min_y = obs[1] - 0.5*sy
    max_x = obs[0] + 0.5*sx
    max_y = obs[1] + 0.5*sy

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


def Uatt(q1, qgoal, d_star=5, z=1):
    """
    Attraction Potential
    """
    dist = distance(q1,qgoal)
    if dist <= d_star:
        return 0.5*z*np.square(dist)
    else:
        return d_star*z*dist-0.5*z*d_star**2


def Uatt_grid(qgoal, xlim, ylim, N, z=1):
    X,Y = np.meshgrid(np.linspace(*xlim,N),np.linspace(*ylim,N))
    grid = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            grid[j,i] = Uatt((X[0,i],Y[j,0]), qgoal, z)

    return X, Y, grid


def Uatt_grad(q, qgoal, d_star=5, z=1):
    dist = distance(q,qgoal)
    if dist <= d_star:
        return z*(q-qgoal)
    else:
        return d_star*z*(q-qgoal)/dist


def Uatt_grad_grid(qgoal,xlim, ylim, N, d_star=5, z=1):
    X,Y = np.meshgrid(np.linspace(*xlim,N),np.linspace(*ylim,N))
    U = np.zeros((len(X),len(Y)))
    V = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(X)):
            U[j,i],V[j,i] = Uatt_grad((X[0,i],Y[j,0]), qgoal, d_star, z)

    return X,Y,U,V


def Urep(q, obs, eta, Q_star):
    dq = dist_to_obs(q,obs)
    if 0 < dq <= Q_star:
        return 0.5*eta*np.square((1/Q_star) - (1/dq))
    else:
        return 0


def Urep_grid(obs,eta,Q_star,xlim, ylim, N):
    X,Y = np.meshgrid(np.linspace(*xlim,N),np.linspace(*ylim,N))
    grid = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            grid[j,i] = Urep((X[0,i],Y[j,0]), obs, eta, Q_star)

    grid[grid == 0] = grid.max()
    return X, Y, grid


def Urep_grad(q, obs, eta, Q_star):
    dq = dist_to_obs(q,obs)
    if 0 < dq <= Q_star:
        qobs = closest_point_on_obs(q, obs)
        return eta*((1/Q_star) - (1/dq))*(q-qobs)/np.square(dist_to_obs(q,obs))
    else:
        return np.array([0,0])


def Urep_grad_grid(qobs, eta, Q_star, xlim, ylim, N):
    X,Y = np.meshgrid(np.linspace(*xlim,N),np.linspace(*ylim,N))
    U = np.zeros((len(X),len(Y)))
    V = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            U[j,i],V[j,i] = Urep_grad((X[0,i],Y[j,0]), qobs, eta, Q_star)

    return X,Y,U,V


def get_path(q0=np.array([0,0],dtype=np.float64), qgoal=np.array([10,0]), eta=0.1, Q_star=5, z=0.1, d_star=5):
    qobs1 = np.array([4,1])
    qobs2 = np.array([7,-1])
    eps = 0.25

    q = q0.copy()
    path = q0.copy()
    length = 0
    while distance(q,qgoal) > eps:
        Utot_grad = Uatt_grad(q,qgoal,d_star,z) + Urep_grad(q, qobs1, eta, Q_star) + Urep_grad(q, qobs2, eta, Q_star)
        q -= Utot_grad
        length += np.linalg.norm(Utot_grad)
        path = np.vstack((path,q))

    return path, length


def get_vectorfield():
    qobs1 = np.array([4,1])
    qobs2 = np.array([7,-1])
    qgoal = np.array([10,0])
    xlim = [-1,11]
    ylim = [-4,4]
    eta = 1
    Q_star = 1
    N = 50

    X,Y,Uobs1,Vobs1 = Urep_grad_grid(qobs1, eta=eta, Q_star=Q_star,xlim=xlim,ylim=ylim, N=N)
    _,_,Uobs2,Vobs2 = Urep_grad_grid(qobs2, eta=eta, Q_star=Q_star,xlim=xlim,ylim=ylim, N=N)
    _,_, Ua, Va = Uatt_grad_grid(qgoal,xlim,ylim,N)
    Utot = Uobs1 + Uobs2 + Ua
    Vtot = Vobs1 + Vobs2 + Va
    return X,Y,Utot, Vtot


def plot_vectorfield(X,Y,Utot, Vtot):
    qobs1 = np.array([4,1])
    qobs2 = np.array([7,-1])
    qgoal = np.array([10,0])
    fig, ax = plt.subplots(figsize=(15,10))
    thresh = 10
    weight_map = thresh/np.sqrt(np.square(Utot) + np.square(Vtot))
    weight_map[weight_map > 1] = 1
    ax.quiver(X,Y,-Utot*weight_map,-Vtot*weight_map, headwidth=2)
    ax.add_patch(Rectangle(qobs1-0.5,1,1))
    ax.add_patch(Rectangle(qobs2-0.5,1,1))
    ax.add_patch(Circle(qgoal,0.25,fill=False, ec='red'))
    ax.set_aspect(1)

    return fig, ax


def plot_path(path):
    qobs1 = np.array([4,1])
    qobs2 = np.array([7,-1])
    qgoal = np.array([10,0])
    fig, ax = plt.subplots(figsize=(15,10))
    ax.add_patch(Rectangle(qobs1-0.5,1,1))
    ax.add_patch(Rectangle(qobs2-0.5,1,1))
    ax.add_patch(Circle(qgoal,0.25,fill=False, ec='red'))
    ax.set_aspect(1)
    ax.plot(*path.T, 'g--')

    return fig, ax