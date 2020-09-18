import logging
import numpy as np
import matplotlib.pyplot as plt
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def heading_from_points(p1, p2, deg=False):
    """Return heading angle from p1 to p2 """
    rel_dist = p2-p1
    theta = np.arctan2(rel_dist[1], rel_dist[0])
    if deg:
        return np.rad2deg(theta)
    else:
        return theta


def isPosValid(pos, W):
    """
    pos - current position
    W - Workspace (2d ndarray)
    OUTPUT:
        bool - is position in workspace valid
    """
    return not W[tuple(pos)]


def isMoveValid(pos, move, W):
    """
    pos - current position
    move - np.array([-1|0|1,-1|0|1])
    W - Workspace (2d ndarray)
    OUTPUT:
        bool - is move from current position valid
    """
    new_pos = pos + move
    return isPosValid(new_pos, W)


def add_pos(history, pos):
    """ Append pos(1x2) to history(nx2)"""
    assert history.shape[1] == 2, f"History shape invalid: {history.shape}"
    return np.append(history, pos.reshape(1,-1), axis=0)


def heading_to_goal(pos, qgoal, deg=False):
    """Determine heading angle from current position to goal position"""
    return heading_from_points(pos, qgoal, deg)


def current_heading_angle(movement_history, qgoal=None, deg=False):
    """
    Returns current heading angle based on "velocity vector"
    If no previous position given, heading determined by discretized heading to qgoal
    """
    if len(movement_history) <= 1:
        pos = movement_history[-1]
        return heading_to_goal(pos, qgoal, deg)
    else:
        p1 = movement_history[-2]
        p2 = movement_history[-1]
        return heading_from_points(p1, p2, deg)


def distance(p1, p2):
    """
    Returns Euclidean distance between points p1, and p2
    """
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1)
    if not isinstance(p2, np.ndarray):
        p2 = np.array(p2)

    return np.linalg.norm(p1-p2)


def MTG_move(pos, qgoal):
    """
    Unobstructed move-to-goal movement decision
    """
    theta = heading_to_goal(pos, qgoal)
    move = np.round(np.array([np.cos(theta), np.sin(theta)]))
    return move.astype(int)


def first_contact_move(pos, qgoal, W):
    heading = heading_to_goal(pos, qgoal)
    angles = np.deg2rad(np.arange(0, 360, 90))
    for theta in angles:
        if theta < heading:
            continue
        else:
            move = np.round(np.array([np.cos(theta), np.sin(theta)])).astype(int)
            if isMoveValid(pos, move, W):
                return move

    raise RuntimeError("No valid first contact move found")


def BF_move(pos, BF_history, qgoal, W):
    """
    Boundary following movement decision
    """
    heading = current_heading_angle(BF_history, qgoal, deg=True)
    angles = np.deg2rad(np.arange(heading-90, heading+180, 90))
    for theta in angles:
        move = np.round(np.array([np.cos(theta), np.sin(theta)])).astype(int)
        if isMoveValid(pos, move, W):
            return move

    raise RuntimeError("No valid Boundary Following move found")


def find_ql(BF_history, BF_distances, hist_loc=False):
    """
    Returns coordinates of qL -- point on obstacle with shortest distance to goal
    """
    assert len(BF_history) == len(BF_distances)
    index = np.argmin(BF_distances)
    qL = BF_history[index]

    if hist_loc:
        return index, qL
    else:
        return qL


def shortest_path_to_ql(BF_history, BF_distances):
    """
    Returns array containing coordinates for shortest path to qL.
    EXCLUDING current position, but INCLUDING qL
    """
    assert (BF_history[0] == BF_history[-1]).all(), (
        "{BF_history[0]} != {BF_history[1]} -- Boundary following path does not start and end at qH"
    )
    rev_index, rev_ql = find_ql(BF_history[::-1], BF_distances[::-1], hist_loc=True)
    index, qL = find_ql(BF_history, BF_distances, hist_loc=True)
    
    hist = BF_history.copy()
    
    if rev_index < index:
        index, qL = rev_index, rev_ql
        path = hist[::-1][1:index+1]
    else:
        path = hist[1:index+1]

    logging.debug(f"QL Path: \n{path}")
    assert (path[-1] == qL).all(), "End of shortest path to qL is not equal to qL"
    return path


def get_m_coords(qstart, qgoal):
    """
    INPUT:
        qstart, qgoal -- 1x2 np.ndarrays
    OUTPUT:
        m_line_coords -- nx2 ndarray of m-line coordinates
    """
    pos = qstart
    m_line_coords = pos.copy().reshape(1, -1)

    while (pos != qgoal).any():
        pos += MTG_move(pos, qgoal)
        m_line_coords = add_pos(m_line_coords, pos)

    return m_line_coords


def plot_path(path, W, cmap='viridis', extents=None, pad=None, cbar=True):
    fig, ax = plt.subplots(figsize=(15, 10))
    rgbs = plt.cm.get_cmap(cmap)(np.linspace(0, 1, path.shape[0]))
    
    if extents:
        xmin = extents[0]
        ymin = extents[2]
        extents = np.array(extents) + np.array([-pad-0.5, pad-0.5, -pad-0.5, pad-0.5])
        ax.imshow(W.T, cmap='gray_r', origin='lower', extent=list(extents))
        ax.plot(np.array(path)[:, 0]+(xmin-pad), np.array(path)[:, 1]+(ymin-pad))
        sct = ax.scatter(np.array(path)[:, 0]+(xmin-pad), np.array(path)[:, 1]+(ymin-pad), c=rgbs, alpha=0.8)    
    else:
        ax.imshow(W.T,cmap='gray_r', origin='lower')
        ax.plot(np.array(path)[:, 0], np.array(path)[:, 1])
        sct = ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], c=rgbs, alpha=0.8)

    if cbar:
        cbar = fig.colorbar(sct, ticks=[0,1], shrink=0.75)
        cbar.ax.set_yticklabels(['START', 'END'])
    return fig, ax


def workspace1_data():
    """
    Retrive (qstart, qgoal, Workspace) tuple for assignment workspace 1
    """
    WO1 = [(1,1),(2,1),(2,5),(1,5)]
    WO2 = [(3,4),(4,4),(4,12),(3,12)]
    WO3 = [(3,12),(12,12),(12,13),(3,13)]
    WO4 = [(12,5),(13,5),(13,13,),(12,13)]
    WO5 = [(6,5),(12,5),(12,6),(6,6)]
    qstart = (0,0)
    qgoal = (10,10)

    obstacles = [WO1, WO2, WO3, WO4, WO5]

    W = np.zeros((15, 15))

    for obs in obstacles:
        xs = sorted(list(set([coord[0] for coord in obs])))
        ys = sorted(list(set([coord[1] for coord in obs])))

        assert (len(xs), len(ys)) == (2,2)

        W[xs[0]:xs[1], ys[0]:ys[1]] = 1

    qstart = np.array(qstart)
    qgoal = np.array(qgoal)

    return qstart, qgoal, W


def workspace2_data(pad=7):
    """
    Retrive (qstart, qgoal, Workspace, extents, pad) tuple for assignment workspace 2
    """
    WO1 = [(-6,-6), (25,-6), (25,-5), (-6,-5)]
    WO2 = [(-6,5),(30,5),(30,6),(-6,6)]
    WO3 = [(-6,-5),(-5,-5),(-5,5),(-6,5)]
    WO4 = [(4,-5),(5,-5),(5,1),(4,1)]
    WO5 = [(9,0),(10,0),(10,5),(9,5)]
    WO6 = [(14,-5),(15,-5),(15,1),(14,1)]
    WO7 = [(19,0),(20,0),(20,5),(19,5)]
    WO8 = [(24,-5),(25,-5),(25,1),(24,1)]
    WO9 = [(29,0),(30,0),(30,5),(29,5)]

    obstacles = [WO1, WO2, WO3, WO4, WO5, WO6, WO7, WO8, WO9]

    qstart = (0,0)
    qgoal = (35,0)

    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')

    for obs in obstacles:
        xs = sorted(list(set([coord[0] for coord in obs])))
        ys = sorted(list(set([coord[1] for coord in obs])))
        if xs[0] < xmin: xmin = xs[0]
        if xs[1] > xmax: xmax = xs[1]
        if ys[0] < ymin: ymin = ys[0]
        if ys[1] > ymax: ymax = ys[1]


    xrange, yrange = (xmax-xmin, ymax-ymin)

    # Transform indices to only be positive with some padding
    def transform(x,y):
        '''Real Position -> Translated Matrix Indices'''
        return (x+(pad-xmin),y+(pad-ymin))

    def itransform(x,y):
        '''Translated Matrix Indices -> Real Position'''
        return (x-(pad-xmin),y-(pad-ymin))


    qstart = np.array(transform(*qstart))
    qgoal = np.array(transform(*qgoal))

    W = np.zeros((xrange+2*pad,yrange+2*pad))

    for obs in obstacles.copy():
        for i,(x,y) in enumerate(obs):
            obs[i] = transform(x,y)
            
        xs = sorted(list(set([coord[0] for coord in obs])))
        ys = sorted(list(set([coord[1] for coord in obs])))
        
        assert (len(xs),len(ys)) == (2,2)
        
        W[xs[0]:xs[1],ys[0]:ys[1]] = 1

    extents = [xmin, xmax, ymin, ymax]

    return qstart, qgoal, W, extents, pad