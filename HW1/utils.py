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
    index, qL = find_ql(BF_history, BF_distances, hist_loc=True)

    # Get rid of repeated qH
    hist = BF_history.copy()[:-1]

    d1 = index
    d2 = len(hist) - index

    if d1 <= d2:
        path = hist[1:index+1]
    else:
        path = hist[index:][::-1]
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


def plot_path(path, W, cmap='viridis', extents=None, pad=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    rgbs = plt.cm.get_cmap(cmap)(np.linspace(0, 1, path.shape[0]))
    if extents:
        xmin = extents[0]
        ymin = extents[2]
        extents = np.array(extents) + np.array([-pad-0.5, pad-0.5, -pad-0.5, pad-0.5])
        ax.imshow(W.T, cmap='gray_r', origin='lower', extent=list(extents))
        ax.plot(np.array(path)[:, 0]+(xmin-pad), np.array(path)[:, 1]+(ymin-pad))
        ax.scatter(np.array(path)[:, 0]+(xmin-pad), np.array(path)[:, 1]+(ymin-pad), c=rgbs)    
    else:
        ax.imshow(W.T,cmap='gray_r', origin='lower')
        ax.plot(np.array(path)[:, 0], np.array(path)[:, 1])
        ax.scatter(np.array(path)[:, 0], np.array(path)[:, 1], c=rgbs)
    return fig, ax
