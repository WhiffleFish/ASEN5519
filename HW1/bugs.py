from utils import *

def BUG1(qstart, qgoal, W):
    """
    INPUT:
        qstart -- (x,y) np.ndarray
        qgoal -- (x,y) np.ndarray
        W -- Workspace np.ndarray
    OUTPUT:
        path -- nx2 np.ndarray
        distance -- float
    """
    pos = qstart.copy()
    move_history = pos.copy().reshape(1,-1)
    while True:
        # Begin MTG Behavior
        while (pos != qgoal).any():
            move = MTG_move(pos, qgoal)
            if isMoveValid(pos, move, W):
                pos += move
                move_history = add_pos(move_history, pos)
            else:
                break

        if (pos == qgoal).all():
            logging.info("Goal Reached")
            break

        # Begin Boundary Following Behavior
        logging.debug(f"BEGIN BOUNDARY FOLLOWING -- POS : {pos}\n")
        BF_history = pos.copy().reshape(1, -1)
        BF_distances = np.array([distance(pos, qgoal)])
        pos += first_contact_move(pos, qgoal, W)
        move_history = add_pos(move_history, pos)
        BF_history = add_pos(BF_history, pos)
        BF_distances = np.append(BF_distances,distance(pos, qgoal))

        while (pos != BF_history[0]).any() and (pos != qgoal).any():
            move = BF_move(pos, BF_history, qgoal, W)

            pos += move
            move_history = add_pos(move_history, pos)
            BF_history = add_pos(BF_history, pos)
            BF_distances = np.append(BF_distances, distance(pos, qgoal))
        logging.debug("END BOUNDARY FOLLOWING\n")

        if (pos==qgoal).all():
            logging.info("Goal Reached")
            break

        logging.debug("BEGIN QL PATH\n")
        index, qL = find_ql(BF_history, BF_distances, hist_loc=True)
        logging.debug(f"QL = {qL}")
        ql_path = shortest_path_to_ql(BF_history, BF_distances)
        move_history = np.vstack([move_history,ql_path])
        pos = move_history.copy()[-1]
        logging.debug(f"END QL PATH -- POS : {pos}\n\n")

    return move_history, get_total_distance(move_history)


def BUG2(qstart, qgoal, W):
    """
    INPUT:
        qstart -- (x,y) np.ndarray
        qgoal -- (x,y) np.ndarray
        W -- Workspace np.ndarray
    OUTPUT:
        path -- nx2 np.ndarray
        distance -- float
    """
    pos = qstart.copy()
    move_history = pos.copy().reshape(1,-1)
    m_line_coords = get_m_coords(qstart.copy(), qgoal)

    while True:
        while (pos != qgoal).any():
            move = MTG_move(pos, qgoal)
            if isMoveValid(pos, move, W):
                pos += move
                move_history = add_pos(move_history, pos)
            else:
                break

        if (pos == qgoal).all():
            logging.info("Goal Reached")
            break

        # Begin Boundary Following Behavior
        BF_history = pos.copy().reshape(1,-1)
        d_h = distance(pos, qgoal) # hit distance
        logging.debug(f"BEGIN BOUNDARY FOLLOWING -- POS : {pos}\n")
        pos += first_contact_move(pos, qgoal, W)
        move_history = add_pos(move_history, pos)
        BF_history = add_pos(BF_history, pos)

        while (pos != qgoal).any():
            if list(pos) in list(map(list, m_line_coords)):
                if distance(pos, qgoal) < d_h:
                    break

            move = BF_move(pos, BF_history, qgoal, W)
            pos += move
            move_history = add_pos(move_history, pos)
            BF_history = add_pos(BF_history, pos)

    return move_history, get_total_distance(move_history)