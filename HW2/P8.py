import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def lin_params(x1,y1,x2,y2):
    """
    Given to points, return associated slope and intercept
    """
    m = (y2-y1)/(x2-x1)
    b = -m*x1+y1
    return m,b


def grid_to_pos(i, j, gmin, gmax, N):
    """
    Convert array location to cartesian coordinates
    """
    dx = (gmax-gmin)/N
    return (gmin + dx*i, gmin + dx*j)


def joint_coords(t1,t2,l1=1,l2=1):
    """
    Return coordinates of both joints given config angles and joint lengths
    """
    x1 = l1*np.cos(t1)
    y1 = l1*np.sin(t1)

    x2 = x1 + l2*np.cos(t1+t2)
    y2 = y1 + l2*np.sin(t1+t2)

    return x1,y1,x2,y2


def check_intersect(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    Check if line given by points (x1,y1),(x2,y2) intersects line given by points (x3,y3),(x4,y4)
    """
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


def get_segments(verts):
    """
    Sort vertices pair-wise to get array of segments
    """
    segments = []
    for i in range(len(verts)):
        if i == len(verts)-1:
            segments.append([verts[i],verts[0]])
        else:
            segments.append([verts[i],verts[i+1]])
        
    return np.array(segments)


def get_obstacles(*verts):
    """
    Arrange vertices into obstacles comprised of line segments
    """
    return [get_segments(v) for v in verts]


def plot_workspace(*verts,xmin=-2,xmax=2,ymin=-2,ymax=2):
    fig, ax = plt.subplots(figsize=(12,7))
    patches = []
    for v in verts:
        polygon = Polygon(v)
        patches.append(polygon)

    p = PatchCollection(patches)
    ax.add_collection(p)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    
    return fig, ax

def to_CSpace(obstacles, N_theta=500):
    """
    Convert output of get_obstacles function to C-space grid representation
    """
    Cspace_grid = np.zeros((N_theta,N_theta))
    for i, t1 in enumerate(np.linspace(0,2*np.pi,N_theta)):
        if round((i*100)/N_theta,1) in [10,20,30,40,50,60,70,80,90,100]:
            print((i*100)/N_theta,'%')
        for j,t2 in enumerate(np.linspace(0,2*np.pi,N_theta)):
            x1,y1,x2,y2 = joint_coords(t1,t2,l1=1,l2=1)
            
            for segments in obstacles:
                
                # Check if element from joint0 -> joint1 collides
                for segment in segments:
                    intersect = False
                    if check_intersect(0,0,x1,y1,*segment[0],*segment[1]):
                        intersect = True
                        Cspace_grid[i,j] = 1
                        break
                    # Check if element from joint1 -> joint2 collides
                    if check_intersect(x1,y1,x2,y2,*segment[0],*segment[1]):
                        Cspace_grid[i,j] = 1
                        break

                if intersect:
                    break

    return Cspace_grid


def plot_CSpace(Cspace_grid):
    fig, ax = plt.subplots(figsize=(12,7))

    ax.imshow(Cspace_grid, origin='lower', extent=[0,360,0,360], cmap="gist_gray_r")
    ax.set_xlabel(r"$\theta_2$ (deg)")
    ax.set_ylabel(r"$\theta_1$ (deg)")

    return fig, ax