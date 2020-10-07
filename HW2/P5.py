import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def angle_between(p1,p2):
    """
    Calculate angle between points p1 and p2 (numpy ndarrays)
    """
    diff = p2-p1
    angle = np.arctan2(*diff[::-1])
    if angle < 0:
        angle += 2*np.pi
    return angle


def find_starting_element_index(arr):
    """
    Determine correct starting element for C-space conversion algorithm
    """
    # y must be minimum of all
    # x must be minimum of set of coords containing minimal y
    arr = arr.copy()
    arr[arr[:, 1] != np.min(arr[:, 1])] = np.array([10**4, 10**4])
    return np.argmin(arr[:,0])


def correct_order(arr):
    """
    Determine order of points for C-space conversion algorithm
    """
    ind = find_starting_element_index(arr)
    return np.roll(arr, -ind, axis=0)


def rotation(theta):
    """
    Rotation matrix
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R


def Cspace_vertices(Vobs, Vrob):
    """
    Vobs : numpy array of obstacle vertices
    Vrob : numpy array of robot vertices
    """

    Vobs = correct_order(Vobs)
    Vrob = correct_order(Vrob)

    vertices = []

    i,j = 0,0
    while i<len(Vobs) or j<len(Vrob):
        vertices.append(Vobs[i%3] + Vrob[j%3])
        if i>=3:
            j += 1
        elif j>=3:
            i += 1
        if angle_between(Vobs[i%3],Vobs[(i+1)%3]) < angle_between(Vrob[j%3],Vrob[(j+1)%3]):
            i += 1
        elif angle_between(Vobs[i%3],Vobs[(i+1)%3]) > angle_between(Vrob[j%3],Vrob[(j+1)%3]):
            j += 1
        else:
            i += 1
            j += 1

    return np.array(vertices)


def plot_Cspace_vertices(verts):
    hull = ConvexHull(verts)
    convex_hull_plot_2d(hull)


def Cspace_vertices_distribution(Vobs, Vrob, n=500):
    
    Vobs = correct_order(Vobs)

    all_verts = []
    for theta in np.linspace(0,2*np.pi,n):
        R = rotation(theta)

        Vr = correct_order(np.matmul(R,Vrob.T).T)

        current_verts = Cspace_vertices(Vobs, Vr)
        all_verts.append(np.array(current_verts))

    return all_verts


def plot_Cspace_dist(Cspace_dist, view_angle=(30,-30), cmap='viridis'):
    fig = plt.figure(figsize=(12,7))
    ax = Axes3D(fig)

    n = len(Cspace_dist)

    rgbs = plt.cm.get_cmap(cmap)(np.linspace(0 ,1 ,n))

    ax.view_init(*view_angle)

    zs = np.rad2deg(np.linspace(0, 2*np.pi, n))
    for i,l in enumerate(Cspace_dist):
        x = list(l[:, 0])
        y = list(l[:, 1])
        z = [zs[i]]*len(x)
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_facecolor(rgbs[i])
        if (i+1)%10 == 0:
            poly.set_edgecolor('black')
        ax.add_collection3d(poly)

    ax.set_xlim3d([-2, 4])
    ax.set_ylim3d([-2, 4])
    ax.set_zlim3d([0, 370])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'$\theta$ (°)')

    return fig, ax
