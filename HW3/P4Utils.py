import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm


class Obstacle(object):
    def __init__(self,verts):
        self.verts = np.array(verts)
        self.segments = self.get_segments(self.verts)
    
    @staticmethod
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



class Cspace(object):
    def __init__(self, obstacles, N_theta=500):
        self.obstacles = obstacles
        self.N_theta = N_theta
        self.c_space = self.to_CSpace(obstacles,N_theta)

    @staticmethod
    def lin_params(x1,y1,x2,y2):
        """
        Given to points, return associated slope and intercept
        """
        m = (y2-y1)/(x2-x1)
        b = -m*x1+y1
        return m,b
    
    @staticmethod
    def grid_to_pos(i, j, gmin, gmax, N):
        """
        Convert array location to cartesian coordinates
        """
        dx = (gmax-gmin)/N
        return (gmin + dx*i, gmin + dx*j)

    @staticmethod
    def joint_coords(t1,t2,l1=1,l2=1):
        """
        Return coordinates of both joints given config angles and joint lengths
        """
        x1 = l1*np.cos(t1)
        y1 = l1*np.sin(t1)

        x2 = x1 + l2*np.cos(t1+t2)
        y2 = y1 + l2*np.sin(t1+t2)

        return x1,y1,x2,y2


    def check_intersect(self,x1,y1,x2,y2,x3,y3,x4,y4):
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
        
        m1,b1 = self.lin_params(x1, y1, x2, y2)
        
        if x3 == x4:
            yk = m1*x3 + b1
            if (min(y3,y4) <= yk <= max(y3,y4)) and (min(y1,y2) <= yk <= max(y1,y2)):
                return True
            else:
                return False
        
        m2,b2 = self.lin_params(x3,y3,x4,y4)

        if m1 == m2:
            return False


        Xa = (b2 - b1) / (m1 - m2)

        if ( (Xa < max( min(x1,x2), min(x3,x4) )) or
            (Xa > min( max(x1,x2), max(x3,x4) )) ):
            return False 
        else:
            return True


    def to_CSpace(self,obstacles, N_theta=500):
        """
        Convert output of get_obstacles function to C-space grid representation
        """
        min_q0_error = np.inf
        min_qg_error = np.inf
        q0_angles = (0,0)
        qg_angles = (0,0)

        Cspace_grid = np.zeros((N_theta,N_theta))
        for i, t1 in tqdm(enumerate(np.linspace(0,2*np.pi,N_theta)), total=N_theta):
            for j,t2 in enumerate(np.linspace(0,2*np.pi,N_theta)):
                x1,y1,x2,y2 = self.joint_coords(t1,t2,l1=1,l2=1)
                
                # e0 = np.linalg.norm(np.array([x2,y2]) - self.q0)
                # eg = np.linalg.norm(np.array([x2,y2]) - self.qgoal)
                # if e0 < min_q0_error:
                #     q0_angles = np.array([t1,t2])
                # if eg < min_qg_error:
                #     qg_angles = np.array([t1,t2])

                for obstacle in obstacles:
                    
                    # Check if element from joint0 -> joint1 collides
                    for segment in obstacle.segments:
                        intersect = False
                        if self.check_intersect(0,0,x1,y1,*segment[0],*segment[1]):
                            intersect = True
                            Cspace_grid[i,j] = 1
                            break
                        # Check if element from joint1 -> joint2 collides
                        if self.check_intersect(x1,y1,x2,y2,*segment[0],*segment[1]):
                            Cspace_grid[i,j] = 1
                            break

                    if intersect:
                        break

        # self.q0_angles = q0_angles
        # self.qg_angles = qg_angles

        return Cspace_grid

    def plot_CSpace(self, figsize=(15,10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.c_space, origin='lower', extent=[0,360,0,360])
        return fig, ax
