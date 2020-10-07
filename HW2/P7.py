import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Jacobian with optional end affector angle
def jacobian(z, x, y, l1, l2, l3, theta3=None):
    """
    Calculate Jacobian for configuration
    INPUTS:
        z -- tuple of angles (theta1, theta2, [theta3]) 
        x -- Desired end affector x position
        y -- Desired end affector y position
        l1 -- Length of arm segment 1
        l2 -- Length of arm segment 2
        l3 -- Length of arm segment 3
    """
    if theta3 is None:
        t1, t2, t3 = z
    else:
        t1, t2 = z
        t3 = theta3

    f1 = l1*np.cos(t1) + l2*np.cos(t1 + t2) + l3*np.cos(t1 + t2 + t3) - x
    f2 = l1*np.sin(t1) + l2*np.sin(t1 + t2) + l3*np.sin(t1 + t2 + t3) - y
    f3 = f1**2 + f2**2

    df1dt1 = -l1*np.sin(t1) - l2*np.sin(t1 + t2) - l3*np.sin(t1 + t2 + t3)
    df2dt1 = l1*np.cos(t1) + l2*np.cos(t1 + t2) + l3*np.cos(t1 + t2 + t3)
    df3dt1 = 2*f1*df1dt1 + 2*f2*df2dt1

    df1dt2 = -l2*np.sin(t1 + t2) - l3*np.sin(t1 + t2 + t3)
    df2dt2 = l2*np.cos(t1 + t2) + l3*np.cos(t1 + t2 + t3)
    df3dt2 = 2*f1*df1dt2 + 2*f2*df2dt2

    df1dt3 = -l3*np.sin(t1 + t2 + t3)
    df2dt3 = l3*np.cos(t1 + t2 + t3)
    df3dt3 = 2*f1*df1dt3 + 2*f2*df2dt3
    if theta3 is None:
        return np.array([[df1dt1, df1dt2, df1dt3],
                         [df2dt1, df2dt2, df2dt3],
                         [df3dt1, df3dt2, df3dt3]])
    else:
        return np.array([[df1dt1, df1dt2],
                         [df2dt1, df2dt2]])


# Error function with optional end affector angle
def func(z, x, y, l1, l2, l3, theta3=None):
    """
    Calculate error for configuration, given a desired endpoint
    INPUTS:
        z -- tuple of angles (theta1, theta2, [theta3]) 
        x -- Desired end affector x position
        y -- Desired end affector y position
        l1 -- Length of arm segment 1
        l2 -- Length of arm segment 2
        l3 -- Length of arm segment 3
    """
    
    if theta3 is None:
        t1, t2, t3 = z
    else:
        t1, t2 = z
        t3 = theta3

    f1 = l1*np.cos(t1) + l2*np.cos(t1 + t2) + l3*np.cos(t1 + t2 + t3) - x
    f2 = l1*np.sin(t1) + l2*np.sin(t1 + t2) + l3*np.sin(t1 + t2 + t3) - y
    f3 = f1**2 + f2**2
    if theta3 is None:
        return [f1, f2, f3]
    else:
        return [f1, f2]


def find_angles(x, y, l1, l2, l3, theta3=None, full_output=False, maxfev=1000):
    """
    Calculate configuration angles for a specified end affector location
    INPUTS:
        x -- Desired end affector x position
        y -- Desired end affector y position
        l1 -- Length of arm segment 1
        l2 -- Length of arm segment 2
        l3 -- Length of arm segment 3
        theta3 (opt) -- specify end affector angle to limit solutions
        full_output -- output debugging info from fsolve
        maxfev -- maximum optimization calls to error fcn (higher number leads to more accurate results)
    """
    if theta3 is None:
        x0 = np.array([np.arctan2(y,x),0,0])
    else:
        x0 = np.array([np.arctan2(y,x),0])
    return fsolve(func, x0, args = (x, y, l1, l2, l3,theta3), 
                 fprime=jacobian, factor=0.1, full_output=full_output,
                 maxfev = maxfev)


def get_all_angles(x, y, l1, l2, l3, N=1000):
    """
    Sweep through all possible end affector angles and determine corresponding angles theta1 
    and theta2 that reach the specified end affector position, if a solution exists.
    INPUT:
        x -- Desired end affector x position
        y -- Desired end affector y position
        l1 -- Length of arm segment 1
        l2 -- Length of arm segment 2
        l3 -- Length of arm segment 3
    OUTPUT:
        t1 -- array of angles theta1(deg) that solve the system
        t2 -- array of angles theta2(deg) that solve the system
        t3 -- array of angles theta3(deg) that solve the system
    """
    t1 = []
    t2 = []
    t3 = []
    for theta3 in np.linspace(0, 2*np.pi, N):
        angles, info_dict, ier, msg = find_angles(x, y, l1, l2, l3,theta3)
        if ier == 1:
            t1.append(angles[0])
            t2.append(angles[1])
            t3.append(theta3)
            
    t1 = np.rad2deg(np.array(t1))
    t2 = np.rad2deg(np.array(t2))
    t3 = np.rad2deg(np.array(t3))

    return t1, t2, t3


def scatter_angles(t1, t2, t3):
    """
    Plot lists of angles (meant to handle output of get_all_angles)
    INPUT:
        t1 -- array of angles theta1(deg)
        t2 -- array of angles theta2(deg)
        t3 -- array of angles theta3(deg)
    OUTPUT:
        fig -- figure that holds the axis
        ax -- axis onto which the plot is drawn
    """
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    ax.scatter(t3,t1,s=3, label=r"$\theta_{1}$ (deg)")
    ax.scatter(t3,t2,s=3, label=r"$\theta_{2}$ (deg)")
    ax.set_xlabel(r"$\theta_{3}$ (deg)",fontsize=12)
    ax.set_ylabel(r"$\theta_{1},\theta_{2}$ (deg)",fontsize=12)
    ax.legend()
    ax.set_title("Configuration angles required to reach point (0,4)")

    return fig, ax


def end_pos(t1, t2, t3, l1, l2, l3):
    """
    INPUT:
        t1 -- theta1
        t2 -- theta2(deg)
        t3 -- theta3(deg)
        l1 -- arm 1 length
        l2 -- arm 2 length
        l3 -- arm 3 length
    OUTPUT:
        resulting position (x,y)
    """
    x = l1*np.cos(t1) + l2*np.cos(t1 + t2) + l3*np.cos(t1 + t2 + t3)
    y = l1*np.sin(t1) + l2*np.sin(t1 + t2) + l3*np.sin(t1 + t2 + t3)
    return x, y

def plot_configuration(t1, t2, t3, l1, l2, l3):
    p0 = [0,0]
    p1 = end_pos(t1,0,0,l1,0,0)
    p2 = end_pos(t1,t2,0,l1,l2,0)
    p3 = end_pos(t1,t2,t3,l1,l2,l3)
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    ax.set_aspect('equal')
    xs = [p0[0],p1[0],p2[0],p3[0]]
    ys = [p0[1],p1[1],p2[1],p3[1]]
    ax.plot(xs,ys, 'r-o', lw=5, ms=10)