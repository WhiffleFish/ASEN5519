import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Jacobian with optional end affector angle
def jacobian(z, x, y, theta3=None):
    """
    Determine Jacobian of 3-link system given some end position (x,y) and configuration tuple
    z containing theta1, theta2, and theta3
    """
    if theta3 is None:
        t1,t2,t3 = z
    else:
        t1,t2 = z
        t3 = theta3

    f1 = 8*np.cos(t1) + 8*np.cos(t1 + t2) + 9*np.cos(t1 + t2 + t3) - x
    f2 = 8*np.sin(t1) + 8*np.sin(t1 + t2) + 9*np.sin(t1 + t2 + t3) - y
    f3 = f1**2 + f2**2

    df1dt1 = -8*np.sin(t1) - 8*np.sin(t1 + t2) - 9*np.sin(t1 + t2 + t3)
    df2dt1 = 8*np.cos(t1) + 8*np.cos(t1 + t2) + 9*np.cos(t1 + t2 + t3)
    df3dt1 = 2*f1*df1dt1 + 2*f2*df2dt1

    df1dt2 = -8*np.sin(t1 + t2) - 9*np.sin(t1 + t2 + t3)
    df2dt2 = 8*np.cos(t1 + t2) + 9*np.cos(t1 + t2 + t3)
    df3dt2 = 2*f1*df1dt2 + 2*f2*df2dt2

    df1dt3 = -9*np.sin(t1 + t2 + t3)
    df2dt3 = 9*np.cos(t1 + t2 + t3)
    df3dt3 = 2*f1*df1dt3 + 2*f2*df2dt3
    if theta3 is None:
        return np.array([[df1dt1, df1dt2, df1dt3],
                         [df2dt1, df2dt2, df2dt3],
                         [df3dt1, df3dt2, df3dt3]])
    else:
        return np.array([[df1dt1, df1dt2],
                         [df2dt1, df2dt2]])


# Error function with optional end affector angle
def func(z, x, y, theta3=None):
    """
    Determine error array for given configuration and desired end location
    """
    if theta3 is None:
        t1,t2,t3 = z
    else:
        t1,t2 = z
        t3 = theta3

    f1 = 8*np.cos(t1) + 8*np.cos(t1 + t2) + 9*np.cos(t1 + t2 + t3) - x
    f2 = 8*np.sin(t1) + 8*np.sin(t1 + t2) + 9*np.sin(t1 + t2 + t3) - y
    f3 = f1**2 + f2**2
    if theta3 is None:
        return [f1, f2, f3]
    else:
        return [f1, f2]


def find_angles(x,y,theta3=None, full_output=True):
    """
    Solve numerically for some desired end position (x,y)
    """
    if theta3 is None:
        x0 = np.ones(3)*np.pi/4
    else:
        x0 = np.ones(2)*np.pi/4
    return fsolve(func, x0, args = (x,y,theta3), fprime=jacobian, factor=0.1, full_output=full_output)


def get_all_angles(x, y, N=1000):
    """
    Sweep theta3 from 0 to 360 deg to find all corresponding theta1 and theta2 numerically
    """
    t1 = []
    t2 = []
    t3 = []
    for theta3 in np.linspace(0, 2*np.pi, N):
        angles, info_dict, ier, msg = find_angles(x, y, theta3)
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
    Plot output of get_all_angles
    """
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    ax.scatter(t3,t1,s=3, label=r"$\theta_{1}$ (deg)")
    ax.scatter(t3,t2,s=3, label=r"$\theta_{2}$ (deg)")
    ax.set_xlabel(r"$\theta_{3}$ (deg)",fontsize=12)
    ax.set_ylabel(r"$\theta_{1},\theta_{2}$ (deg)",fontsize=12)
    ax.legend()
    ax.set_title("Configuration angles required to reach point (0,4)")

    return fig, ax


def end_pos(t1, t2, t3):
    x = 8*np.cos(t1) + 8*np.cos(t1 + t2) + 9*np.cos(t1 + t2 + t3)
    y = 8*np.sin(t1) + 8*np.sin(t1 + t2) + 9*np.sin(t1 + t2 + t3)
    return x,y