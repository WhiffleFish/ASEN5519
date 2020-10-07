# ASEN 5519 HW 2

All files intended to be run in Python Jupyter environment

## Problem 3
- Utility functions supplied in P3.py
    - Import utility functions by running %run P3.py
- Get all angles numerically as an inverse kinematic solution for a given x,y coordinate point with get_all_angles
- Plot the found angles with scatter_angles


## Problem 5
- Utility functions supplied in P5.py
    - Import utility functions by running %run P5.py
- Define obstacle vertices (Vobs) with numpy array
- Define robot vertices (Vrob) with numpy array
- Get C-space vertices with function Cspace_vertices
- Plot the C-space vertices with plot_Cspace_vertices

- Get vertex distribution over rotation angles 0-360° with Cspace_vertices_distribution
- Plot the full 3-dimensional distribution with plot_Cspace_dist


## Problem 7
- Utility functions supplied in P7.py
    - Import utility functions by running %run P7.py
- Change lengths of segments, if need be, by changing values of variables l1,l2,l3
- Change angles of joints, if need be, by changing values of variables t1,t2,t3
- Run cell to plot updated values


## Problem 9
- Utility functions supplied in P8.py
    - Import utility functions by running %run P8.py
- Define an array of vertices for an obstacle with a numpy array
- Plot the workspace with plot_workspace, which takes an arbitrary number of vertex arrays as an argument
- Define vertices explicitly as obstacles with get_obstacles taking the vertices as arguments
- Get C-space grid with to_Cspace, taking the output of get_obstacles as an input
- Plot the C-space grid with plot_CSpace
