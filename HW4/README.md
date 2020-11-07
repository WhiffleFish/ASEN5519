# ASEN 5519 HW4


## Problem 1
Run cells in `P1Demo.ipynb` OR
```
%run P1Utils.py # Load necessary functions
DG = build_P1_graph()
sequence, n_iter, path_length = a_star(DG, heuristic=False)
```


## Problem 2
Run cells in `P2Demo.ipynb` OR
- `%run P2Utils.py`
- load default workspace parameters (A,B1 or B2) with `load_env` taking strings "A","B1", or "B2" as arguments and returning qstart, qgoal, obstacles, x boundaries, and y boundaries.
- Class `PRMSolver` can be instantiated manually by inputting qstart, qgoal, obstacles, x boundaries, and y boundaries or by simply taking the outputs of `load_env` as arguments.
- Sample n points with radius r with `PRMSolver.sample(r,n)`
- Path is smoothed with `PRMSolver.smooth_path(N)`, where N is the number of smoothing iterations (this step is optional)
- The data is visualized with `PRMSolver.plot()`
    - Takes boolean argument `path_only` where, if true, plots only the nodes and edges of the path that are part of the solution path from qstart to qgoal.

Ex.
```
PRM = PRMSolver(*load_env('A'))
PRM.sample(r=1,n=200)
PRM.smooth_path(100)
PRM.plot(path_only=False)
```


## Problem 3
Run cells in `P3Demo.ipynb` OR
- `%run P3Utils.py`
- load default workspace parameters (A,B1 or B2) with `load_env` taking strings "A","B1", or "B2" as arguments and returning qstart, qgoal, obstacles, x boundaries, and y boundaries.
- Class `RRTSolver` can be instantiated manually by inputting qstart, qgoal, obstacles, x boundaries, and y boundaries or by simply taking the outputs of `load_env` as arguments.
    - Additional optional arguments are `pgoal, eps, max_iter,step_size` that respectively control the goal bias probability of sampling qgoal, the goal state tolerance, maximum number of sampling iterations, and distance between connected nodes.
- Sample n points with `RRTSolver.search()`
- The data is visualized with `RRTSolver.plot()`
    - Takes boolean argument `path_only` where, if true, plots only the nodes and edges of the path that are part of the solution path from qstart to qgoal.

Ex.
```
RRT = RRTSolver(*load_env('A'), pgoal=0.005)
RRT.search()
RRT.plot(path_only=True)
```
