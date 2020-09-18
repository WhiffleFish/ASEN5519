# Algorithmic Motion Planning (ASEN 5519) HW 1
## Bug 1 / Bug 2 Demonstation
Sample Workspaces are initialized and solved in FinalResultsDemo.ipynb

For use in Jupyter Notebooks, import all functions with the following command: ```%run bugs.py```  
Otherwise, if simply running a python script, use: ```from bugs.py import *```


## Workspace 1
- Load Data : ```qstart, qgoal, W = workspace1_data()```
- Get Bug1 Path Data : ``` path = BUG1(qstart, qgoal, W)```
- Get Bug2 Path Data : ``` path = BUG2(qstart, qgoal, W)```
- Plot a Path : ```fig, ax = plot_path(path, W) ```

## Workspace 2
- Load Data : ```qstart, qgoal, W, extents, pad = workspace2_data()```
- Get Bug1 Path Data : ``` path = BUG1(qstart, qgoal, W)```
- Get Bug2 Path Data : ``` path = BUG2(qstart, qgoal, W)```
- Plot a Path : ```fig, ax = plot_path(path, W, extents=extents, pad=pad) ```

## OR simply run all cells in FinalResultsDemo.ipynb