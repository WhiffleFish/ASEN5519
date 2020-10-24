# ASEN 5519 HW3

## Exercise 2
- Run all cells in ```P2Demo.ipynb```  
OR
- Import requisite utility functions: ```%run P2Utils.py```
- Define start position and goal position as coordinate tuples
- Define obstacles by instantiating obstacle objects for each
    - Obstacle object defined by coorinate point and key-word argument ```c``` which is a boolean determining whether or not the inputted coordinate is the center point of the obstacle. If ```c==False```, the coordinate is taken as the lower-left corner of the obstacle.
    - i.e ```obs1 = Obstacle((4,1),c=True)``` defines an square obstacle centered at point (4,1)
- Group all obstacle objects in a list
- Instantiate ```RectGradDescent``` object with start position tuple, goal position tuple, and the list of obstacles i.e. ```RGD = RectGradDescent(q0, qgoal, obstacles)```
    - Each obstacle has optional ```Q_star,eta``` args

- Plot vectorfield with ```RGD.plot_v_field()```
- Plot path with ```RGD.plot_path()```


## Exercise 3
- Run all cells in ```P3Demo.ipynb```  
OR
- Define start position and goal position as coordinate tuples
- Define obstacles by instantiating obstacle objects for each
    - Obstacle object defined by coorinate point and key-word argument ```c``` which is a boolean determining whether or not the inputted coordinate is the center point of the obstacle. If ```c==False```, the coordinate is taken as the lower-left corner of the obstacle.
    - Remaining args are width and height.
    - i.e ```obs1 = Obstacle((4,1),2,4)``` defines an obstacle with lower-left corner at (4,1) with width 2 and height 4.
- Group all obstacle objects in a list
- Instantiate ```WaveFront``` object:
    - ```WF = WaveFront(q0, qgoal, obstacles)```
- Plot path with ```WF.plot_path()```

## Exercise 4
- Run all cells in ```P4Demo.ipynb```  
OR
- Define vertices of obstacles as numpy arrays
- Instantiate obstacle objects with single argument being the array of vertices
- Group all obstacle objects in a list
- Instantiate Cspace object ```CS = Cspace(obstacles)```
- Plot Cspace with ```CS.plot_CSpace()```
- Instantiate WaveFront Object and make CSpace grid amenable to wave-front grid with the following:
```
xs = np.linspace(0,2*np.pi, CS.c_space.shape[0])
ys = np.linspace(0,2*np.pi, CS.c_space.shape[1])
qgoal_ang = (np.linspace(0,2*np.pi,N_theta)[np.argmin((np.linspace(0,2*np.pi,N_theta)-np.pi)**2)],0)
WF = WaveFront((0,0),qgoal_ang[::-1],obs_grid =CS.c_space,x=xs,y=ys)
```
- Plot wave-front path with ```WF.plot_path()```
- Plot arm path with ```WF.plot_arm_path(CS) ```