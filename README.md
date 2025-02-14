# PathParam
**PathParam — Path-Parametric Planning & Control**

For the implementation details, please check the [paper](https://arxiv.org/pdf/2410.04664), and/or the [website](https://PathParam.github.io/).
## Quick Start
Follow these steps:

1. Create a python environment with python 3.10. For example, with conda:
    ```bash
    conda create --name pathparam python=3.10
    conda activate pathparam
    pip install -r requirements.txt
    ```

2. Install [acados](https://docs.acados.org/installation/), as well as its [python interface](https://docs.acados.org/python_interface/index.html#). 

3. Update python path to ensure that the package is visible to python.
    ```
    export PATHPARAM_PATH=$path_to_pathparam
    export PYTHONPATH=$PYTHONPATH:$PATHPARAM_PATH
    ```
    Make sure to add this to your `.bashrc` or `.zshrc` files.

## Examples
### Comparison of moving frames
To compare the Frenet-Serret (FSF) and Parallel Transform(PTF) frames, run the following command:
```bash
python examples/moving_frame_comparison.py
```

### Robotic manipulator
In here we conduct the experiments associated with the robotic manipulator example in the paper. 

With this set of examples we aim to answer the question *Why path-parametric?*. We do this by conducting three different experiments:

1. [Tracking vs Following](examples/robotic_manipulator/example1_tracking_vs_following.py): Shows the advantage of an spatial reference over a temporal one. Run it with:
    ```bash
    python examples/robotic_manipulator/example1_tracking_vs_following.py --case f --d
    ```  
    The options are the following ones:
    - `--case`: `f` (following) or `t` (tracking)
    - `--d`: Introduces disturbance that mimics the robot blockage

2. [Velocity Profile](examples/robotic_manipulator/example2_velocity_profile.py): Shows how path-parametric approaches allow to have a desired velocity profile, a very attractive feature for certain applications. Run it with:
    ```bash
    python examples/robotic_manipulator/example2_velocity_profile.py --case 2
    ```  
    where `--case` defines the velocity profile as `1` (constant), `2` (sinusoidal) or `3` (quadratic).

3. [Corridors - Control around a reference](examples/robotic_manipulator/example3_corridor.py): Shows how path-parametric references are a more general way to conduct motion control around a trajectory -- by means of corridors/tunnels/funnels--, accounting for deviations within a predetermined volume, and therefore, achieving behaviors that are considerably more flexible. Run it with:
    ```bash
    python examples/robotic_manipulator/example3_corridor.py
    ```  


### Time Minimization vs Progress Maximization: Race Car
When comparing progress maximization against a standard time minimization approach, we used the well-known [contouring control](https://github.com/alexliniger/MPCC) based formulation.

### Differentiable Parametric Collision-Free Corridor
To replicate these results, see the original [corrgen repository](https://github.com/jonarriza96/corrgen).