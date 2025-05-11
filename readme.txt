PRE-GENERATED DATASET:
For using the pre-generated datasets, go to "{space_name}/data" where space_name = Plane, Sphere, Torus, V2.1, Recon_Surface. The Torus and Recon_Surface datasets have 10k and 50k versions, which indicate number of trajectories in dataset. For Table 1 in the paper, the 10k versions were used.


GENERATING CUSTOM DATASET:
For generating Mountain (Smooth), Plane, Sphere, Torus datasets, please run "{space_name}_data_generator.py" where space_name = plane, sphere, torus, surface.

Adjustable parameters: length of sequence, maximum allowable velocity step.


GENERATING ON CUSTOM MANIFOLDS:

For generating dataset on custom manifolds specified via a function z=f(x,y), modify the Mountain (Smooth) directory as follows:

1. In "surface_math.py", specify the function f as a function that can act on pytorch tensors
2.  Run surface_data_generator.py after modifying parameters as necessary

For generating dataset on custom manifolds specified via an embedding with coordinate functions X(u,v), Y(u,v), Z(u,v), modify the Torus directory as follows:

1. In "torus_math.py", specify the immersion function inside the class "Torus" as a function that can act on PyTorch tensors
2.  Run torus_data_generator.py after modifying parameters as necessary

