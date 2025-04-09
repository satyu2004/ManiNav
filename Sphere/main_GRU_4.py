from execute import execute

execute(model_name='GRU', path='Sphere', hidden_dims=[64], N_trajectories=10000, num_epochs=1000)
# execute(model_name='GRU', path='Sphere', hidden_dims=[8], N_trajectories=100, num_epochs=10)