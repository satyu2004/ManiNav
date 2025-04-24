from execute import execute

execute(model_name='RNN', path='Sphere', hidden_dims=[32], num_layers=2, N_trajectories=10000, num_epochs=1000)