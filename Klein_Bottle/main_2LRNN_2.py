from execute import execute

execute(model_name='RNN_multilayer', path='Torus', hidden_dims=[16], num_layers=2, N_trajectories=10000, num_epochs=1000)