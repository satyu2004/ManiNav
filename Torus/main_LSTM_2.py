from execute import execute

execute(model_name='LSTM', path='Torus', hidden_dims=[16], N_trajectories=10000, num_epochs=1000)