from execute import execute

execute(model_name='LSTM', path='Recon_Surface', hidden_dims=[64], N_trajectories=10000, num_epochs=1000)