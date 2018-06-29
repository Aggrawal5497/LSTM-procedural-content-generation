# LSTM-procedural-content-generation

## Try 1 : Without attention
- Input : (batch_size, latent_dim, latent_dim) where ```latent_dim```, any size for noise matrix

- Output : (batch_size, grid_row, grid_cols) 

### For Training
- Input : random noise = X

- Output : grids from generator = y_

- Labels : grids from training example = y
