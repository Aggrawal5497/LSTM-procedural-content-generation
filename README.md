# Tensorflow generator-discriminator without TRAIN OP

- <b>Generator</b> : starts sampling process by a special go token, ends the sampling process after flat image pixels are generated.
- <b>Discriminator</b> : Takes the generated samples from generator in shape `[batch_size, image_flat_dim]`, simple single layer neural netword using sigmoid activation.


# LSTM-procedural-content-generation

## Try 1 : Without attention
- Input : (batch_size, latent_dim, latent_dim) where ```latent_dim```, any size for noise matrix

- Output : (batch_size, grid_row, grid_cols) 

### For Training
- Input : random noise = X

- Output : grids from generator = y_

- Labels : grids from training example = y
