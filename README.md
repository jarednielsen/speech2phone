# speech2phone

## boundary recognition

Approaches

- Recurrent network
- Merging (like piecewise linear regression) with the criterion over a metric using dynamic time-warping

## /embedding

Options for embedding include:

- spectrum
- cepstrum
- single linear layer (we could try [this](https://ai.stanford.edu/~ang/papers/nips02-metric.pdf) or just SGD)
- more complex learned network
- autoencoder

These will all be specifiable by importing from the embedding module. The spectrum works pretty well as an embedding space, as we found by doing some PCA. I think we'll use it as a baseline.

## Things to try

- trinemes
- dynamic time-warping
