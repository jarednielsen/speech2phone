# speech2phone

## Directory Structure

- *experiments/*: Put the results of *mag* experiments as subfolders in here.
- *pca/*: Seong will put his PCA code here, using the preprocessing module and *librosa*.
- *preprocessing/*: Kyle will build this out, and Jared will figure out Python's dumb module system.
- *pretty_images/*: Random plots that are interesting and could be useful in the final report. For example, a PCA .png
- *temp_jaredkyleseong/*: The equivalent of branches. Put work-in-progress here, and bring it out into the main system when it's done.

## Rules

- The directory containing `speech2phone` must be on the environment variable `PYTHONPATH`.
- To append it, run `export PYTHONPATH="${PYTHONPATH}:/my/other/path"`.
- For example, if I have `Users/jarednielsen/Desktop/speech2phone`, then I must have `Users/jarednielsen/Desktop` on my `PYTHONPATH`.
- **Use absolute imports everywhere**. For example, `import speech2phone` or `import speech2phone.preprocessing`.
- See `speech2phone/__init__.py` and `speech2phone/preprocessing/__init__.py` for examples of how to set up subpackages.
- `/preprocessing` applies classic data processing methods (i.e. not learned) to the data, while `/embedding` applies learned methods. For example, Mel spectrogram stuff should be handled in `/preprocessing`.

## boundary recognition

Approaches

- Recurrent network
- Merging (like piecewise linear regression) with the criterion over a metric using dynamic time-warping

## `/embedding`

Options for embedding include:

- spectrum
- cepstrum
- single linear layer (we could try [this](https://ai.stanford.edu/~ang/papers/nips02-metric.pdf) or just SGD)
- more complex learned network
- autoencoder

These will all be specifiable by importing from the embedding module. The spectrum works pretty well as an embedding space, as we found by doing some PCA (see `/pca`). I think we'll use it as a baseline.

## TODO

- create spectrum preprocessor with any customizations you like (Seong)
- phoneme visualization with PCA and Mel spectrogram in `/visualization/pca` (Seong)
- use caching with a pickled dataset file (add to .gitignore) (Kyle)
- add padding from surrounding audio, kwarg default to 0 (Kyle)
- use preprocessing module to do random forests and xgboost (Seong and Jared)
- add pytest and mag (Jared)
- start using semi-supervised learning metric (Jared)
- start playing with learned embeddings in `/embedding` (Kyle)

## Things to try (add ideas here)

- trinemes
- dynamic time-warping
- reapply models to TIMIT to quantify results (quantified semi-supervised learning)
