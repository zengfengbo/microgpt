# microgpt

A minimal pure Rust GPT training and inference example, translated from Andrej Karpathy's microgpt.py.

The training data is bundled in `data/names.txt` and embedded into the binary at compile time.

## Usage

Train a model, save it to `model.weights`, and run inference:

```sh
cargo run --release
```

Train only:

```sh
cargo run --release -- train model.weights
```

Run inference from saved weights:

```sh
cargo run --release -- infer model.weights
```
