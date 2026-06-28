# microgpt

A minimal pure Rust GPT training and inference example, translated from Andrej Karpathy's microgpt.py.

The training data is bundled in `data/names.txt` and embedded into the binary at compile time.

## Usage

Show command help:

```sh
cargo run -- --help
```

Train a model for the default 300 steps, save it to `model.weights`, and run inference:

```sh
cargo run --release
```

Train only:

```sh
cargo run --release -- train model.weights
```

Train with a custom number of steps:

```sh
cargo run --release -- train model.weights 100
```

Run inference from saved weights:

```sh
cargo run --release -- infer model.weights
```
