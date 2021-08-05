/** You could basically use this config to train your own BERT classifier,
    with the following changes:

    1. change the `bert_model` variable to "bert-base-uncased" (or whichever you prefer)
    2. swap in your own DatasetReader. It should generate instances
       that look like {"tokens": TextField(...), "label": LabelField(...)}.

       You don't need to add the "[CLS]" or "[SEP]" tokens to your instances,
       that's handled automatically by the token indexer.
    3. replace train_data_path and validation_data_path with real paths
    4. any other parameters you want to change (e.g. dropout)
 */


# For a real model you'd want to use "bert-base-uncased" or similar.
local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "lazy": false,
        "type": "span_based_reader",
        "tokenizer": {
            "type": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        },
    },
    "train_data_path": "/path/to/training/data",
    "validation_data_path": "/path/to/validation/data",
    "model": {
        "span_extractor": {
        "type": "endpoint",
        "input_dim": 768
        },
        "feedforward": {
        "input_dim": 1536,
        "num_layers": 1,
        "hidden_dims": 250,
        "activations": "relu",
        "dropout": 0.1
        },
        "type": "span_based_sp",
        "bert_model": bert_model,
        "dropout": 0.0,
        "token_based_metric": "token_sequence_accuracy",
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 5
    },
    "trainer": {
        "type": "weak_sup_trainer",
        "optimizer": {
            "type": "adam",
            "lr": 0.00001
        },
        "validation_metric": "+den_acc",
        "checkpointer": {
            "num_serialized_models_to_keep": 0
        },
        "num_epochs": 6,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
