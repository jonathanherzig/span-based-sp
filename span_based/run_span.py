import argparse
import datetime
import json
import logging
import os
import random

from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
from torch import cuda


def _get_logger():
    DIR = os.path.dirname(__file__)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(DIR, '../logs/log_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        'train_weak_sup_acc', 'test_spans_f1', 'train_loss', 'best_val_seq_acc',
        'best_val_loss', 'best_val_den_acc', 'test_den_acc', 'best_epoch', 'learning_rate',
        'use_lex', 'domain', 'split', 'is_weak_sup'))
    return logger


def _run_experiment(config_file, serialization_dir, config_override, cuda_ind, learning_rate,
                    use_lexicon):
    config_override["trainer"].update({"optimizer": {"lr": learning_rate}, "cuda_device": cuda_ind})
    config_override["model"]["domain_utils"].update({"is_use_lexicon": use_lexicon})
    train_model_from_file(parameter_filename=config_file,
                          serialization_dir=serialization_dir,
                          overrides=json.dumps(config_override),
                          force=True)


def _run_all(config_file, serialization_dir, scores_dir, config_override, cuda_ind, learning_rate,
             main_domain, split, is_weak_supervision, use_lexicon):
    """Runs an experiment and logs all results."""
    _run_experiment(config_file, serialization_dir, config_override, cuda_ind,
                    learning_rate, use_lexicon)

    score = json.load(open(scores_dir, 'r'))
    training_weak_sup_acc = round(score['training_weak_sup_acc'], 5) if 'training_weak_sup_acc' in score else 'N/A'
    test_spans_f1 = round(score['test_spans_f1'],5) if 'test_spans_f1' in score else 'N/A'
    train_loss = round(score['training_loss'], 5)
    best_epoch = round(score['best_epoch'], 5)
    best_validation_loss = round(score['best_validation_loss'], 5)
    best_validation_den_accuracy = round(score['best_validation_den_acc'], 5)
    best_validation_seq_acc = round(score['best_validation_seq_acc'], 5)
    test_den_accuracy = round(score['test_den_acc'], 5)
    logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(training_weak_sup_acc, test_spans_f1, train_loss, best_validation_seq_acc,
                                                    best_validation_loss, best_validation_den_accuracy, test_den_accuracy,
                                                    best_epoch, learning_rate,
                                                    use_lexicon,
                                                    main_domain, split, is_weak_supervision))


def _parse_args():
    parser = argparse.ArgumentParser(
      description='experiment parser.',
      formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--domain', '-d', default='geo_iid',
                        choices=['geo_iid', 'geo_template', 'geo_len',
                                 'geo_iid_spans', 'geo_template_spans', 'geo_len_spans',
                                 'geo_iid_f1', 'geo_template_f1', 'geo_len_f1',
                                 'clevr_iid', 'clevr_closure',
                                 'clevr_iid_spans', 'clevr_closure_spans',
                                 'clevr_iid_f1', 'clevr_closure_f1',
                                 'scan_iid', 'scan_jump', 'scan_right', 'scan_around_right',
                                 'scan_iid_spans', 'scan_right_spans', 'scan_around_right_spans',
                                 'scan_iid_f1', 'scan_right_f1', 'scan_around_right_f1'])
    parser.add_argument('--learning_rate', '-r', default=0.00001)
    parser.add_argument('--is_weak_supervision', '-w', action='store_true',
                        help='Whether to use weak or strong supervision.')
    parser.add_argument('--use_lexicon', '-l', action='store_true',
                        help='Whether to use the lexicon.')
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()
    domain = args.domain
    main_domain, split = domain.split('_', 1)
    is_weak_supervision = args.is_weak_supervision
    use_lexicon = args.use_lexicon
    learning_rate = args.learning_rate

    serialization_dir = "tmp/output"
    scores_dir = "tmp/output/metrics.json"
    config_file = "span_based/configs/span_based_config.jsonnet"

    num_devices_available = cuda.device_count()
    print('num_devices_available={}'.format(num_devices_available))
    config_override = dict()
    cuda_ind = 0 if num_devices_available > 0 else -1  # train on gpu, if possible

    # Domain specific modifications
    if main_domain == 'geo':
        if split == 'iid':
            config_override["train_data_path"] = "datasets/geo/funql/train.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev.json"
            config_override["test_data_path"] = "datasets/geo/funql/test.json"
        elif split == 'template':
            config_override["train_data_path"] = "datasets/geo/funql/train_template.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_template.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_template.json"
        elif split == 'len':
            config_override["train_data_path"] = "datasets/geo/funql/train_len.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_len.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_len.json"
        elif split == 'iid_spans':
            config_override["train_data_path"] = "datasets/geo/funql/train_spans.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_spans.json"
            config_override["test_data_path"] = "datasets/geo/funql/test.json"
        elif split == 'template_spans':
            config_override["train_data_path"] = "datasets/geo/funql/train_template_spans.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_template_spans.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_template.json"
        elif split == 'len_spans':
            config_override["train_data_path"] = "datasets/geo/funql/train_len_spans.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_len_spans.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_len.json"
        elif split == 'iid_f1':
            config_override["train_data_path"] = "datasets/geo/funql/train.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_spans.json"
        elif split == 'template_f1':
            config_override["train_data_path"] = "datasets/geo/funql/train_template.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_template.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_template_spans.json"
        elif split == 'len_f1':
            config_override["train_data_path"] = "datasets/geo/funql/train_len.json"
            config_override["validation_data_path"] = "datasets/geo/funql/dev_len.json"
            config_override["test_data_path"] = "datasets/geo/funql/test_len_spans.json"
        else:
            print('No {} split!'.format(split))
            raise Exception
        config_override["model"] = {
                                    "is_weak_supervision": is_weak_supervision,
                                    "domain_utils": {"type": "geo_domain_utils"},
                                    "tree_mapper": "geo_tree_mapper",
                                    "denotation_based_metric": {"type": "denotation_accuracy",
                                                                "executor": "geo_executor"},
                                    }
        config_override["dataset_reader"] = {"is_weak_supervision": is_weak_supervision,
                                             "domain_utils": "geo_domain_utils"}
        config_override["trainer"] = {"num_epochs": 250, "patience": 50}
    elif main_domain == 'clevr':
        if split == 'iid':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/test.json"
        elif split == 'closure':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/closure.json"
        elif split == 'iid_spans':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small_spans.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev_spans.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/test.json"
        elif split == 'closure_spans':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small_spans.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev_spans.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/closure.json"
        elif split == 'iid_f1':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/test_spans.json"
        elif split == 'closure_f1':
            config_override["train_data_path"] = "datasets/clevr/dsl/train_small.json"
            config_override["validation_data_path"] = "datasets/clevr/dsl/dev.json"
            config_override["test_data_path"] = "datasets/clevr/dsl/closure_spans.json"
        else:
            raise Exception
        config_override["trainer"] = {"num_epochs": 18, "patience": 6}
        config_override["dataset_reader"] = {"is_weak_supervision": is_weak_supervision,
                                             "domain_utils": "clevr_domain_utils"}
        config_override["model"] = {
                                    "is_weak_supervision": is_weak_supervision,
                                    "domain_utils": {"type": "clevr_domain_utils"},
                                    "tree_mapper": "clevr_tree_mapper",
                                    "denotation_based_metric": {"type": "denotation_accuracy",
                                                                "executor": "clevr_executor"}
                                    }
    elif main_domain == 'scan':
        if split == 'iid':
            config_override["train_data_path"] = "datasets/scan/dsl/train_simple.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_simple.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_simple.json"
        elif split == 'jump':
            config_override["train_data_path"] = "datasets/scan/dsl/train_jump.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_jump.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_jump.json"
        elif split == 'right':
            config_override["train_data_path"] = "datasets/scan/dsl/train_right.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_right.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_right.json"
        elif split == 'around_right':
            config_override["train_data_path"] = "datasets/scan/dsl/train_around_right.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_around_right.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_around_right.json"
        elif split == 'iid_spans':
            config_override["train_data_path"] = "datasets/scan/dsl/train_simple_spans.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_simple_spans.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_simple.json"
        elif split == 'right_spans':
            config_override["train_data_path"] = "datasets/scan/dsl/train_right_spans.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_right_spans.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_right.json"
        elif split == 'around_right_spans':
            config_override["train_data_path"] = "datasets/scan/dsl/train_around_right_spans.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_around_right_spans.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_around_right.json"
        elif split == 'iid_f1':
            config_override["train_data_path"] = "datasets/scan/dsl/train_simple.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_simple.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_simple_spans.json"
        elif split == 'right_f1':
            config_override["train_data_path"] = "datasets/scan/dsl/train_right.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_right.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_right_spans.json"
        elif split == 'around_right_f1':
            config_override["train_data_path"] = "datasets/scan/dsl/train_around_right.json"
            config_override["validation_data_path"] = "datasets/scan/dsl/dev_around_right.json"
            config_override["test_data_path"] = "datasets/scan/dsl/test_around_right_spans.json"
        else:
            raise Exception
        config_override["trainer"] = {"num_epochs": 12, "patience": 2}
        config_override["dataset_reader"] = {"is_weak_supervision": is_weak_supervision,
                                             "domain_utils": "scan_domain_utils"}
        config_override["model"] = {
                                    "is_weak_supervision": is_weak_supervision,
                                    "domain_utils": {"type": "scan_domain_utils"},
                                    "tree_mapper": "scan_tree_mapper",
                                    "denotation_based_metric": {"type": "denotation_accuracy",
                                                                "executor": "scan_executor"}
                                    }
    else:
        raise Exception

    config_override["evaluate_on_test"] = True
    random.seed(0)
    import_submodules('span_based')
    import_submodules('utils')

    logger = _get_logger()

    _run_all(config_file, serialization_dir, scores_dir, config_override, cuda_ind, learning_rate,
             main_domain, split, is_weak_supervision, use_lexicon)
