"""
The `predict` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""

from typing import List, Iterator, Optional
import argparse
import sys
import json

from radgraph.allennlp.commands.subcommand import Subcommand
from radgraph.allennlp.common import logging as common_logging
from radgraph.allennlp.common.checks import check_for_gpu, ConfigurationError
from radgraph.allennlp.common.file_utils import cached_path
from radgraph.allennlp.common.util import lazy_groups_of
from radgraph.allennlp.models.archival import load_archive
from radgraph.allennlp.predictors.predictor import Predictor, JsonDict
from radgraph.allennlp.data import Instance


@Subcommand.register("predict")
class Predict(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Run the specified model against a JSON-lines input file."""
        subparser = parser.add_parser(
            self.name, description=description, help="Use a trained model to make predictions."
        )

        subparser.add_argument(
            "archive_file", type=str, help="the archived model to make predictions with"
        )
        subparser.add_argument("input_file", type=str, help="path to or url of the input file")

        subparser.add_argument("--output-file", type=str, help="path to output file")
        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument(
            "--batch-size", type=int, default=1, help="The batch size to use for processing"
        )

        subparser.add_argument(
            "--silent", action="store_true", help="do not print output to stdout"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--use-dataset-reader",
            action="store_true",
            help="Whether to use the dataset reader of the original model to load Instances. "
                 "The validation dataset reader will be used if it exists, otherwise it will "
                 "fall back to the train dataset reader. This behavior can be overridden "
                 "with the --dataset-reader-choice flag.",
        )

        subparser.add_argument(
            "--dataset-reader-choice",
            type=str,
            choices=["train", "validation"],
            default="validation",
            help="Indicates which model dataset reader to use if the --use-dataset-reader "
                 "flag is set.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--predictor", type=str, help="optionally specify a specific predictor to use"
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=_predict)
        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )

    return Predictor.from_archive(
        archive, args.predictor, dataset_reader_to_load=args.dataset_reader_choice
    )


class _PredictManager:
    def __init__(
            self,
            predictor: Predictor,
            input_file: str,
            output_file: Optional[str],
            batch_size: int,
            print_to_console: bool,
            has_dataset_reader: bool,
    ) -> None:

        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, "w")
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader
        else:
            self._dataset_reader = None

    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)

    def run(self) -> List:
        index = 0
        ret = []
        for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
            for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                ret.append(result)
                index = index + 1
        return ret


def _predict(args: argparse.Namespace) -> None:
    from radgraph.allennlp.common.plugins import import_plugins
    from radgraph.allennlp.common.util import import_module_and_submodules
    import_plugins()
    import_module_and_submodules("dygie")

    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(
        predictor,
        args.input_file,
        args.output_file,
        args.batch_size,
        not args.silent,
        args.use_dataset_reader,
    )
    manager.run()
