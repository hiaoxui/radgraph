from typing import List, Dict

import numpy

from radgraph.allennlp.common.util import JsonDict
from radgraph.allennlp.data import Instance
from radgraph.allennlp.predictors.predictor import Predictor
from radgraph.allennlp.data.fields import LabelField


@Predictor.register("text_classifier")
class TextClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.

    Registered as a `Predictor` with name "text_classifier".
    """

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
