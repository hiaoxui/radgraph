from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from radgraph.allennlp.common.checks import ConfigurationError
from radgraph.allennlp.common.file_utils import cached_path
from radgraph.allennlp.data.dataset_readers.dataset_reader import DatasetReader
from radgraph.allennlp.data.dataset_readers.dataset_utils import to_bioul
from radgraph.allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from radgraph.allennlp.data.instance import Instance
from radgraph.allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from radgraph.allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("conll2003")
class Conll2003DatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    ```
    WORD POS-TAG CHUNK-TAG NER-TAG
    ```

    with a blank line indicating the end of each sentence
    and `-DOCSTART- -X- -X- O` indicating the end of each article,
    and converts it into a `Dataset` suitable for sequence tagging.

    Each `Instance` contains the words in the `"tokens"` `TextField`.
    The values corresponding to the `tag_label`
    values will get loaded into the `"tags"` `SequenceLabelField`.
    And if you specify any `feature_labels` (you probably shouldn't),
    the corresponding values will get loaded into their own `SequenceLabelField` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent `Instance`. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Registered as a `DatasetReader` with name "conll2003".

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label : `str`, optional (default=`ner`)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels : `Sequence[str]`, optional (default=`()`)
        These labels will be loaded as features into the corresponding instance fields:
        `pos` -> `pos_tags`, `chunk` -> `chunk_tags`, `ner` -> `ner_tags`
        Each will have its own namespace : `pos_tags`, `chunk_tags`, `ner_tags`.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme : `str`, optional (default=`IOB1`)
        Specifies the coding scheme for `ner_labels` and `chunk_labels`.
        Valid options are `IOB1` and `BIOUL`.  The `IOB1` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the chosen `tag_label`.
    """

    _VALID_LABELS = {"ner", "pos", "chunk"}

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tag_label: str = "ner",
        feature_labels: Sequence[str] = (),
        coding_scheme: str = "IOB1",
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB1"

    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ner_tags = fields
                    # TextField requires `Token` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

    def text_to_instance(  # type: ignore
        self,
        tokens: List[Token],
        pos_tags: List[str] = None,
        chunk_tags: List[str] = None,
        ner_tags: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_chunks = (
                to_bioul(chunk_tags, encoding=self._original_coding_scheme)
                if chunk_tags is not None
                else None
            )
            coded_ner = (
                to_bioul(ner_tags, encoding=self._original_coding_scheme)
                if ner_tags is not None
                else None
            )
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if "pos" in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use pos_tags as "
                    "features. Pass them to text_to_instance."
                )
            instance_fields["pos_tags"] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        if "chunk" in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use chunk tags as "
                    "features. Pass them to text_to_instance."
                )
            instance_fields["chunk_tags"] = SequenceLabelField(coded_chunks, sequence, "chunk_tags")
        if "ner" in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use NER tags as "
                    " features. Pass them to text_to_instance."
                )
            instance_fields["ner_tags"] = SequenceLabelField(coded_ner, sequence, "ner_tags")

        # Add "tag label" to instance
        if self.tag_label == "ner" and coded_ner is not None:
            instance_fields["tags"] = SequenceLabelField(coded_ner, sequence, self.label_namespace)
        elif self.tag_label == "pos" and pos_tags is not None:
            instance_fields["tags"] = SequenceLabelField(pos_tags, sequence, self.label_namespace)
        elif self.tag_label == "chunk" and coded_chunks is not None:
            instance_fields["tags"] = SequenceLabelField(
                coded_chunks, sequence, self.label_namespace
            )

        return Instance(instance_fields)
