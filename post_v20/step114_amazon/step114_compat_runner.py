"""Pre-outcome Transformers compatibility shim for locked Step 114 code.

This runner does not alter any Step 114 scientific choice.  It only discards
``token_type_ids`` when an AutoTokenizer supplies that field to a DistilBERT
sequence-classification model whose forward method does not accept it.
"""

from __future__ import annotations

import runpy

from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)


_ORIGINAL_FORWARD = DistilBertForSequenceClassification.forward


def _forward_without_token_type_ids(self, *args, token_type_ids=None, **kwargs):
    del token_type_ids
    return _ORIGINAL_FORWARD(self, *args, **kwargs)


DistilBertForSequenceClassification.forward = _forward_without_token_type_ids
runpy.run_path("step114_pipeline.py", run_name="__main__")
