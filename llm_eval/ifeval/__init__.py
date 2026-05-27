"""IFEval instruction-following evaluation.

The instructions*.py modules are vendored from Google Research's
instruction_following_eval (Apache 2.0), lightly adapted to drop the nltk /
immutabledict / absl dependencies. `evaluate` is the project's own glue.
"""

from .evaluator import evaluate

__all__ = ["evaluate"]
