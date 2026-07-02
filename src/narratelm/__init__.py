"""narratelm — narrate epub books into chapterized audiobooks locally with VibeVoice."""

import os

# We synthesize one window at a time and encode parts by forking ffmpeg, which
# trips the HF tokenizers "process forked after parallelism" warning on every
# window. Disabling tokenizer parallelism silences it with no real speed cost
# for our single-call-at-a-time usage. `setdefault` respects an explicit override.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
