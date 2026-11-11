"""Local speech-to-text with speaker labels for long, messy recordings."""

import os

# pyannote-audio 4.x reports usage to otel.pyannote.ai unless told not to: the pipeline's
# name, the library version, a random session id, the length of the audio and the number of
# speakers asked for. No audio and no text, but a tool whose point is that nothing leaves
# the machine does not send that either. Set PYANNOTE_METRICS_ENABLED=true to opt back in.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

__version__ = "0.1.0"
