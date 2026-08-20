"""Domain errors with stable CLI exit codes."""


class TranscriberError(RuntimeError):
    exit_code = 1


class InputAudioError(TranscriberError):
    exit_code = 3


class EnvironmentError(TranscriberError):
    exit_code = 4


class ModelError(TranscriberError):
    exit_code = 4


class TranscriptionError(TranscriberError):
    exit_code = 5


class OutputError(TranscriberError):
    exit_code = 6

