class AnalyzerError(Exception):
    exit_code: int = 2


class InputError(AnalyzerError):
    """The input could not be read as a chrome trace."""

    exit_code = 1


class AnalysisError(AnalyzerError):
    """The input was read but cannot be analyzed."""

    exit_code = 2
