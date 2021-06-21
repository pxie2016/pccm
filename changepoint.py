class Changepoint:
    """
    A class that represents a changepoint. Could be either a
    fixed-value changepoint or a "flexible" changepoint, which
    depends on other variables.
    """

    def __init__(self, value: list[float]) -> None:
        self.value = value

    @classmethod
    def fixed(cls, value: float, sample_size: int) -> "Changepoint":
        return cls([value] * sample_size)

    @classmethod
    def flex(cls, value: list[float]) -> "Changepoint":
        return cls(value)