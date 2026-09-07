"""Fixture: shapes the no-raise-in-result-service rule must spare."""


class _Crawler:
    def __init__(self) -> None:
        self._crawler: object | None = None

    @property
    def crawler(self) -> object:
        if self._crawler is None:
            raise RuntimeError("CrawlerService requires an injected crawler")
        return self._crawler


async def relay() -> None:
    try:
        await _work()
    except Exception:
        raise


async def _work() -> None:
    return None
