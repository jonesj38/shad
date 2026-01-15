"""CLI module for Shad."""

__all__ = ["cli"]


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as entry point."""
    if name == "cli":
        from shad.cli.main import cli
        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
