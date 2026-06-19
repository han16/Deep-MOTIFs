from __future__ import annotations

"""Deep-MOTIFs: Bayesian-guided positive-unlabeled deep learning for
multi-omics ASD risk-gene prioritization.

Run from the command line with::

    python -m deep_motifs --project-root C:/path/to/Deep-MOTIFs
"""

__all__ = ["main"]


def main():
    """Entry point for ``python -m deep_motifs`` (lazy import keeps
    ``import deep_motifs`` lightweight)."""
    from .cli import main as _main
    return _main()
