"""``python -m whetstone_envs.optim.study``.

The study is documented as a module invocation as well as the
``whetstone-study`` console script, and the two must be the same program:
this delegates to :func:`~whetstone_envs.optim.study.cli.main` rather than
reimplementing any dispatch, so a subcommand cannot exist under one entry
point and not the other.
"""

from __future__ import annotations

from whetstone_envs.optim.study.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
