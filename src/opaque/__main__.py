"""Entry point for ``python -m opaque`` — used by the installed git hook, which cannot rely
on the ``opaque`` console script being on the hook's PATH."""

from .cli import main

main()
