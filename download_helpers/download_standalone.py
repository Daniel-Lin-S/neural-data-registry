"""Guard explicitly authorised standalone downloads outside registry storage.

Input
-----
A canonical dataset name, its source URL spellings and absolute destination.
The installed ``brainctl`` command must provide read-only registry queries.

Output
------
No registry records are changed. Persistent empty lock files in /tmp reserve
canonical source, name and destination identities using kernel flock locks.
Files remain after exit, but locks are automatically released on process exit.
Read-only registry checks run before locking and again before yielding. This
cannot serialize against registry ingestion, so callers must explicitly opt
into the exception to the normal registry-intake-lock requirement.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

LOCK_DIRECTORY = Path("/tmp")
LOCK_FILE_MODE = 0o444
QUERY_TIMEOUT_SECONDS = 60
NO_MATCH_MESSAGE = "No dataset matched the supplied ID, name, or URL"


def check_registered_identities(
    name: str,
    urls: Sequence[str],
    destination: Path,
) -> None:
    """Reject registered identities using the installed protected CLI.

    Parameters
    ----------
    name : str
        Canonical dataset name to check, including registered aliases.
    urls : sequence of str
        Equivalent source URLs to check without provider network access.
    destination : pathlib.Path
        Absolute destination, checked as a registered path identifier.
    """
    executable = shutil.which("brainctl")
    if executable is None:
        raise RuntimeError("Standalone preflight requires installed brainctl.")
    queries = [("--name", name)]
    queries.extend(("--url", url) for url in dict.fromkeys(urls))
    queries.append((str(destination),))
    environment = dict(os.environ, NO_COLOR="1", COLUMNS="200")
    for query in queries:
        result = subprocess.run(
            [executable, "query", *query,
             "--field", "name", "--field", "storage_path"],
            capture_output=True, text=True, check=False,
            timeout=QUERY_TIMEOUT_SECONDS, env=environment,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            raise FileExistsError(
                f"Download rejected: registered dataset matches {query!r}. "
                f"Existing dataset and storage path: {output}"
            )
        if result.returncode != 2 or NO_MATCH_MESSAGE not in output:
            raise RuntimeError(
                f"Registry check failed for {query!r} "
                f"(exit {result.returncode}): {output}"
            )


@contextmanager
def identity_lock(identity: str) -> Iterator[None]:
    """Reserve one identity across users without stale process locks.

    Parameters
    ----------
    identity : str
        Namespaced canonical source, dataset name or destination identity.

    Yields
    ------
    None
        The caller owns the lock until the context exits or process dies.
    """
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    path = LOCK_DIRECTORY / f"ndr-download-{digest}.lock"
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL,
                             LOCK_FILE_MODE)
    except FileExistsError:
        descriptor = os.open(path, flags)
    else:
        os.fchmod(descriptor, LOCK_FILE_MODE)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"Expected a regular lock file: {path}")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Download already running for {identity!r}; lock: {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def standalone_download_intake(
    name: str,
    source_identity: str,
    urls: Sequence[str],
    destination: Path,
) -> Iterator[None]:
    """Check registry conflicts and hold standalone identity reservations.

    Parameters
    ----------
    name : str
        Canonical dataset name.
    source_identity : str
        Provider-qualified source ID, independent of URL spelling/version.
    urls : sequence of str
        Equivalent source URLs for registry checks.
    destination : pathlib.Path
        Absolute output directory; no dataset contents are read here.

    Yields
    ------
    None
        The download may run under the explicitly authorised lock exception.
    """
    destination = destination.resolve()
    check_registered_identities(name, urls, destination)
    identities = (
        f"name:{name.casefold()}",
        f"source:{source_identity}",
        f"destination:{destination}",
    )
    with ExitStack() as stack:
        for identity in sorted(identities):
            stack.enter_context(identity_lock(identity))
        check_registered_identities(name, urls, destination)
        print("Registry checks passed; standalone identity locks held.",
              flush=True)
        yield
