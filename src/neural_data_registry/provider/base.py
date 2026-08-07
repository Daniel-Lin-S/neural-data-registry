from __future__ import annotations
from dataclasses import asdict, dataclass
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse
from neural_data_registry.enums import Provider


CONNECTIVITY_TIMEOUT_SECONDS = 30
HEAD_FALLBACK_STATUS_CODES = frozenset({405, 501})
HTTP_SCHEMES = frozenset({"http", "https"})


class ProviderDownloadError(RuntimeError):
    """Report a provider-specific download or connectivity failure."""


@dataclass(frozen=True)
class DownloadConnectivityReport:
    """Describe a successful HTTP connectivity probe for a download source."""

    provider: str
    source_url: str
    final_url: str
    status_code: int
    probe_method: str
    total_size_bytes: None = None
    total_size_status: str = (
        "unavailable: the Git/DataLad source does not expose a trustworthy "
        "total annexed-dataset size before cloning"
    )

    def as_dict(self) -> dict[str, object]:
        """Return the report in a JSON-serializable form."""
        return asdict(self)


def _format_command_error(exc: subprocess.CalledProcessError) -> str:
    """Render captured DataLad output without losing the underlying failure."""
    command = " ".join(str(item) for item in exc.cmd)
    sections = [f"DataLad command failed with exit status {exc.returncode}: {command}"]
    if exc.stdout:
        sections.extend(["stdout:", exc.stdout.strip()])
    if exc.stderr:
        sections.extend(["stderr:", exc.stderr.strip()])
    return "\n".join(sections)

def _is_resumable_workspace(destination: Path) -> bool:
    """Return whether a prior clone left a Git workspace DataLad can resume."""
    return (destination / ".git").exists()
def _find_command(name: str) -> str | None:
    """Find a command on PATH or beside the active Python interpreter."""
    command = shutil.which(name)
    if command:
        return command
    sibling = Path(sys.executable).with_name(name)
    return str(sibling) if sibling.is_file() and os.access(sibling, os.X_OK) else None



def provider_for_url(url: str) -> Provider:
    """Identify a provider from a dataset URL, defaulting to ``other``."""
    host = (urlparse(url).hostname or "").lower()
    providers = {
        "openneuro.org": Provider.OPENNEURO,
        "dandiarchive.org": Provider.DANDI,
        "nemar.org": Provider.NEMAR,
        "physionet.org": Provider.PHYSIONET,
        "kaggle.com": Provider.KAGGLE,
        "neurovault.org": Provider.NEUROVAULT,
        "synapse.org": Provider.SYNAPSE,
    }
    for domain, provider in providers.items():
        if host == domain or host.endswith("." + domain):
            return provider
    return Provider.OTHER

def _mirror_source(mirror: str, dataset_id: str) -> str:
    """Resolve a mirror URL or URL template for one OpenNeuro dataset."""
    if "{dataset_id}" in mirror:
        return mirror.format(dataset_id=dataset_id)
    if mirror.rstrip("/").endswith(".git"):
        return mirror
    return mirror.rstrip(chr(47)) + chr(47) + dataset_id + ".git"


def _openneuro_source(url: str, mirror: str | None) -> str:
    """Resolve the Git source used for an OpenNeuro dataset download."""
    match = re.search(r"/datasets/(ds[0-9]+)", url)
    if not match:
        raise ProviderDownloadError(
            "OpenNeuro URLs must contain a dataset identifier such as ds004212"
        )
    if mirror:
        return _mirror_source(mirror, match.group(1))
    return f"https://github.com/OpenNeuroDatasets/{match.group(1)}.git"


def _http_opener(proxy: str | None) -> request.OpenerDirector:
    """Build an HTTP opener using an explicit or inherited proxy setting."""
    if proxy:
        handler = request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        handler = request.ProxyHandler()
    return request.build_opener(handler)


def _require_http_source(source: str) -> None:
    """Validate that a source can be checked with HTTP(S)."""
    scheme = urlparse(source).scheme.lower()
    if scheme not in HTTP_SCHEMES:
        raise ProviderDownloadError(
            "Connectivity checks require an HTTP or HTTPS mirror URL; "
            f"resolved source is {source}"
        )


def _download_environment(proxy: str | None, command_dir: Path) -> dict[str, str]:
    """Return a subprocess environment with proxy settings and tool PATH."""
    environment = os.environ.copy()
    inherited_path = environment.get("PATH", "")
    tool_path = str(command_dir)
    environment["PATH"] = tool_path + (os.pathsep + inherited_path if inherited_path else "")
    if proxy:
        for variable in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            environment[variable] = proxy
    return environment


def check_download_connectivity(
    url: str,
    *,
    proxy: str | None = None,
    mirror: str | None = None,
) -> DownloadConnectivityReport:
    """Probe the HTTP(S) source used by a supported provider download.

    Parameters
    ----------
    url : str
        Provider dataset URL.
    proxy : str or None, optional
        Explicit HTTP(S) proxy. Standard proxy environment variables are used
        when omitted.
    mirror : str or None, optional
        Mirror URL, URL base, or template containing ``{dataset_id}``.

    Returns
    -------
    DownloadConnectivityReport
        Successful probe details. The total dataset size is unavailable before
        the DataLad clone because annexed payload metadata is not yet present.
    """
    provider = provider_for_url(url)
    if provider is not Provider.OPENNEURO:
        raise ProviderDownloadError(
            f"Automatic downloads for {provider.value} are not configured yet"
        )
    source = _openneuro_source(url, mirror)
    _require_http_source(source)
    opener = _http_opener(proxy)
    probe = request.Request(source, method="HEAD")
    probe_method = "HEAD"
    try:
        response = opener.open(probe, timeout=CONNECTIVITY_TIMEOUT_SECONDS)
    except error.HTTPError as exc:
        if exc.code not in HEAD_FALLBACK_STATUS_CODES:
            raise ProviderDownloadError(
                f"HTTP connectivity check to {source} failed with status "
                f"{exc.code}: {exc.reason}"
            ) from exc
        probe = request.Request(source, headers={"Range": "bytes=0-0"})
        probe_method = "GET range"
        try:
            response = opener.open(probe, timeout=CONNECTIVITY_TIMEOUT_SECONDS)
        except error.HTTPError as fallback_exc:
            raise ProviderDownloadError(
                f"HTTP connectivity check to {source} failed with status "
                f"{fallback_exc.code}: {fallback_exc.reason}"
            ) from fallback_exc
        except error.URLError as fallback_exc:
            raise ProviderDownloadError(
                f"Could not connect to {source}: {fallback_exc.reason}"
            ) from fallback_exc
    except error.URLError as exc:
        raise ProviderDownloadError(
            f"Could not connect to {source}: {exc.reason}"
        ) from exc
    with response:
        status_code = response.getcode()
        final_url = response.geturl()
    return DownloadConnectivityReport(
        provider=provider.value,
        source_url=source,
        final_url=final_url,
        status_code=status_code,
        probe_method=probe_method,
    )


def download_from_url(url: str, version: str, destination: Path, *, proxy: str | None = None, mirror: str | None = None) -> Provider:
    """Download or resume a provider dataset with DataLad in destination."""
    provider = provider_for_url(url)
    if provider is not Provider.OPENNEURO:
        raise ProviderDownloadError(f"Automatic downloads for {provider.value} are not configured yet")
    source = _openneuro_source(url, mirror)
    datalad = _find_command("datalad")
    if not datalad:
        raise ProviderDownloadError("OpenNeuro download requires DataLad; install the package with neural-data-registry[download]")
    if not _find_command("git-annex"):
        raise ProviderDownloadError(
            "OpenNeuro download requires git-annex >= 10.20230126; "
            "install the package with neural-data-registry[download]"
        )
    environment = _download_environment(proxy, Path(datalad).parent)
    try:
        if destination.exists() and _is_resumable_workspace(destination):
            get_command = [datalad, "get", "--recursive", "."]
        else:
            if destination.exists() and any(destination.iterdir()):
                raise ProviderDownloadError(f"Incoming workspace is not a resumable DataLad dataset: {destination}")
            command = [datalad, "clone", source, str(destination)]
            command.extend(["--branch", version])
            subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
            get_command = [datalad, "get", "--recursive", "."]
        subprocess.run(
            get_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=destination,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        raise ProviderDownloadError(_format_command_error(exc)) from exc
    return provider
