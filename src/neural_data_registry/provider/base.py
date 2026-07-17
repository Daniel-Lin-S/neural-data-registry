from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from neural_data_registry.enums import Provider

class ProviderDownloadError(RuntimeError): pass
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

def download_from_url(url: str, version: str, destination: Path, *, proxy: str | None = None, mirror: str | None = None) -> Provider:
    """Download or resume a provider dataset with DataLad in destination."""
    provider = provider_for_url(url)
    if provider is not Provider.OPENNEURO:
        raise ProviderDownloadError(f"Automatic downloads for {provider.value} are not configured yet")
    match = re.search(r"/datasets/(ds[0-9]+)", url)
    if not match:
        raise ProviderDownloadError("OpenNeuro URLs must contain a dataset identifier such as ds004212")
    datalad = _find_command("datalad")
    if not datalad:
        raise ProviderDownloadError("OpenNeuro download requires DataLad; install the package with neural-data-registry[download]")
    if not _find_command("git-annex"):
        raise ProviderDownloadError(
            "OpenNeuro download requires git-annex >= 10.20230126; "
            "install the package with neural-data-registry[download]"
        )
    source = _mirror_source(mirror, match.group(1)) if mirror else f"https://github.com/OpenNeuroDatasets/{match.group(1)}.git"
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
