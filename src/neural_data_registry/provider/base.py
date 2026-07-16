from __future__ import annotations
import re, shutil, subprocess
from pathlib import Path
from urllib.parse import urlparse
from neural_data_registry.enums import Provider

class ProviderDownloadError(RuntimeError): pass
def provider_for_url(url: str) -> Provider:
    """Identify the supported provider represented by a dataset URL."""
    host = urlparse(url).netloc.lower()
    if host.endswith("openneuro.org"): return Provider.OPENNEURO
    if host.endswith("dandiarchive.org"): return Provider.DANDI
    if host.endswith("nemar.org"): return Provider.NEMAR
    raise ProviderDownloadError(f"Cannot identify a supported provider from URL: {url}")

def download_from_url(url: str, version: str, destination: Path) -> Provider:
    """Download a provider dataset into a staging destination."""
    provider = provider_for_url(url)
    if provider is not Provider.OPENNEURO: raise ProviderDownloadError(f"Automatic downloads for {provider.value} are not configured yet")
    match = re.search(r"/datasets/(ds\d+)", url)
    if not match: raise ProviderDownloadError("OpenNeuro URLs must contain a dataset identifier such as ds004212")
    git = shutil.which("git")
    if not git: raise ProviderDownloadError("OpenNeuro download requires git")
    command = [git, "clone", f"https://github.com/OpenNeuroDatasets/{match.group(1)}.git", str(destination)]
    if version != "latest": command[2:2] = ["--branch", version]
    try: subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc: raise ProviderDownloadError(exc.stderr.strip() or "OpenNeuro download failed") from exc
    return provider
