from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from sqlalchemy import select

from neural_data_registry.config import Settings, get_settings
from neural_data_registry.db.models import Dataset, HealthCheckHistory
from neural_data_registry.db.session import create_database, get_session_factory
from neural_data_registry.enums import DatasetStatus
from neural_data_registry.provider import base as provider_base
from neural_data_registry.storage import directory_size, ensure_layout, process_lock


@dataclass(frozen=True)
class HealthCheckReport:
    """Immediate result returned to a query while deep checks run separately."""

    dataset_id: str
    status: DatasetStatus
    history_id: str
    warning: str | None = None
    repair_in_progress: bool = False


def _now() -> datetime:
    return datetime.now().astimezone()


def _has_visible_payload(path: Path) -> bool:
    """Return whether a directory contains a non-hidden file or symlink."""
    for _, directories, files in os.walk(path):
        directories[:] = [name for name in directories if not name.startswith(".")]
        if any(not name.startswith(".") for name in files):
            return True
    return False


def _is_datalad_dataset(path: Path) -> bool:
    """Detect a DataLad/git-annex checkout from repository markers."""
    git_marker = path / ".git"
    return git_marker.exists() and (
        (path / ".datalad").exists()
        or (git_marker.is_dir() and (git_marker / "annex").exists())
    )


def _write_manifest(item: Dataset, config: Settings) -> None:
    from neural_data_registry.service import dataset_dict

    path = config.registry_dir / f"{item.id}.json"
    path.write_text(json.dumps(dataset_dict(item), indent=2), encoding="utf-8")


def _log_problem(
    item: Dataset, history: HealthCheckHistory, message: str, config: Settings
) -> None:
    ensure_layout(config)
    record = {
        "timestamp": _now().isoformat(timespec="seconds"),
        "dataset_id": item.id,
        "name": item.name,
        "storage_path": item.storage_path,
        "health_check_id": history.id,
        "result": history.result,
        "message": message,
    }
    with (config.logs_dir / "critical_errors.log").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _new_history(item: Dataset, db) -> HealthCheckHistory:
    history = HealthCheckHistory(
        dataset_id=item.id,
        previous_status=item.status,
        resulting_status=item.status,
        result="running",
    )
    db.add(history)
    db.flush()
    return history


def _finish_history(
    db,
    item: Dataset,
    history: HealthCheckHistory,
    *,
    result: str,
    message: str,
    config: Settings,
    status: DatasetStatus | None = None,
    repair_attempted: bool = False,
    repair_succeeded: bool | None = None,
    log_problem: bool = False,
) -> None:
    if status is not None:
        item.status = status
    history.result = result
    history.resulting_status = item.status
    history.message = message
    history.repair_attempted = repair_attempted
    history.repair_succeeded = repair_succeeded
    history.completed_at = _now()
    db.commit()
    _write_manifest(item, config)
    if log_problem:
        _log_problem(item, history, message, config)


def _missing_reason(item: Dataset, path: Path | None) -> str | None:
    if path is None:
        return "Dataset has no registered storage path."
    if not path.is_dir():
        return f"Dataset path does not exist or is not a directory: {path}"
    if not _has_visible_payload(path):
        return f"Dataset directory has no non-hidden payload files: {path}"
    return None


def _finish_healthy_with_size(
    db,
    item: Dataset,
    history: HealthCheckHistory,
    path: Path,
    *,
    message: str,
    status: DatasetStatus,
    config: Settings,
    repair_attempted: bool = False,
    repair_succeeded: bool | None = None,
) -> HealthCheckReport:
    """Persist a verified dataset's logical size before marking it healthy."""
    try:
        item.size_bytes = directory_size(path)
    except OSError as exc:
        error = f"Dataset size calculation failed: {exc}"
        _finish_history(
            db,
            item,
            history,
            result="error",
            message=error,
            config=config,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            log_problem=True,
        )
        return HealthCheckReport(item.id, item.status, history.id, warning=error)

    _finish_history(
        db,
        item,
        history,
        result="healthy",
        message=message,
        status=status,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        config=config,
    )
    return HealthCheckReport(item.id, item.status, history.id)


def _load_history(db, item: Dataset, history_id: str | None) -> HealthCheckHistory:
    if history_id is None:
        return _new_history(item, db)
    history = db.get(HealthCheckHistory, history_id)
    if history is None or history.dataset_id != item.id:
        raise ValueError(f"Health-check history does not match dataset {item.id}")
    return history


def _quick_check(
    dataset_id: str,
    config: Settings,
    *,
    history_id: str | None = None,
    refresh_size: bool = False,
) -> tuple[HealthCheckReport, bool]:
    """Run path checks and return whether DataLad verification remains."""
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        item = db.get(Dataset, dataset_id)
        if item is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        history = _load_history(db, item, history_id)
        path = Path(item.storage_path).expanduser() if item.storage_path else None
        missing = _missing_reason(item, path)
        if missing:
            _finish_history(
                db,
                item,
                history,
                result="missing",
                message=missing,
                status=DatasetStatus.MISSING,
                config=config,
                log_problem=True,
            )
            return (
                HealthCheckReport(
                    item.id,
                    item.status,
                    history.id,
                    warning=missing,
                ),
                False,
            )

        assert path is not None
        if not _is_datalad_dataset(path):
            status = (
                DatasetStatus.AVAILABLE
                if item.status in (DatasetStatus.MISSING, DatasetStatus.BROKEN)
                else item.status
            )
            if refresh_size:
                return (
                    _finish_healthy_with_size(
                        db,
                        item,
                        history,
                        path,
                        message="Dataset path contains payload files.",
                        status=status,
                        config=config,
                    ),
                    False,
                )
            _finish_history(
                db,
                item,
                history,
                result="healthy",
                message="Dataset path contains payload files.",
                status=status,
                config=config,
            )
            return HealthCheckReport(item.id, item.status, history.id), False

        db.commit()
        repair_pending = item.status in (
            DatasetStatus.MISSING,
            DatasetStatus.BROKEN,
        )
        warning = None
        if repair_pending:
            warning = (
                "Dataset is unavailable; DataLad verification and repair are in "
                "process. Please try again later."
            )
        return (
            HealthCheckReport(
                item.id,
                item.status,
                history.id,
                warning=warning,
                repair_in_progress=repair_pending,
            ),
            True,
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _command_text(result: subprocess.CompletedProcess[str]) -> str:
    details = (result.stderr or result.stdout or "").strip()
    return details[-4000:] if details else f"exit status {result.returncode}"


def _run_command(
    command: list[str],
    path: Path,
    config: Settings,
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=config.health_command_timeout_seconds,
        check=False,
    )


class AnnexInspectionError(RuntimeError):
    """Raised when annex availability cannot be determined."""


def _missing_annex_files(
    path: Path,
    config: Settings,
    *,
    annex: str,
    environment: dict[str, str],
) -> list[str]:
    """Return files that git-annex successfully reports as absent locally."""
    result = _run_command(
        [annex, "find", "--not", "--in=here"],
        path,
        config,
        environment=environment,
    )
    if result.returncode:
        raise AnnexInspectionError(
            f"git-annex availability check failed: {_command_text(result)}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _complete_datalad_check(
    dataset_id: str,
    history_id: str,
    config: Settings,
) -> HealthCheckReport:
    """Use annex missing-file output as the sole BROKEN classification."""
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        item = db.get(Dataset, dataset_id)
        history = db.get(HealthCheckHistory, history_id)
        if item is None or history is None:
            raise ValueError(f"Dataset or health history no longer exists: {dataset_id}")
        path = Path(item.storage_path).expanduser() if item.storage_path else None
        missing_path = _missing_reason(item, path)
        if missing_path:
            _finish_history(
                db,
                item,
                history,
                result="missing",
                message=missing_path,
                status=DatasetStatus.MISSING,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(
                item.id, item.status, history.id, warning=missing_path
            )

        assert path is not None
        if not _is_datalad_dataset(path):
            message = "DataLad repository markers disappeared during verification."
            _finish_history(
                db,
                item,
                history,
                result="error",
                message=message,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(
                item.id, item.status, history.id, warning=message
            )

        git_annex = provider_base._find_command("git-annex")
        if not git_annex:
            message = "Health check could not run because git-annex is missing."
            _finish_history(
                db,
                item,
                history,
                result="error",
                message=message,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(item.id, item.status, history.id, warning=message)

        environment = provider_base._download_environment(
            config.download_proxy, Path(git_annex).parent
        )
        try:
            missing_files = _missing_annex_files(
                path,
                config,
                annex=git_annex,
                environment=environment,
            )
        except (AnnexInspectionError, OSError, subprocess.TimeoutExpired) as exc:
            message = f"Annex availability could not be determined: {exc}"
            _finish_history(
                db,
                item,
                history,
                result="error",
                message=message,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(item.id, item.status, history.id, warning=message)

        if not missing_files:
            return _finish_healthy_with_size(
                db,
                item,
                history,
                path,
                message="git-annex reports all annexed content is present locally.",
                status=DatasetStatus.AVAILABLE,
                config=config,
            )

        examples = ", ".join(missing_files[:5])
        missing_message = f"Annexed content is not present locally: {examples}"
        item.status = DatasetStatus.BROKEN
        db.commit()

        datalad = provider_base._find_command("datalad")
        if not datalad:
            message = missing_message + "; DataLad repair could not start: datalad is missing."
            _finish_history(
                db,
                item,
                history,
                result="broken",
                message=message,
                status=DatasetStatus.BROKEN,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(item.id, item.status, history.id, warning=message)

        try:
            repair = _run_command(
                [datalad, "get", "--recursive", "."],
                path,
                config,
                environment=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            repair_error = str(exc)
        else:
            repair_error = None if repair.returncode == 0 else _command_text(repair)

        if repair_error is not None:
            message = missing_message + f"; DataLad retrieval failed: {repair_error}"
            _finish_history(
                db,
                item,
                history,
                result="broken",
                message=message,
                status=DatasetStatus.BROKEN,
                repair_attempted=True,
                repair_succeeded=False,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(item.id, item.status, history.id, warning=message)

        try:
            remaining = _missing_annex_files(
                path,
                config,
                annex=git_annex,
                environment=environment,
            )
        except (AnnexInspectionError, OSError, subprocess.TimeoutExpired) as exc:
            message = (
                missing_message
                + f"; post-repair annex availability could not be determined: {exc}"
            )
            _finish_history(
                db,
                item,
                history,
                result="broken",
                message=message,
                status=DatasetStatus.BROKEN,
                repair_attempted=True,
                repair_succeeded=False,
                config=config,
                log_problem=True,
            )
            return HealthCheckReport(item.id, item.status, history.id, warning=message)

        if not remaining:
            return _finish_healthy_with_size(
                db,
                item,
                history,
                path,
                message="DataLad repaired all locally missing annexed content.",
                status=DatasetStatus.AVAILABLE,
                repair_attempted=True,
                repair_succeeded=True,
                config=config,
            )

        remaining_examples = ", ".join(remaining[:5])
        message = (
            missing_message
            + f"; annexed content is still absent after repair: {remaining_examples}"
        )
        _finish_history(
            db,
            item,
            history,
            result="broken",
            message=message,
            status=DatasetStatus.BROKEN,
            repair_attempted=True,
            repair_succeeded=False,
            config=config,
            log_problem=True,
        )
        return HealthCheckReport(item.id, item.status, history.id, warning=message)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _worker_environment(config: Settings) -> dict[str, str]:
    environment = os.environ.copy()
    environment["NDR_DATA_ROOT"] = str(config.data_root)
    environment["NDR_DATABASE_URL"] = config.resolved_database_url
    environment["NDR_HEALTH_COMMAND_TIMEOUT_SECONDS"] = str(
        config.health_command_timeout_seconds
    )
    if config.download_proxy:
        environment["NDR_DOWNLOAD_PROXY"] = config.download_proxy
    if config.download_mirror:
        environment["NDR_DOWNLOAD_MIRROR"] = config.download_mirror
    return environment


def launch_health_worker(
    config: Settings,
    *,
    dataset_id: str | None = None,
    history_id: str | None = None,
    all_datasets: bool = False,
) -> bool:
    """Launch a detached worker so query latency never includes network work."""
    command = [sys.executable, "-m", "neural_data_registry.health_worker"]
    if all_datasets:
        command.append("--all")
    elif dataset_id and history_id:
        command.extend(["--dataset-id", dataset_id, "--history-id", history_id])
    else:
        raise ValueError("Provide all_datasets or a dataset/history pair")
    try:
        subprocess.Popen(
            command,
            env=_worker_environment(config),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False
    return True


def request_health_check(
    dataset_id: str, config: Settings | None = None
) -> HealthCheckReport:
    """Run a fast query-time check and defer DataLad work to a worker."""
    config = config or get_settings()
    report, needs_datalad = _quick_check(dataset_id, config)
    if not needs_datalad:
        return report
    if launch_health_worker(
        config, dataset_id=dataset_id, history_id=report.history_id
    ):
        return report

    db = get_session_factory(config.resolved_database_url)()
    try:
        item = db.get(Dataset, dataset_id)
        history = db.get(HealthCheckHistory, report.history_id)
        assert item is not None and history is not None
        message = "Could not launch the background DataLad health worker."
        _finish_history(
            db,
            item,
            history,
            result="error",
            message=message,
            config=config,
            log_problem=True,
        )
        return HealthCheckReport(
            item.id, item.status, history.id, warning=message
        )
    finally:
        db.close()


def _mark_skipped(history_id: str, config: Settings) -> None:
    db = get_session_factory(config.resolved_database_url)()
    try:
        history = db.get(HealthCheckHistory, history_id)
        if history is None:
            return
        item = db.get(Dataset, history.dataset_id)
        if item is None:
            return
        _finish_history(
            db,
            item,
            history,
            result="skipped",
            message="Another health worker already holds the global lock.",
            config=config,
        )
    finally:
        db.close()


def run_health_checks(
    dataset_ids: Sequence[str] | None = None,
    config: Settings | None = None,
    *,
    pending_history: tuple[str, str] | None = None,
) -> bool:
    """Run one or all checks synchronously under the global worker lock."""
    config = config or get_settings()
    create_database(config)
    with process_lock("registry-health-worker", config) as acquired:
        if not acquired:
            if pending_history:
                _mark_skipped(pending_history[1], config)
            return False

        if pending_history:
            _complete_datalad_check(
                pending_history[0], pending_history[1], config
            )
            return True

        if dataset_ids is None:
            db = get_session_factory(config.resolved_database_url)()
            try:
                dataset_ids = list(db.scalars(select(Dataset.id)))
            finally:
                db.close()

        for dataset_id in dataset_ids:
            report, needs_datalad = _quick_check(
                dataset_id, config, refresh_size=True
            )
            if needs_datalad:
                _complete_datalad_check(dataset_id, report.history_id, config)
        return True


def maybe_launch_cooldown_check(config: Settings | None = None) -> bool:
    """Launch one all-dataset check per environment per 24-hour cooldown."""
    config = config or get_settings()
    ensure_layout(config)
    environment_id = str(Path(os.environ.get("VIRTUAL_ENV", sys.prefix)).resolve())
    marker_name = hashlib.sha256(environment_id.encode()).hexdigest()[:20] + ".stamp"
    marker = config.health_cooldown_dir / marker_name
    now = time.time()

    with process_lock("health-cooldown-marker", config) as acquired:
        if not acquired:
            return False
        if (
            marker.exists()
            and now - marker.stat().st_mtime < config.health_check_cooldown_seconds
        ):
            return False
        marker.touch()
        launched = launch_health_worker(config, all_datasets=True)
        if not launched:
            marker.unlink(missing_ok=True)
        return launched
