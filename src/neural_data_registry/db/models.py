from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from neural_data_registry.enums import DatasetStatus, JobStatus, Provider, StorageMode


class Base(DeclarativeBase):
    """Base class for all registry database models."""


class Dataset(Base):
    """A dataset registered in the registry.

    Notes
    -----
    Dataset rows are append-only in the service layer; no delete operation is
    exposed, so registered data remains auditable for its lifetime.
    """
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("provider", "source_url", "version", name="uq_dataset_source_version"),)
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
        info={"output_name": "dataset_id", "label": "Dataset ID"},
    )
    name: Mapped[str] = mapped_column(String(255), index=True)
    provider: Mapped[Provider] = mapped_column(SqlEnum(Provider, native_enum=False), index=True)
    source_url: Mapped[str | None] = mapped_column(
        String(2048), nullable=True, index=True, info={"cli_hidden": True}
    )
    version: Mapped[str] = mapped_column(String(128), default="unknown")
    modalities: Mapped[str] = mapped_column(Text, default="")
    size_bytes: Mapped[int] = mapped_column(
        Integer, default=0, info={"label": "Size"}
    )
    status: Mapped[DatasetStatus] = mapped_column(SqlEnum(DatasetStatus, native_enum=False), default=DatasetStatus.INGESTING)
    storage_path: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"cli_hidden": True}
    )
    storage_mode: Mapped[StorageMode] = mapped_column(
        SqlEnum(StorageMode, native_enum=False),
        default=StorageMode.REFERENCE,
        server_default=StorageMode.REFERENCE.name,
        info={"cli_hidden": True},
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), info={"serialize": False}
    )
    aliases: Mapped[list["DatasetAlias"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class DatasetAlias(Base):
    """An alternate identifier that resolves to a registered dataset."""
    __tablename__ = "dataset_aliases"
    __table_args__ = (UniqueConstraint("value", name="uq_dataset_alias_value"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    value: Mapped[str] = mapped_column(String(2048))
    dataset: Mapped[Dataset] = relationship(back_populates="aliases")


class IngestionJob(Base):
    """The audit record for one dataset ingestion attempt."""
    __tablename__ = "ingestion_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    status: Mapped[JobStatus] = mapped_column(SqlEnum(JobStatus, native_enum=False), default=JobStatus.PENDING)
    mode: Mapped[str] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
