"""Data ingestion module for WhatsApp conversation datasets."""

from src.ingestion.loader import DataValidationError, load_dataset

__all__ = ["load_dataset", "DataValidationError"]

