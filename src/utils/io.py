"""
File I/O utilities for loading and saving data.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    filepath: Union[str, Path],
    fieldnames: List[str] = None
) -> None:
    """
    Save list of dictionaries to CSV file.

    Args:
        data: List of dictionaries
        filepath: Output file path
        fieldnames: Column names (auto-detected if None)
    """
    if not data:
        return

    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of dictionaries.

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_output_filename(
    prefix: str,
    extension: str,
    output_dir: Union[str, Path] = None,
    include_timestamp: bool = True
) -> Path:
    """
    Generate a unique output filename.

    Args:
        prefix: Filename prefix
        extension: File extension (without dot)
        output_dir: Output directory (uses current if None)
        include_timestamp: Whether to include timestamp

    Returns:
        Generated file path
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = ensure_dir(output_dir)

    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.{extension}"
    else:
        filename = f"{prefix}.{extension}"

    return output_dir / filename


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.

    Args:
        filepath: File path

    Returns:
        Dictionary with file information
    """
    path = Path(filepath)

    if not path.exists():
        return {"exists": False}

    stat = path.stat()

    return {
        "exists": True,
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
