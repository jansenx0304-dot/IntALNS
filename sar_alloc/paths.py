from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
