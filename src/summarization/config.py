from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path="configs/local.yaml"):
    """Load project settings from a YAML file."""
    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    for section in ("project", "data", "paths"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    for name in ("data_dir", "artifacts_dir", "reports_dir"):
        value = config["paths"].get(name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid path: {name}")

        output_path = Path(value).expanduser()

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        config["paths"][name] = output_path.resolve()

    return config
