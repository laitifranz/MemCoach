import logging.config
import sys
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv()

_config_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color": {
            "class": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(levelname)-8s%(reset)s | %(filename)s:%(lineno)d | %(message)s",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
        "standard": {
            "format": "[%(levelname)-8s] %(filename)s:%(lineno)d | %(message)s"
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"level": "INFO", "handlers": ["stdout", "stderr"]},
}


def setup_logging(force_color: bool | None = None) -> None:
    # Default: color only when stdout is a TTY and we’re not inside a batch job
    use_color = force_color if force_color is not None else sys.stdout.isatty()

    cfg = deepcopy(_config_dict)
    formatter = "color" if use_color else "standard"
    cfg["handlers"]["stdout"]["formatter"] = formatter
    cfg["handlers"]["stderr"]["formatter"] = formatter
    logging.config.dictConfig(cfg)
