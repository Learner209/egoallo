# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys

from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        # Customize prefixes and styles for different log levels
        if record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "dark_grey", attrs=["bold"])
            log = colored(log, "dark_grey", attrs=["bold"])
        elif record.levelno == logging.INFO:
            prefix = colored("INFO", "magenta", attrs=["bold"])
            log = colored(log, "magenta", attrs=["bold"])
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "light_yellow", attrs=["bold", "blink"])
            log = colored(log, "light_yellow", attrs=["bold", "blink"])
        elif record.levelno == logging.ERROR:
            prefix = colored("ERROR", "red", attrs=["bold", "underline"])
            log = colored(log, "red", attrs=["bold", "underline"])
        elif record.levelno == logging.CRITICAL:
            prefix = colored("CRITICAL", "white", attrs=["bold", "blink", "underline"])
            log = colored(log, "white", attrs=["bold", "blink", "underline"])
        else:
            return log  # Return the original log if no specific styling is required

        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(output=None, distributed_rank=0, *, color=True, name="imagenet", abbrev_name=None, level=logging.DEBUG):
    """
    Initialize the detectron2 logger and set its verbosity level.

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        level: The logging level to use. Can be a logging constant (e.g. logging.DEBUG)
            or string name (e.g. "DEBUG"). Defaults to logging.DEBUG.

    Returns:
        logging.Logger: a logger
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False

        if abbrev_name is None:
            abbrev_name = name

        plain_formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(pathname)s:%(lineno)d: %(message)s",
            datefmt="%m/%d %H:%M:%S"
        )

        # stdout logging: master only
        if distributed_rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(level)
            if color:
                formatter = _ColorfulFormatter(
                    "[%(asctime)s.%(msecs)03d]: " + "%(pathname)s:%(lineno)d: " + "%(message)s",
                    datefmt="%m/%d %H:%M:%S",
                    root_name=name,
                    abbrev_name=str(abbrev_name),
                )
            else:
                formatter = plain_formatter
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # file logging: all workers
        if output is not None:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "log.txt")
            if distributed_rank > 0:
                filename = filename + f".rank{distributed_rank}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(level)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


if __name__ == "__main__":
    logger = setup_logger(name=__name__, abbrev_name="main:logger")