import logging


def setup_logger(name, log_file, level=logging.INFO, formatter=None):
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    elif isinstance(formatter, str):
        formatter = logging.Formatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
