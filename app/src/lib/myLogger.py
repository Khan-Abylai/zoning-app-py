import datetime
import logging

loggers = {}


def myLogger(name):
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    now = datetime.datetime.now()
    handler = logging.FileHandler(now.strftime("%Y-%m-%d") + '.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    loggers[name] = logger

    return logger
