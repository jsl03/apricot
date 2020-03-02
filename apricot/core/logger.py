import logging

def get_logger():
    logger = logging.getLogger('apricot')
    logger.propagate = 0
    fmt = logging.Formatter(
        '%(levelname)s : %(name)s : %(module)s - %(funcName)s - %(message)s')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger
