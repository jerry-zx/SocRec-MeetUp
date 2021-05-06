import logging

def init_logger(logger_save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setLevel(logging.DEBUG)
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(logger_save_path, mode='w')
    fhlr.setLevel(logging.DEBUG)
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger

