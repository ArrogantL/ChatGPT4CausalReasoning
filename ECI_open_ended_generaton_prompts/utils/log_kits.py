# -*- coding: utf-8 -*-

import logging
import os

def get_simple_logger(target_file, is_file=True, is_console=True,level=logging.DEBUG, mode="a+"):
    logger = logging.getLogger(target_file)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(message)s')
    if is_file:
        if len(os.path.dirname(target_file))>0:
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
        handler = logging.FileHandler(target_file, mode=mode, encoding="UTF-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if is_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def get_logger(target_file, is_file=True, is_console=True,level=logging.DEBUG, mode="a+"):
    logger = logging.getLogger(target_file)
    logger.setLevel(level=level)
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
    if is_file:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        handler = logging.FileHandler(target_file, mode=mode, encoding="UTF-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if is_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

# # set logger
#     file_h=[]
#     if is_file:
#         os.makedirs(os.path.dirname(target_file), exist_ok=True)
#         file_h.append(logging.FileHandler(target_file, mode=mode, encoding="UTF-8"))
#     if is_console:
#         file_h.append(logging.StreamHandler())
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=level,
#         handlers=file_h
#     )
if __name__ == '__main__':
    logger = get_logger('test.log')
    logger.info('test')
