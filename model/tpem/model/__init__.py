import logging


def create_model(opt):
    from .model import TPEM as M
    m = M(opt)
    return m
