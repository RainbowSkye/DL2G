import logging
logger = logging.getLogger('base')


def create_model(opt, m_items):
    from .model import DDPM as M
    m = M(opt, m_items)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
