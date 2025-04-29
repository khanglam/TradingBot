import logging
import os

def setup_logger(log_file='tradingbot.log'):
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

def log_trade(trade):
    logger.info(f"TRADE: {trade}")

def log_error(error):
    logger.error(f"ERROR: {error}")

def log_event(event):
    logger.info(f"EVENT: {event}")
