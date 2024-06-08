import os

from enum import IntEnum

class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    SILENT = 4

log_level_env = os.environ.get('TRACEON_LOG_LEVEL')

if log_level_env is None or log_level_env.upper() not in dir(LogLevel):
    _log_level = LogLevel.INFO
else:
    _log_level = LogLevel[log_level_env.upper()]

def set_log_level(level):
    global _log_level
    assert isinstance(level, LogLevel)
    _log_level = level

def log_debug(msg):
    if _log_level <= LogLevel.DEBUG:
        print('DEBUG: ', msg)
 
def log_info(msg):
    if _log_level <= LogLevel.INFO:
        print(msg)
 
def log_warning(msg):
    if _log_level <= LogLevel.WARNING:
        print('WARNING: ', msg)

def log_error(msg):
    if _log_level <= LogLevel.ERROR:
        print('ERROR: ', msg)
