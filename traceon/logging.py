import os

from enum import IntEnum

__pdoc__ = {}
__pdoc__['log_info'] = False
__pdoc__['log_debug'] = False
__pdoc__['log_warning'] = False
__pdoc__['log_error'] = False


class LogLevel(IntEnum):
    """Enumeration representing a certain verbosity of logging."""
    
    DEBUG = 0
    """Print debug, info, warning and error information."""

    INFO = 1
    """Print info, warning and error information."""

    WARNING = 2
    """Print only warnings and errors."""

    ERROR = 3
    """Print only errors."""
     
    SILENT = 4
    """Do not print anything."""


_log_level_env = os.environ.get('TRACEON_LOG_LEVEL')

if _log_level_env is None or _log_level_env.upper() not in dir(LogLevel):
    _log_level = LogLevel.INFO
else:
    _log_level = LogLevel[_log_level_env.upper()]

def set_log_level(level):
    """Set the current `LogLevel`. Note that the log level can also 
    be set by setting the environment value TRACEON_LOG_LEVEL to one
    of 'debug', 'info', 'warning', 'error' or 'silent'."""
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
