import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional
from functools import wraps
import json

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure and return a logger for the application"""
    # Determine log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger("superdoc")
    
    # Add file handler if LOG_FILE environment variable is set
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger

def timed_execution(logger: Optional[logging.Logger] = None):
    """Decorator to measure and log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log execution time
            if logger:
                logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator

async def async_timed_execution(logger: Optional[logging.Logger] = None):
    """Decorator to measure and log async function execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log execution time
            if logger:
                logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator

def get_domain_handler(domain: str):
    """Import and return appropriate domain handler based on domain name"""
    # This function is used in the main app but implemented here to avoid circular imports
    from domain_handlers import InsuranceHandler, LegalHandler
    
    domain_handlers = {
        "insurance": InsuranceHandler(),
        "legal": LegalHandler(),
        # Add more domain handlers as they become available
    }
    
    # Return requested domain handler or default to insurance
    return domain_handlers.get(domain, domain_handlers.get("insurance"))

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string with error handling"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def get_env_boolean(env_var: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(env_var, str(default)).lower()
    return value in ('true', '1', 't', 'yes', 'y')

def get_env_list(env_var: str, default: List = None, separator: str = ',') -> List:
    """Get list from environment variable with separator"""
    if default is None:
        default = []
    value = os.getenv(env_var)
    if not value:
        return default
    return [item.strip() for item in value.split(separator)]

def get_env_int(env_var: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(env_var, default))
    except (ValueError, TypeError):
        return default

class LimitedSizeDict(dict):
    """Dictionary with limited size, ejecting oldest items"""
    
    def __init__(self, size_limit: int = 1000, *args, **kwargs):
        self.size_limit = size_limit
        self.time_added = {}
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(self) >= self.size_limit:
            self._remove_oldest()
        super().__setitem__(key, value)
        self.time_added[key] = time.time()

    def _remove_oldest(self):
        """Remove the oldest item from the dictionary"""
        if not self:
            return
            
        oldest_key = min(self.time_added.items(), key=lambda x: x[1])[0]
        del self[oldest_key]
        del self.time_added[oldest_key]