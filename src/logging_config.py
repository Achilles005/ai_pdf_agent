from loguru import logger

class LoggingConfig:
    """
    Configures and manages logging for the application.
    """
    def __init__(self, log_file: str = "logs/app.log"):
        self.log_file = log_file
        self._setup_logging()

    def _setup_logging(self):
        """
        Sets up logging with Loguru.
        """
        logger.add(self.log_file, rotation="1 MB", retention="10 days", level="INFO")

    def get_logger(self):
        """
        Get the Loguru logger instance.
        """
        return logger
