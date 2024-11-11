import requests
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()

class SlackIntegration:
    """
    Handles Slack integration using a webhook URL to post messages.
    """
    def __init__(self, webhook_url: str):
        """
        Initialize the SlackIntegration class.

        Args:
            webhook_url (str): Slack webhook URL for sending messages.
        """
        self.webhook_url = webhook_url

    def post_message(self, message: str):
        """
        Posts a message to Slack using the webhook URL.

        Args:
            message (str): The message to post.
        """
        payload = {"text": message}
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Message posted to Slack successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error posting message to Slack: {str(e)}", exc_info=True)
            raise
