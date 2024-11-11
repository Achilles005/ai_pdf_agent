import json
from logging_config import LoggingConfig

logger = LoggingConfig().get_logger()
config = json.load(open("config.json"))

def build_app():
    try:
        open_ai_key = config["OPENAI_API_KEY"]
        slack_webhook_url = config["SLACK_WEBHOOK_URL"]
        return open_ai_key , slack_webhook_url
    except Exception as e:
        logger.error(f"{str(e)}", exc_info=True)