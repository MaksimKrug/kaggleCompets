import logging

logging.basicConfig(
    filename="src/logs/logs.log", format="%(asctime)s:%(levelname)s:%(message)s"
)
logger = logging.getLogger("parent")
logger.setLevel(logging.INFO)
