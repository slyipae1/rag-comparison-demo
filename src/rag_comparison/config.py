"""Runtime settings. All paths are local by default and env-overridable."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_DIR: str = "./logs"
    PLOT_DIR: str = "./plots"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT: int = 5


settings = Settings()
