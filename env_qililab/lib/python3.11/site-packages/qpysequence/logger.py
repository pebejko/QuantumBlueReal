# Copyright 2023 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from loguru import logger

# Default log level (can be overridden by environment variable)
default_level = os.environ.get("QPYSEQUENCE_LOG_LEVEL", "INFO").upper()

# Remove any pre-existing handlers to avoid double logging
logger.remove()

# Add a default handler that writes to stderr
# You can customize formatting here. Loguru supports structured formatting,
# colorization, and other features.
logger.add(
    sys.stderr,
    level=default_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    backtrace=True,  # Allows better traceback
    diagnose=True,  # Provides more context in exceptions
)


def set_level(level: str):
    """Dynamically set the logging level at runtime."""
    # Remove all handlers and re-add with new level
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )


def add_file_logging(filepath: str, level: str = "DEBUG"):
    """Add a file handler for detailed logs."""
    logger.add(
        filepath,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} - {message}",
        rotation="1 week",
        retention="1 month",
        compression="zip",
    )
