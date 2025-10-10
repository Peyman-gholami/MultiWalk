import logging
import time
import os
from datetime import datetime
import pytz


class EventLogger:
    def __init__(self, log_file_name='events.log'):
        # Ensure log directory exists
        log_directory = './log/'
        os.makedirs(log_directory, exist_ok=True)

        self.log_file_path = os.path.join(log_directory, log_file_name)
        self.setup_logger()

        self.event_start_times = {}

    def setup_logger(self):
        # Create a logger and set the logging level
        self.logger = logging.getLogger(self.log_file_path)
        self.logger.setLevel(logging.INFO)

        # Check if the logger already has handlers
        if not self.logger.handlers:
            # Create a file handler that logs even debug messages
            file_handler = logging.FileHandler(self.log_file_path, mode='a')
            file_handler.setLevel(logging.INFO)

            # Create a logging format
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(file_handler)

    def log_start(self, event_name):
        if event_name in self.event_start_times:
            raise ValueError(f'Event "{event_name}" already exists. Cannot start the same event twice.')
        self.event_start_times[event_name] = time.time()

    def log_end(self, event_name, attribute_value):
        if not isinstance(attribute_value, dict):
            raise ValueError('attribute_value must be a dictionary')
        cst = pytz.timezone('US/Central')
        end_time = datetime.now(cst).strftime('%Y-%m-%d %H:%M:%S %Z')
        start_timestamp = self.event_start_times.pop(event_name, None)
        if start_timestamp is not None:
            duration = time.time() - start_timestamp
            start_time = datetime.fromtimestamp(start_timestamp, cst).strftime('%Y-%m-%d %H:%M:%S %Z')
            attributes_str = ', '.join(f'{key}: {value}' for key, value in attribute_value.items())
            self.logger.info(
                f'{event_name}, start_time: {start_time}, end_time: {end_time}, duration: {duration:.2f}s, attributes: {{{attributes_str}}}')
        else:
            self.logger.warning(f'End log for event {event_name} called without a corresponding start log.')
