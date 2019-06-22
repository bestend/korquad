import os
import re

MODEL_FILE_FORMAT = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
MODEL_REGEX_PATTERN = re.compile(r'^.*weights\.(\d+)\-\d+\.\d+\.h5$')
LAST_MODEL_FILE_FORMAT = 'last.h5'
TEAMS_WEBHOOK_URL = os.environ.get('TEAMS_WEBHOOK_URL', '')