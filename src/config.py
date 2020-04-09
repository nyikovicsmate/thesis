import sys
import logging
import pathlib
import time

# set the project's root directory
ROOT_PATH = pathlib.Path(__file__).parent.parent

# https://docs.python.org/3/library/time.html#time.strftime
_timestamp = time.strftime("%Y_%m_%d_%M_%S")
_logfile = ROOT_PATH.joinpath(_timestamp + ".log")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
# https://docs.python.org/3/library/logging.html#logrecord-attributes
_formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)5s %(funcName)s(): %(message)s")

_handlers = [
    logging.StreamHandler(sys.stdout),   # stdout
    # TODO: uncomment in production
    # logging.FileHandler(_logfile)
]

for h in _handlers:
    h.setFormatter(_formatter)
    LOGGER.addHandler(h)




