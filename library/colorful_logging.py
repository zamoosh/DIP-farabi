from inspect import currentframe, stack
import logging
import datetime
import sys

ACKNOWLEDGE = 28
SUCCESS = 25
SAY = 15
LOG = 10
BOLD = 10
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(SAY, "SAY")
logging.addLevelName(ACKNOWLEDGE, "ACKNOWLEDGE")
logging.addLevelName(BOLD, "LOG")

RESET = "\033[0m"
UNDERLINE = '\033[4m'
BLACK = '\033[30m'
DARK_RED = '\033[31m'
DARK_GREEN = '\033[32m'
DARK_YELLOW = "\x1b[33m"
DARK_BLUE = "\033[34m"
DARK_PINK = "\033[35m"
DARK_CYAN = "\033[36m"

GRAY = "\033[37m"

BRIGHT_BLACK = '\033[90m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PINK = '\033[95m'
CYAN = '\033[96m'
BOLD_FONT = '\033[97m'


class ColorfulLogging(logging.Formatter):
    FORMAT = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s, func: '%(funcName)s', msg: '%(message)s'"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    FORMATS = {
        logging.CRITICAL: RED + FORMAT + RESET,
        logging.FATAL: RED + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.WARN: DARK_YELLOW + FORMAT + RESET,
        logging.INFO: BLUE + FORMAT + RESET,
        logging.DEBUG: DARK_RED + FORMAT + RESET,
        ACKNOWLEDGE: PINK + FORMAT + RESET,
        SUCCESS: GREEN + FORMAT + RESET,
        SAY: GRAY + FORMAT + RESET,
        LOG: FORMAT,
        BOLD: BOLD_FONT + FORMAT + RESET,
        logging.NOTSET: CYAN + FORMAT + RESET,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def color_print(text: str, color: str = RESET, fmt: bool = False, lvl: str = "LOG") -> None:
    """
    Possible values for is as follows:
    ['red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'gray', 'black', 'dark red', 'dark green', 'dark yellow', 'dark blue', 'dark pink', 'dark cyan', 'bright black']
    """
    
    possible_colors = {
        'red': RED, 'green': GREEN, 'yellow': YELLOW, 'blue': BLUE, 'pink': PINK,
        'cyan': CYAN, 'gray': GRAY, 'black': BLACK, 'bold': BOLD_FONT,
        'dark red': DARK_RED, 'dark green': DARK_GREEN, 'dark yellow': DARK_YELLOW,
        'dark blue': DARK_BLUE, 'dark pink': DARK_PINK,
        'dark cyan': DARK_CYAN, 'bright black': BRIGHT_BLACK
    }
    color = color.lower()
    out_text: str = text
    if color in possible_colors:
        color = possible_colors[color]
        out_text = f"{color}{text}"
    if fmt:
        # has a bug, when we have '%' in string. need to be fixed
        out_text = f"{color}"
        now = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
        frame = currentframe()
        line = f'{frame.f_back.f_lineno}'
        fn = f'{sys._getframe().f_back.f_code.co_name}'
        out_text += f'{lvl}    {now}, File "%s", Line {line}, in "{fn}", msg: {text}'
        file = f'{UNDERLINE}{stack()[1].filename.split("/")[-1]}{RESET}{color}'
        out_text = out_text % file
    out_text += RESET
    out_text += "\n"
    sys.stderr.write(out_text)


class CustomLogger(logging.Logger):
    
    def __init__(
            self,
            name: str,
            level: int = logging.DEBUG,
            file: bool = False,
            filename: str = 'logfile.log',
            mode: str = 'a'
    ):
        super(CustomLogger, self).__init__(name, level)
        
        self.setLevel(level=level)
        
        if file:
            handler = logging.FileHandler(filename=filename, mode=mode)
            handler.setLevel(level=level)
            handler.setFormatter(logging.Formatter(ColorfulLogging.FORMAT, datefmt=ColorfulLogging.DATE_FORMAT))
        else:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(ColorfulLogging())
        
        self.addHandler(handler)
    
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)
    
    def say(self, msg, *args, **kwargs):
        if self.isEnabledFor(SAY):
            self._log(SAY, msg, args, **kwargs)
    
    def acknowledge(self, msg, *args, **kwargs):
        if self.isEnabledFor(ACKNOWLEDGE):
            self._log(ACKNOWLEDGE, msg, args, **kwargs)
    
    def simple(self, msg, *args, **kwargs):
        if self.isEnabledFor(LOG):
            self._log(LOG, msg, args, **kwargs)
    
    def bold(self, msg, *args, **kwargs):
        if self.isEnabledFor(BOLD):
            self._log(BOLD, msg, args, **kwargs)
