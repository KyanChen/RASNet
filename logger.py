import sys
import time


class Logger(object):
    def __init__(self, filename="Default.log", is_terminal_show=True):
        self.is_terminal_show = is_terminal_show
        if self.is_terminal_show:
            self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.write('*'*50)
        self.write('Init Logger!!!')
        self.write('*'*50)

    def write(self, message, with_time=True):
        if with_time:
            message = time.strftime('%Y-%m-%d %H:%M:%S  ', time.localtime()) + message
        message = message + '\n'
        if self.is_terminal_show:
            self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        if self.is_terminal_show:
            self.terminal.flush()
        self.log.flush()

