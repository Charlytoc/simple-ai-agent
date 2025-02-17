def colorize(text, color):
    colors = {
        "reset": "\033[0m",
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",  
        "cyan": "\033[36m",
        "white": "\033[37m",
    }
    return f"{colors[color]}{text}{colors['reset']}"


class Printer:
    # ANSI escape sequences for colors
    colors = {
        "reset": "\033[0m",
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def __init__(self, identifier=None):
        self.identifier = identifier

    def _color_text(self, text, color):
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def print_colored(self, *args, color="reset", sep=" ", end="\n"):
        if self.identifier:
            print(f"[{self.identifier}] ", end="")
        colored_texts = [self._color_text(str(arg), color) for arg in args]
        print(sep.join(colored_texts), end=end)
        return ""

    def red(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="red", sep=sep, end=end)

    def green(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="green", sep=sep, end=end)

    def yellow(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="yellow", sep=sep, end=end)

    def blue(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="blue", sep=sep, end=end)

    def magenta(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="magenta", sep=sep, end=end)

    def cyan(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="cyan", sep=sep, end=end)

    def white(self, *args, sep=" ", end="\n"):
        self.print_colored(*args, color="white", sep=sep, end=end)
