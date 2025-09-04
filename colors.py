## This module allows to print colors and other text styles in the terminal using ANSI escape codes.

class Text:
    """
    Class Text contains the ANSI code to end the text.
    This must be concatenated as Text.end at the end of the text. Forgetting to close the text will cause that the rest of the printed text has the same format.
    """

    end = "\033[0m"

class Regular(Text):
    """
    This subclass contains two options: 1) Write text with default color in bold or underscored; 2) write text with normal style in color.
    This must be concatenated as Regular.* at  the beginning of the text to modify.
    """

    bold = "\033[1m"
    underline = "\033[4m"

    blue = "\033[0;34m"
    cyan = "\033[0;36m"
    green = "\033[0;32m"
    red = "\033[0;31m"
    yellow = "\033[0;33m"
    dark_gray = "\033[1;30m"

class Bold(Text):
    """
    This subclass contains allows to print bolded text in color.
    This must be concatenated as Bold.* at  the beginning of the text to modify.
    """

    blue = "\033[1;34m"
    cyan = "\033[1;36m"
    green = "\033[1;32m"
    red = "\033[1;31m"
    yellow = "\033[1;33m"
    dark_gray = "\033[1;30m"

class Underline(Text):
    """
    This subclass contains allows to print underscored text in color.
    This must be concatenated as Underline.* at  the beginning of the text to modify.
    """
    
    blue = "\033[4;34m"
    cyan = "\033[4;36m"
    green = "\033[4;32m"
    red = "\033[4;31m"
    yellow = "\033[4;33m"
    dark_gray = "\033[1;30m"

class Background(Text):
    """
    I do not remember what this subclass does.
    This must be concatenated as Background.* at  the beginning of the text to modify.
    """

    classic = "\033[0;30m\033[47m"