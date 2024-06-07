from pathlib import Path

from transformers import BertTokenizer


def get_default_tokenizer():
    default_tokenizer = (
        Path(__file__).parent.joinpath("extra", "huggingface_tokenizers", "bert-base-uncased.tok").resolve(True)
    )
    return BertTokenizer.from_pretrained(default_tokenizer.resolve())


def indent(s, num_spaces=0, indent_first=False):
    """
    Indents a string by a number of spaces. Indents every line individually except for
    the first line if indent_first is not set.

    :param s: string to indent
    :param num_spaces: number of spaces to indent
    :param indent_first: if set, first line is indented as well, otherwise no spaces
                         added to first line
    :return: s with each line indented
    """
    # split by newline
    s = str.split(s, "\n")
    # add num_spaces spaces to front of each line except the first
    if indent_first:
        s = [(num_spaces * " ") + str.lstrip(line) for line in s]
    else:
        first_line = s[0]
        s = [(num_spaces * " ") + str.lstrip(line) for line in s[1:]]
        s = [first_line] + s
    # join with spaces
    s = "\n".join(s)
    return s


class Messages:
    """
    Class to print messages. Defines colors and formatting for bash shells.
    Class methods are wrapper for specific formatting combinations and can be called
    without defining an object.

    :FIELDS:
        :HEADER: purple
        :OKBLUE: blue
        :OKCYAN: blue - used in HINT messages
        :OKGREEN: green - used in SUCCESS messages
        :WARNING: yellow - used in WARNING messages
        :FAIL: red - used in ERROR messages
        :ENDC: formatting: reverts to default
        :BOLD: formatting: bolt font
        :UNDERLINE: formatting: underlined font

    Example::

        print(f"{Messages.WARNING}WARNING: This is a Warning.{Messages.ENDC}")
        Messages.warn("This is a Warning.")  # equal result
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    max_indent = 10
    start_msg = ""

    @classmethod
    def warn(cls, message: str) -> None:
        """
        Prints a yellow warning message with aligned indent and "[WARNING]".
        If the message has multiple lines, it is aligned to the right of the colon.

        :param message: Message to print
        """
        print(f"{cls.start_msg}{cls.WARNING}[WARNING] {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def error(cls, message: str) -> None:
        """
        Prints a red error message with aligned indent and "[ERROR]".
        If the message has multiple lines, it is aligned to the right of the colon.

        :param message: Message to print
        """
        print(f"{cls.start_msg}{cls.FAIL}[ERROR]   {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def success(cls, message: str) -> None:
        """
        Prints a green success message with aligned indent and "[SUCCESS]"
        If the message has multiple lines, it is aligned to the right of the colon.

        :param message: Message to print
        """
        print(f"{cls.start_msg}{cls.OKGREEN}[SUCCESS] {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def hint(cls, message: str) -> None:
        """
        Prints a blue hint message with aligned indent and "[HINT]".
        If the message has multiple lines, it is aligned to the right of the colon.

        :param message: Message to print
        """
        print(f"{cls.start_msg}{cls.OKCYAN}[HINT]    {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def info(cls, message: str) -> None:
        """
        Prints a dark blue info message with aligned indent and "[INFO]".
        If the message has multiple lines, it is aligned to the right of the colon.

        :param message: Message to print
        """
        print(f"{cls.start_msg}{cls.OKCYAN}[INFO]    {indent(message, cls.max_indent)}{cls.ENDC}")


def round_to(x, base):
    """
    Rounds to an arbitrary base, e.g. 5 or 2 or 0.002.

    Result is not exact.

    :param x: number to be rounded
    :param base: base
    :return: rounded number

    :Example:
        >>> round_to(5, 2)
        4
        >>> round_to(5.1, 0.2)
        5.0
        >>> round_to(5.199, 0.2)
        5.2
    """
    return base * round(x / base)


def convert(multi_true, multi_pred):
    """
    Helper function to convert labels and logits into a format that is accepted by
    wandb.plot. Applicable for multi-class classification (not multi-label).

    Example: tensor(0,0,1,0,0,0,1,...,0) will become array(2,6,...) for ground truth
    input (multi_true)

    :param multi_true: list of multi hot labels of value 0 or 1, array has dimension n*d
    :param multi_pred: list of logits, array has dimension n*d
    :return: (labels, logits) as requested by wandb.plot
    """
    # convert tensor to array
    multi_pred = multi_pred.tolist()
    # convert multi hot to index list
    # tensor(0,0,1,0,0,0,1,..,0)
    # -> array(2,6,...)
    multi_true = [[i for i, e in enumerate(sl) if e == 1] for sl in multi_true.tolist()]

    t = []
    p = []
    for i, pred in enumerate(multi_pred):
        for clss in multi_true[i]:
            t.append(clss)
            p.append(pred)
    return t, p


class AverageMeter:
    """
    Computes and stores the average and current value of some kind of sequential calls.
    """

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self, name: str, fmt: str = ":f"):
        """
        Initializes the counter.

        :param name: Name of the counter during __str__
        :param fmt: format of the numbers during __str__
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """
        Resets the counter and all properties to the default value 0.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        """
        Update the AverageMeter with a new value or the same value multiple times.

        :param val: value to update the AverageMeter with
        :param n: how often to update with this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """
        Formatting the latest value and the average collected.

        :return: formatted string according to format specified during __init__
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def huggingface_tokenize_and_pad(tokenizer, string: str, seq_length: int):
    """
    Tokenizes a string given a huggingface tokenizer.
    Assumes, that the tokenizer has a "[CLS]" start token, a "[SEP]" end token and a
    "[PAD]" padding token.
    If the string is too long it will be cut to the specific length.
    It the string is too short, it will be padded with the "[PAD]" token.

    :param tokenizer: hugging face tokenizer
    :param string: string to be encoded
    :param seq_length: length of the output
    :return: list of integers (IDs) of the tokenized string of length seq_length
    """
    # tokenize the questions
    token_start = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    token_end = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    token_pad = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]

    tokens = tokenizer.tokenize(string)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # get sublist if length of sequence is longer than max length
    ids = ids[0 : seq_length - 2] if len(ids) >= seq_length - 2 else ids
    # prepend start token
    ids.insert(0, token_start)
    # append end token
    ids.append(token_end)
    # pad if required
    ids.extend([token_pad] * (seq_length - len(ids)))

    return ids
