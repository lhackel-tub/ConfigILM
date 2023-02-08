import gc
import sys

import numpy as np


def _indent(s, num_spaces=0, indent_first=False):
    """
    Indents a string by a number of spaces. Indents every line individually except for
    the first line if indent_first is not set
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
    # color prefixes for use in print
    # e.g. print(f"{bcolors.WARNING}This is a Warning.{bcolors.ENDC}")
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def warn(cls, message: str):
        print(f"\n{cls.WARNING}WARNING: {_indent(message, 9)}{cls.ENDC}")

    @classmethod
    def error(cls, message: str):
        print(f"\n{cls.FAIL}ERROR: {_indent(message, 7)}{cls.ENDC}")

    @classmethod
    def success(cls, message: str):
        print(f"\n{cls.OKGREEN}SUCCESS: {_indent(message, 10)}{cls.ENDC}")

    @classmethod
    def hint(cls, message: str):
        print(f"\n{cls.OKCYAN}HINT: {_indent(message, 6)}{cls.ENDC}")


def round_to(x, base):
    """
    Rounds to an arbitrary base, e.g. 5 or 2 or 0.002
    Result is not exact
    :param x: number to be rounded
    :param base: base
    :return: rounded number
    """
    return base * round(x / base)


def convert(multi_true, multi_pred):
    """
    Helper function to convert labels and logits into a format that is accepted by
    wandb.plot
    :param multi_true: list of multi hot labels of value 0 or 1, array has dimension n*d
    :param multi_pred: list of logits, array has dimension n*d
    :return: labels and logits as requested by wandb.plot
    """
    # convert tensor to array
    multi_pred = [list(np.asarray(x)) for x in multi_pred]
    # convert multi hot to index list
    # tensor(0,0,1,0,0,0,1,..,0)
    # -> array(2,6,...)
    multi_true = [list(np.where(x == 1)[0]) for x in multi_true]

    t = []
    p = []
    for i, pred in enumerate(multi_pred):
        for clss in multi_true[i]:
            t.append(clss)
            p.append(pred)
    return t, p


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {
            o_id: o
            for o_id, o in all_refr
            if o_id not in marked and not isinstance(o, type)
        }

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids, so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def huggingface_tokenize_and_pad(tokenizer, string: str, seq_length: int):
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
