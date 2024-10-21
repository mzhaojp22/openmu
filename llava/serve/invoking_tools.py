import re
from beartype import beartype
from beartype.typing import Optional, Union
from functools import partial, wraps


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def identity(t):
    return t


def exists(val):
    return val is not None


def try_except(fn, callback=identity):
    @wraps(fn)
    def inner(*args):
        try:
            return fn(*args)
        except Exception as e:
            return callback(e)

    return inner


# invoking api call functions


def is_valid_string(s):
    return exists(re.fullmatch(r"'[^']*'|\"[^\"]*\"", s))


def is_valid_integer(s):
    return exists(re.fullmatch(r"[+-]?\d+", s))


def is_valid_float(s):
    return exists(re.fullmatch(r"[+-]?\d+(\.\d+)?", s))


def parse_param(s: str) -> Optional[Union[int, float, str]]:
    if is_valid_string(s):
        return str(s)
    elif is_valid_integer(s):
        return int(s)
    elif is_valid_float(s):
        return float(s)

    return None


@beartype
def replace_fn(registry, matches, delimiter="→"):
    orig_text = matches.group(0)

    text_without_end_api_token = matches.group(1)
    end_api_token = matches.group(4)
    function_name = matches.group(2)

    # unable to find function in registry

    if function_name not in registry:
        return orig_text

    fn = registry[function_name]

    params = matches.group(3).split(",")
    params = list(map(lambda s: s.strip(), params))
    params = list(filter(len, params))
    params = list(map(parse_param, params))

    # if any of the parameters are not parseable, return

    if any([(not exists(p)) for p in params]):
        return orig_text

    # just return original text if there is some error with the function

    out = try_except(fn, always(None))(*params)

    # the api calling function can also arrest the process, by returning None

    if not exists(out):
        return orig_text

    # return original text with the output delimiter and the stringified output

    return f"{text_without_end_api_token} {delimiter} {str(out)} {end_api_token}"


# main function, which takes a registry of functions, the text in question, and
# makes all the appropriate api calls and append the output


def create_function_regex(api_start=" [", api_stop="]"):
    api_start_regex, api_stop_regex = map(re.escape, (api_start, api_stop))
    return rf"({api_start_regex}(\w+)\(([^)]*)\))({api_stop_regex})"


def num_matches(substr: str, text: str):
    return len(re.findall(re.escape(substr), text))


def has_api_calls(text, api_start=" [", api_stop="]"):
    regex = create_function_regex(api_start, api_stop)
    matches = re.findall(regex, text)
    return len(matches) > 0


def replace_all_but_first(text: str, api_start=" [", api_stop="]") -> str:
    regex = create_function_regex(api_start, api_stop)

    count = 0

    def replace_(matches):
        orig_text = matches.group(0)
        nonlocal count
        count += 1
        if count > 1:
            return ""
        return orig_text

    return re.sub(regex, replace_, text)


def invoke_tools(
    registry,
    text: str,
    delimiter: str = "→",
    api_start=" [",
    api_stop="]",
) -> str:
    regex = create_function_regex(api_start, api_stop)
    replace_ = partial(replace_fn, registry, delimiter=delimiter)
    return re.sub(regex, replace_, text)


# # text = "make these calls: [GetAudioChords()], [GetAudioDownbeat()], [GetAudioTempo()] and [GetAudioKey()]."
# text = "The tempo is [GetAudioTempo()] beats per minute."
# print(invoke_tools(function_registry, text))
