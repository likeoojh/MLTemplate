from typing import Any, Dict, List, Optional
from unicodedata import normalize
import hashlib
import io
import re
import pickle


def normalize_str(
    inputs: str, 
    type: str = "NFD",
) -> str:
    """
    Normalize string with NFD
    Change non-alphanumeric `\W == [^a-zA-Z0-9_]` into `_`
    Make string to lowercase
    Args:
        inputs (str): input string to be normalized
    Returns:
        str: normalized string
    """
    normalize_inputs = normalize(type, inputs)
    normalize_inputs = re.sub("#+", "num", normalize_inputs)
    normalize_inputs = re.sub("\W+", "_", normalize_inputs)
    normalize_inputs = normalize_inputs.lower()
    return normalize_inputs


def normalize_list(
    params: Optional[List]
) -> List:
    """
    Normalize string in each element of input list
    Args:
        params (list): input list to be normalized
    Returns:
        list: normalized list
    """
    if params is None:
        return []
    else:
        return [normalize_str(col) for col in params]


def object_hash(
    object: Dict[Any, Any],
) -> str:
    """
    Hash object with MD5
    Args:
        object (dict): Input dict

    Returns:
        str: hased string
    """
    m = hashlib.md5()
    with io.BytesIO() as bio:
        pickle.dump(object, bio)
        data = bio.getvalue()
        m.update(data)
    return m.hexdigest()


def list_to_tuple_string(array: List[str]) -> str:
    """
    Change list into lowered string with tuple form
    Args:
        array (List): input array
    Returns:
        str: string in tuple form
    """
    if len(array) == 1:
        return f"('{array[0].lower()}')"
    elif len(array) > 1:
        return f"{tuple(i.lower() for i in array)}"
