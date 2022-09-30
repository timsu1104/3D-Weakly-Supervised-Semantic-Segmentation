import json
from typing import Dict, List
from munch import Munch

def read_attr(path):
    """Read attribute from a json file. """
    with open(path) as f:
        data = json.load(f)
    d = [
        Munch.fromDict(dict(
            name=ent["object_name"].replace(" ", "").replace("_", " "),
            attr=ent["object_attrbution"]
        ))
        for ent in data
    ]
    return d

def compose_text(entities:List[Dict], mode:str='compose'):
    """Generate text according to a list of attribute. """
    texts = []
    if mode == 'compose':
        for e in entities:
            name, attr = e.name, e.attr
            words = ["a"]
            words.extend(attr.size)
            words.extend(attr.shape)
            words.extend(attr.color)
            words.append(name)
            text = " ".join(words)
            texts.append(text)
    return texts

def generate_text(path:str, mode:str='compose'):
    """Generate text from attribute json. """
    d = read_attr(path)
    texts = compose_text(d, mode=mode)
    return texts