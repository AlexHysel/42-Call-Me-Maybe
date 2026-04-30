def special_to_standart(text: str) -> str:
    """
    Replaces special AI characters for space, tab and new line with
    standart human ones
    """
    return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')

def standart_to_special(text: str) -> str:
    """
    Replaces standart spaces, tabs and new line chars with special ones
    AI can understand
    """
    return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')

def escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('"', '\\"')