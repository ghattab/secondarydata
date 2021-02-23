def txt_category_to_dict(category_str):
    """
    Parameters
    ----------
    category_str: str of nominal values from dataset meta information

    Returns
    -------
    dict of the nominal values and their one letter encoding

    Example
    -------
    "bell=b, convex=x" -> {"bell": "b", "convex": "x"}
    """
    string_as_words = category_str.split()
    result_dict = {}
    for word in string_as_words:
        seperator_pos = word.find("=")
        key = word[: seperator_pos]
        val = word[seperator_pos + 1 :][0]
        result_dict[key] = val
    return result_dict


# from [a, b], [c, d] -> [a, b]; [c, d]
def replace_comma_in_text(text):
    result_text = ""
    replace = True
    for sign in text:
        if sign == '[':
            replace = False
        if sign == ']':
            replace = True
        if sign == ',':
            if replace:
                result_text += ';'
            else:
                result_text += sign
        else:
            result_text += sign

    return result_text


# ('list', 4) -> "list[0], list[1], list[2], list[3]"
def generate_str_of_list_elements_with_indices(list_name, list_size):
    result_str = ""
    for i in range(0, list_size):
        result_str += list_name + "[" + str(i) + "], "
    return result_str[: -2]


# checks if a str is a number that could be interpreted as a float
def is_number(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


d = txt_category_to_dict("cobwebby=c, evanescent=e, flaring=r, grooved=g")
print(d)
