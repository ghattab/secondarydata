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


def replace_comma_in_text(text):
    """
    Parameters
    ----------
    text: str of nominal values for a single mushroom species from primary_data_edited.csv

    Returns
    -------
    replace commas outside of angular brackets with semicolons (but not inside of them)

    Example
    -------
    text = "[a, b], [c, d]"
    return: "[a, b]; [c, d]"
    """

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


def generate_str_of_list_elements_with_indices(list_name, list_size):
    """
    Parameters
    ----------
    list_name: str, name of the list
    list_size: int, number of list elements

    Returns
    -------
    str of list elements with angular bracket indexation separated with commas

    Example
    -------
    list_name = "l"
    list_size = 3
    return = "l[0], l[1], l[2]"
    """

    result_str = ""
    for i in range(0, list_size):
        result_str += list_name + "[" + str(i) + "], "
    return result_str[: -2]


# checks if a str is a number that could be interpreted as a float
def is_number(val):
    """
    Parameters
    ----------
    val: str, arbitrary input

    Returns
    -------
    bool, True if val is interpretable as a float and False else
    """

    try:
        float(val)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    print(txt_category_to_dict("cobwebby=c, evanescent=e, flaring=r, grooved=g"))

