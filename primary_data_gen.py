import os

import text_attr_match
import data_cat

"""
WARNING: Cannot be run since the used source book is not freely available. To run this module,
a EPUB copy of the book has to be acquired and the unpacked HTML files put into data/mushrooms_and_toadstools/.
The generated data set primary_data_generated.csv is available as well as manually edited an enriched
version primary_data_edited.csv. This version is relevant and used by the other modules. This module
is mainly for transparency.
"""


class FunghiType:
    """
    Container class representing a mushroom species
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        *args:
        [0]: family, [1]: name, [2]: is_edible
        [3:23]: categories/attributes like cap-diameter
        """

        self.family = args[0]
        self.name = args[1]
        self.is_edible = args[2]
        self.categories_dict = {
            "cap-diameter": args[3], "cap-shape": args[4],
            "cap-surface": args[5], "cap-color": args[6],
            "has-bruises": args[7], "gill-attachment": args[8],
            "gill-spacing": args[9], "gill-color": args[10],
            "stem-height": args[11], "stem-width": args[12],
            "stem-root": args[13], "stem-surface": args[14],
            "stem-color": args[15],
            "veil-type": args[16], "veil-color": args[17],
            "has-ring": args[18], "ring-type": args[19],
            "spore-color": args[20],
            "habitat": args[21], "season": args[22]
        }

    @classmethod
    def generate_from_source(cls, name, info_text, sizes, habitat, season, is_edible):
        """
        Parameters
        ----------
        name: str
        mushroom name
        info_text: str
        description text containing most of the attributes extracted with data_col_match.py
        sizes: list
        metrical attributes cap-diameter, stem-height and stem-width
        habitat: str
        attribute outside of info_text
        season: str
        attribute outside of info_text
        is_edible: str
        binary class "p" poisonous or "e" edible

        Returns
        -------
        FunghiType
        alternative constructor
        """

        family = ""
        color_dict = text_attr_match.get_color_category_dict(info_text)
        has_ring = text_attr_match.get_has_feature(info_text, ["ring"])
        funghi_type_attributes_list = [family,
                                       name,
                                       is_edible,
                                       [sizes[0], sizes[1]],  # cap_diameter
                                       text_attr_match.get_category_attributes_list(info_text, ["cap"],
                                                                                    data_cat.cap_shape_key_words_dict),
                                       text_attr_match.get_category_attributes_list(info_text, ["cap"],
                                                                                    data_cat.cap_surface_key_words_dict),
                                       color_dict["cap"],  # cap_color
                                       text_attr_match.get_has_feature(info_text, ["bruis", "bleed"]),
                                       text_attr_match.get_category_attributes_list(info_text, ["gill"], data_cat.gill_attachment_key_words_dict),
                                       text_attr_match.get_category_attributes_list(info_text, ["gill"],
                                                                                    data_cat.gill_spacing_key_words_dict),
                                       color_dict["gill"],  # gill_color
                                       [sizes[2], sizes[3]],  # stem_height
                                       [sizes[4], sizes[5]],  # stem_width
                                       text_attr_match.get_category_attributes_list(info_text, ["stem"],
                                                                                    data_cat.stem_root_key_words_dict),
                                       text_attr_match.get_category_attributes_list(info_text, ["stem"],
                                                                                    data_cat.stem_surface_key_words_dict),
                                       color_dict["stem"],  # stem_color
                                       text_attr_match.get_category_attributes_list(info_text, ["veil"],
                                                                                    data_cat.veil_type_key_words_dict),
                                       color_dict["veil"],  # veil_color
                                       has_ring,
                                       text_attr_match.get_category_attributes_list(info_text, ["ring"],
                                                                                    data_cat.stem_surface_key_words_dict)
                   if has_ring == 't' else ['f'],
                                       # color_dict["ring"] if has_ring == 't' else ['f'],  # ring_color
                                       color_dict["spore"],  # spore_color
                                       text_attr_match.get_attributes_in_sentence_list(habitat,
                                                                                       data_cat.habitat_key_words_dict),
                                       text_attr_match.get_attributes_in_sentence_list(season,
                                                                                       data_cat.season_categories_dict)
                                       ]
        return cls(*funghi_type_attributes_list)


def write_to_csv(file_name, funghi_type_dict):
    """
    Parameters
    ----------
    file_name: str
    name of the written csv file
    funghi_type_dict: dict
    mushroom species each representing one line in the csv file

    Funtionality
    ------------
    writes each entry in funghi_type_dict as one line of a csv file with name file_name
    """

    file = open(file_name, "w")
    file.write(data_cat.PRIMARY_DATASET_HEADER + "\n")
    for funghi in funghi_type_dict:
        funghi_string = (";" + str(funghi) + ";" +
                         str(funghi_type_dict[funghi].is_edible))
        for category_key in funghi_type_dict[funghi].categories_dict:
            category_val = funghi_type_dict[funghi].categories_dict[category_key]
            if category_val:
                funghi_string += ";" + str(category_val).replace('\'', '')
            else:
                funghi_string += ";"
        file.write(funghi_string + "\n")


def get_html_files(directory_str):
    """
    Parameters
    ----------
    directory_str: str
    file path of the directory containing the html files representing the source book

    Return
    ------------
    list of files
    the html files in the directory directory_str
    """
    html_files = []
    for file_name in os.listdir(directory_str):
        html_files.append(open(directory_str + "/" + file_name))
    return html_files


def get_html_lines(html_files):
    """
    Parameters
    ----------
    html_files: list of files
    list of html files created by get_html_files()

    Return
    ------------
    list of strs
    all lines from html_files in one list
    """
    html_lines = []
    for file in html_files:
        lines = file.readlines()
        html_lines = html_lines + lines
    return html_lines


def remove_tags(html_str):
    """
    Parameters
    ----------
    html_str: str
    line of html code

    Return
    ------------
    str
    line only containing the text between html tags
    """
    while html_str.find("<") != -1:
        start_ind = html_str.find("<", 0)
        end_ind = html_str.find(">", start_ind)
        html_str = html_str[0 : start_ind : ] + html_str[end_ind + 1 : :]
    return html_str.replace("\n", "")


def get_funghi_book_entry_dict_from_html(html_lines):
    """
    Parameters
    ----------
    html_lines: list of strs
    list of html lines created by get_html_lines()

    Return
    ------------
    dict {str: list of strs}
    each entry contains one mushroom name and all corresponding html lines

    Excample
    ------------
    {'Fly Agaric': [all html lines up to the next name], 'Panther cap': ...}
    """
    funghi_dict = {}
    funghi_name = ""
    entry_lines = []
    for i in range(0, len(html_lines)):
        if "class=\"chapterHeadA" in html_lines[i] or "class=\"chapterheada" in html_lines[i]:
            if funghi_name:
                funghi_dict[funghi_name] = entry_lines
            entry_lines = []
            funghi_name = remove_tags(html_lines[i])
        else:
            entry_lines.append(html_lines[i])
    funghi_dict[funghi_name] = entry_lines
    return funghi_dict


def get_funghi_type_dict(funghi_dict):
    """
    Parameters
    ----------
    funghi_dict: dict {str: list of strs}
    is the name: html lines dict created by get_funghi_book_entry_dict_from_html()

    Return
    ------------
    dict {str: FunghiType}
    each entry contains a mushroom name and the corresponding FunghiType created with generate_funghi()
    """
    funghis = {}
    for funghi_name in funghi_dict:
        funghis[funghi_name] = generate_funghi(funghi_dict, funghi_name)
    return funghis


def generate_funghi(funghi_dict, funghi_name):
    """
    Parameters
    ----------
    funghi_dict: dict {str: list of strs}
    is the name: html lines dict created by get_funghi_book_entry_dict_from_html()
    funghi_name: str
    name of the mushroom, key in funghi_dict

    Return
    ------------
    FunghiType
    goes through the html lines stored for the dict entry, filters the relevant parts
    and creates an FunghiType with FunghiType.generate_from_source()
    """
    html_lines = funghi_dict[funghi_name]
    info_text = ""
    sizes = []
    habitat = ""
    season = ""
    is_edible = "p"
    for line in html_lines:
        if "class=\"paraNoIndent\"" in line:
            if not info_text:
                info_text = text_into_sentences(remove_tags(line))
            elif "SIZE" in line:
                sizes = [float(s) for s in remove_tags(line).replace("-", " ").split() if s.isdigit()]
                if len(sizes) < 6:
                    for i in range(len(sizes), 6):
                        sizes.append(0)
            elif "HABITAT" in line:
                habitat = remove_tags(line).replace("HABITAT ", "")
            elif "SEASON" in line:
                season = remove_tags(line).replace("SEASON ", "")
            elif "EDIBLE" in remove_tags(line) and "INEDIBLE" not in remove_tags(line):
                is_edible = "e"

    return FunghiType.generate_from_source(funghi_name, info_text, sizes, habitat, season, is_edible)


def text_into_sentences(text):
    """
    Parameters
    ----------
    text: str
    text containing sentences with punctuation

    Return
    ------------
    list of strs
    each entry corresponds to one sentence in text
    """

    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    return text.split("<stop>")


def get_funghi_type_dict_from_csv(file_path, start_entry, end_entry, **kwargs):
    """
    Parameters
    ----------
    file_path: str
    path of the csv file to be read
    start_entry: int
    line in csv file to start from inclusively (+ 1 to skip header)
    end_entry: int
    line in csv file to end on exclusively (+ 1 to skip header)

    **kwargs
    sep: str, default = ';'
    seperator used in the csv file

    Return
    ------------
    dict {str: FunghiType}
    each entry contains a mushroom name and the corresponding FunghiType created with the constructor
    """

    if 'sep' not in kwargs:
        kwargs['sep'] = ';'
    file = open(file_path)
    lines = file.readlines()
    funghi_type_dict = {}
    if len(lines) < end_entry:
        end_entry = len(lines) - 2
    for i in range(1 + start_entry, 1 + end_entry):
        attributes = lines[i].split(kwargs['sep'])
        for i in range(3, len(attributes)):
            attributes[i] = get_list_from_str(attributes[i])
        funghi_type_dict[attributes[1]] = FunghiType(*attributes)
    return funghi_type_dict


def get_list_from_str(text):
    """
    helper function for get_funghi_type_dict_from_csv()

    Parameters
    ----------
    text: str
    text containing an attribute value in Python list format

    Return
    ------------
    list of strs or list of floats
    list of the values as direct translation from the text
    nominal attributes as strs, metrical attributes as floats

    Example
    -------------
    text = "['x', 'f']" -> return ['x', 'f']
    text = "[10.0, 20.0]" -> return [10.0, 20.0]
    """

    remove_strs = ['[', ']', ' ', '\n']
    for remove_str in remove_strs:
        text = text.replace(remove_str, '')
    if ',' in text:
        result_list = text.split(',')
    else:
        result_list = [text]
    # if elements are not numbers returns as str, otherwise converts to float
    return result_list if not result_list[0].isdigit() else [float(n) for n in result_list]


if __name__ == "__main__":
    """
    WARNING: 
    Running this module overwrites the following files in data:
        - primary_data_generated.csv
    
    Running this module results in the html files in data/mushrooms_and_toadstools being read and
    a CSV primary_data_generated.csv to be created based on the these html files (the files may be
    extracted from the EPUB version of the book "Mushrooms & Toadstools" by Patrick Harding)
    """

    html_files = get_html_files(data_cat.FILE_PATH_BOOK_HTML)
    html_lines = get_html_lines(html_files)
    funghi_dict = get_funghi_book_entry_dict_from_html(html_lines)

    funghi_type_dict = get_funghi_type_dict(funghi_dict)
    write_to_csv(data_cat.FILE_PATH_PRIMARY_GENERATED, funghi_type_dict)
