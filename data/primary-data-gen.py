import os

from "text-attr-match" import *
from "data-cat" import *


class Funghi_Type:

    def __init__(self, family, name, is_edible,
                 cap_diameter, cap_shape, cap_surface, cap_color, does_bruis_or_bleed,
                 gill_attachment, gill_spacing, gill_color,
                 stem_height, stem_width, stem_root, stem_surface, stem_color,
                 veil_type, veil_color,
                 has_ring, ring_type,
                 spore_color, habitat, season):
        self.family = family
        self.name = name
        self.is_edible = is_edible
        self.cap_diameter = cap_diameter
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.does_bruis_or_bleed = does_bruis_or_bleed
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_color = gill_color
        self.stem_height = stem_height
        self.stem_width = stem_width
        self.stem_root = stem_root
        self.stem_surface = stem_surface
        self.stem_color = stem_color
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.has_ring = has_ring
        self.ring_type = ring_type
        self.spore_color = spore_color
        self.habitat = habitat
        self.season = season
        self.categories_dict = {
            "cap-diameter": self.cap_diameter, "cap-shape": self.cap_shape,
            "cap-surface": self.cap_surface, "cap-color": self.cap_color,
            "has-bruises": self.does_bruis_or_bleed, "gill-attachment": self.gill_attachment,
            "gill-spacing": self.gill_spacing, "gill-color": self.gill_color,
            "stem-height": self.stem_height, "stem-width": self.stem_width,
            "stem-root": self.stem_root, "stem-surface": self.stem_surface,
            "stem-color": self.stem_color,
            "veil-type": self.veil_type, "veil-color": self.veil_color,
            "has-ring": self.has_ring, "ring-type": self.ring_type,
            "spore-color": self.spore_color,
            "habitat": self.habitat, "season": self.season
        }

    @classmethod
    def generate_from_source(cls, name, info_text, sizes, habitat, season, is_edible):
        family = ""
        color_dict = text_category_attribut_matching.get_color_category_dict(info_text)
        has_ring = text_category_attribut_matching.get_has_feature(info_text, ["ring"])
        return cls(family,
                   name,
                   is_edible,
                   [sizes[0], sizes[1]],  # cap_diameter
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["cap"],
                                                                                dataset_categories.cap_shape_key_words_dict),
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["cap"],
                                                                                dataset_categories.cap_surface_key_words_dict),
                   color_dict["cap"],  # cap_color
                   text_category_attribut_matching.get_has_feature(info_text, ["bruis", "bleed"]),
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["gill"], dataset_categories.gill_attachment_key_words_dict),
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["gill"],
                                                                                dataset_categories.gill_spacing_key_words_dict),
                   color_dict["gill"],  # gill_color
                   [sizes[2], sizes[3]],  # stem_height
                   [sizes[4], sizes[5]],  # stem_width
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["stem"],
                                                                                dataset_categories.stem_root_key_words_dict),
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["stem"],
                                                                                dataset_categories.stem_surface_key_words_dict),
                   color_dict["stem"],  # stem_color
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["veil"],
                                                                                dataset_categories.veil_type_key_words_dict),
                   color_dict["veil"],  # veil_color
                   has_ring,
                   text_category_attribut_matching.get_category_attributes_list(info_text, ["ring"],
                                                                                dataset_categories.stem_surface_key_words_dict)
                   if has_ring == 't' else ['f'],
                   # color_dict["ring"] if has_ring == 't' else ['f'],  # ring_color
                   color_dict["spore"],  # spore_color
                   text_category_attribut_matching.get_attributes_in_sentence_list(habitat,
                                                                                   dataset_categories.habitat_key_words_dict),
                   text_category_attribut_matching.get_attributes_in_sentence_list(season,
                                                                                   dataset_categories.season_categories_dict)
                   )


def write_to_csv(file_name, funghi_type_dict):
    file = open(file_name, "w")
    file.write(dataset_categories.PRIMARY_DATASET_HEADER + "\n")
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


def write_metricals_to_csv(file_name, funghi_type_dict):
    file = open(file_name, "w")
    header_str = "name;"
    for category in dataset_categories.metrical_categories_list:
        header_str += str(category) + "-min;" + str(category) + "-max;"
    file.write(header_str[:-1] + "\n")
    for funghi in funghi_type_dict:
        funghi_str = str(funghi)
        for category_key in dataset_categories.metrical_categories_list:
            category_val = funghi_type_dict[funghi].categories_dict[category_key]
            if len(category_val) == 1:
                mean = category_val[0]
                category_val[0] = mean - (mean / 4)
                category_val.append(mean + (mean / 4))
            funghi_str += ";" + str(category_val[0]) + ";" + str(category_val[1])
        file.write(funghi_str + "\n")


def get_html_files(directory_str):
    html_files = []
    for file_name in os.listdir(directory_str):
        html_files.append(open(directory_str + "/" + file_name))
    return html_files


def get_html_lines(html_files):
    html_lines = []
    for file in html_files:
        lines = file.readlines()
        html_lines = html_lines + lines
    return html_lines


def remove_tags(html_str):
    count = 0
    while html_str.find("<") != -1:
        start_ind = html_str.find("<", 0)
        end_ind = html_str.find(">", start_ind)
        html_str = html_str[0 : start_ind : ] + html_str[end_ind + 1 : :]
    return html_str.replace("\n", "")


# Takes list of strs and returns dict of str : list of strs, funghi name : all following lines without html-tags
def get_funghi_book_entry_dict_from_html(html_lines):
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
    funghis = {}
    for funghi_name in funghi_dict:
        funghis[funghi_name] = generate_funghi(funghi_dict, funghi_name)
    return funghis


def generate_funghi(funghi_dict, funghi_name):
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

    return Funghi_Type.generate_from_source(funghi_name, info_text, sizes, habitat, season, is_edible)


def text_into_sentences(text):
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    return text.split("<stop>")


# gets dict {funghi_name : funghi_type} from edited primary data
# from line 3 on, 1 line = 1 funghi (entry 0 = Fly Agaric, ...), start_entry inclusive, end_entry exclusive
def get_funghi_type_dict_from_csv(file_path, start_entry, end_entry, **kwargs):
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
        funghi_type_dict[attributes[1]] = Funghi_Type(
            attributes[0], attributes[1], attributes[2], attributes[3], attributes[4], attributes[5],
            attributes[6], attributes[7], attributes[8], attributes[9], attributes[10], attributes[11],
            attributes[12], attributes[13], attributes[14], attributes[15], attributes[16], attributes[17],
            attributes[18], attributes[19], attributes[20], attributes[21], attributes[22]
        )
    return funghi_type_dict


def get_list_from_str(text):
    remove_strs = ['[', ']', ' ', '\n']
    for remove_str in remove_strs:
        text = text.replace(remove_str, '')
    if ',' in text:
        result_list = text.split(',')
    else:
        result_list = [text]
    # if elements are not numbers returns as str, otherwise converts to int
    return result_list if not result_list[0].isdigit() else [float(n) for n in result_list]


if __name__ == "__main__":
    html_files = get_html_files(dataset_categories.FILE_PATH_BOOK_HTML)
    html_lines = get_html_lines(html_files)
    funghi_dict = get_funghi_book_entry_dict_from_html(html_lines)

    funghi_type_dict = get_funghi_type_dict(funghi_dict)
    write_to_csv(dataset_categories.FILE_PATH_PRIMARY_GENERATED, funghi_type_dict)
    funghi_type_edited_dict = get_funghi_type_dict_from_csv(dataset_categories.FILE_PATH_PRIMARY_EDITED, 0, 173)
    # write_metricals_to_csv("data/primary_data_metricals_edited.csv", funghi_type_edited_dict)
