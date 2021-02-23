import pandas as pd
import random

from "primary-data-gen" import *
from "data-cat" import *
from "gen-corr-norm" import *

FILE_PATH_IN = "data/primary_data_edited_exclude_stemless.csv"
FILE_PATH_OUT1 = "data/secondary_data_generated.csv"
FILE_PATH_OUT2 = "data/secondary_data_generated_with_intervals.csv"


class Funghi_Entry:
    def __init__(self, *args):
        self.family = args[0]
        self.name = args[1]
        self.is_edible = args[2]
        self.categories = args[3:]


def write_to_csv(file_name, funghi_entry_list, use_intervals):
    file = open(file_name, "w")
    if not use_intervals:
        file.write(dataset_categories.PRIMARY_DATASET_HEADER.replace("family;name;", "") + "\n")
    else:
        file.write(dataset_categories.DATASET_HEADER_MIN_MAX.replace("name;", "") + "\n")
    for funghi_entry in funghi_entry_list:
        funghi_str = funghi_entry.is_edible
        for category in funghi_entry.categories:
            funghi_str += ";" + str(category)
        file.write(funghi_str + "\n")


def generate_funghi_entry_list(funghi_type_dict, number, use_intervals):
    funghi_entry_list = []
    for funghi_key in funghi_type_dict:
        funghi_type_categories_list = list(funghi_type_dict[funghi_key].categories_dict.values())
        funghi_class_list = [funghi_type_dict[funghi_key].family,
                             funghi_type_dict[funghi_key].name,
                             funghi_type_dict[funghi_key].is_edible]
        # generate normal distributions based on metrical attributes
        metrical_attributes_columnindex_dict = {0: 0, 8: 1, 9: 2}
        metrical_attributes = [funghi_type_categories_list[0],
                               funghi_type_categories_list[8],
                               funghi_type_categories_list[9]]
        # std = 3 for 99.7% of normals being in interval min-max
        # single values are interpreted as mean +- (mean/4)
        for attribute in metrical_attributes:
            # make safe all inputs are interpreted as metrical
            for i in range(0, len(attribute)):
                attribute[i] = float(attribute[i])
            if len(attribute) == 1:
                mean = attribute[0]
                attribute[0] = mean - (mean / 4)
                attribute.append(mean + (mean / 4))
        normal_values = generate_correlated_normals.get_correlated_normals_in_interval(
            500, metrical_attributes, 3)
        for entry_count in range(0, number):
            funghi_entry_attributes_list = [] + funghi_class_list
            for category_count in range(0, len(funghi_type_categories_list)):
                # nominal values
                if category_count not in metrical_attributes_columnindex_dict.keys():
                    funghi_entry_attributes_list \
                        .append(random.choice(funghi_type_categories_list[category_count]))
                # metrical values
                else:
                    # draw value from correlated gaussian dist
                    if not use_intervals:
                        funghi_entry_attributes_list.append(round(
                            normal_values[metrical_attributes_columnindex_dict[category_count]][entry_count], 2))
                    # put interval borders in seperate categories
                    else:
                        funghi_entry_attributes_list.append(funghi_type_categories_list[category_count][0])
                        funghi_entry_attributes_list.append(funghi_type_categories_list[category_count][1])
            funghi_entry_list.append(Funghi_Entry(*funghi_entry_attributes_list))
    return funghi_entry_list


def generate_funghi_entry_dummies_list(number):
    funghi_entry_list = []
    funghi_entry_attributes_list_e = ['Family', 'Dummy Shroom', 'e', 17.2, 'x', 'g', 'l', 't', 's', 'd', 'y', 10.5, 11.2,
                                      's', 's', 'n', 'u', 'w', 't', 'p', 'w', 'l', 'u']
    funghi_entry_attributes_list_p = ['Family', 'Deadly Dummy Shroom', 'p', 17.2, 'x', 'g', 'l', 't', 's', 'd', 'y', 10.5, 11.2,
                                      's', 's', 'n', 'u', 'w', 't', 'p', 'w', 'l', 'u']
    for i in range(0, number):
        if i % 2 == True:
            funghi_entry_list.append(Funghi_Entry(*funghi_entry_attributes_list_e))
        else:
            funghi_entry_list.append(Funghi_Entry(*funghi_entry_attributes_list_p))
    return funghi_entry_list


if __name__ == "__main__":
    csv = pd.read_csv(dataset_categories.FILE_PATH_PRIMARY_EDITED, sep=';', header=0, low_memory=False)

    edited_funghi_type_dict = primary_data_generation. \
        get_funghi_type_dict_from_csv(dataset_categories.FILE_PATH_PRIMARY_EDITED, 0, 173)

    funghi_entry_list = generate_funghi_entry_list(edited_funghi_type_dict, 353, False)
    funghi_entry_list_with_intervals = generate_funghi_entry_list(edited_funghi_type_dict, 500, True)
    write_to_csv(dataset_categories.FILE_PATH_SECONDARY_GENERATED, funghi_entry_list, False)
    write_to_csv(FILE_PATH_OUT2, funghi_entry_list_with_intervals, True)

    # read secondary data as pandas.DataFrame, shuffle and write to a new CSV
    data_secondary = pd.read_csv(dataset_categories.FILE_PATH_SECONDARY_GENERATED,
                                 sep=';', header=0, low_memory=False)
    data_secondary = data_secondary.sample(frac=1, random_state=1)
    data_secondary.to_csv(dataset_categories.FILE_PATH_SECONDARY_SHUFFLED, sep=';', index=False)
