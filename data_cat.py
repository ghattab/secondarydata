"""
This module contains constants used by the other modules:
file paths, csv headers as well as dicts and lists helping in reading out the source book
"""

# path of the book version HTML files directory
FILE_PATH_BOOK_HTML = "data/mushrooms_and_toadstools"
# file path for original data from 1987
FILE_PATH_1987 = "data/1987_data.csv"
FILE_PATH_1987_NO_MISS = "data/1987_data_no_miss.csv"
# file paths for primary data
FILE_PATH_PRIMARY_GENERATED = "data/primary_data_generated.csv"
FILE_PATH_PRIMARY_EDITED = "data/primary_data_edited.csv"
# file paths for secondary data
FILE_PATH_SECONDARY_GENERATED = "data/secondary_data_generated.csv"
FILE_PATH_SECONDARY_SHUFFLED = "data/secondary_data_shuffled.csv"
FILE_PATH_SECONDARY_NO_MISS = "data/secondary_data_no_miss.csv"
# file paths of secondary and original dataset encoded and matched columns
FILE_PATH_SECONDARY_MATCHED = "data/secondary_data_encoded_matched.csv"
FILE_PATH_1987_MATCHED = "data/1987_data_encoded_matched.csv"
# file paths used to generate the column-matched datasets
FILE_PATH_COLUMN_MATCHING = "data/data_columns_encoded_matching.csv"
FILE_PATH_COLUMN_MATCHING_EDITED = "data/data_columns_encoded_matching_edited.csv"


## dataset variables and possible values for primary dataset generation ##
PRIMARY_DATASET_HEADER = """family;name;class;cap-diameter;cap-shape;cap-surface;cap-color;does-bruise-or-bleed;gill-attachment;gill-spacing;gill-color;stem-height;stem-width;stem-root;stem-surface;stem-color;veil-type;veil-color;has-ring;ring-type;spore-print-color;habitat;season"""
DATASET_HEADER_MIN_MAX = """name;class;cap-diameter-min;cap-diameter-max;cap-shape;cap-surface;cap-color;does-bruis-or-bleed;gill-attachment;gill-spacing;gill-color;stem-height-min;stem-height-max;stem-width-min;stem-width-max;stem-root;stem-surface;stem-color;veil-type;veil-color;has-ring;ring-type;spore-color;habitat;season"""

features_list = ["cap", "gill", "stem", "veil", "ring", "spore"]

categories_secondary_list = ["class", "cap-diameter", "cap-shape", "cap-surface", "cap-color",
    "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color", "stem-height",
    "stem-width", "stem-root", "stem-surface", "stem-color", "veil-type", "veil-color", "has-ring",
    "ring-type", "spore-color", "habitat", "season"]

categories_original_list = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
    "population", "habitat"]

metrical_categories_list = ["cap-diameter", "stem-height", "stem-width"]

# category 4, 8, 13, 15, 18
color_categories_dict = {'brown': 'n', 'buff': 'b', 'gray': 'g', 'green': 'r', 'pink': 'p', 'purple': 'u', 'red': 'e',
                         'white': 'w', 'yellow': 'y', 'blue': 'l', 'orange': 'o', 'black': 'k'}

# category 2
cap_shape_categories_dict = {'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', 'sunken': 's', 'spherical': 'p',
                             'others': 'o'}

cap_shape_key_words_dict = {'spher': 'p', 'ball': 'p', 'egg': 'p', 'bell': 'b', 'conical': 'c',
                            'convex': 'x', 'round': 'x', 'flat': 'f', 'sunken': 's', 'depress': 's'}

# category 3
cap_surface_categories_dict = {'fibrous': 'i', 'grooves': 'g', 'scaly': 'y', 'smooth': 's', 'shiny': 'h',
                               'leathery': 'l', 'silky': 'k', 'sticky': 't', 'wrinkled': 'w', 'fleshy': 'e'}

cap_surface_key_words_dict = {'dry': 'd', 'fibr': 'i', 'groov': 'g', 'furrow': 'g', 'striat' : 'g',
                              'shin': 'h', 'leath': 'l', 'scal': 'y', 'granul': 'y',  'smooth': 's', 'greas': 's',
                              'sticky': 't', 'slim': 't', 'wrink': 'w', 'lined': 'w'}

# category 6
gill_attachment_categories_dict = {'adnate': 'a', 'adnexed': 'x', 'decurrent': 'd', 'free': 'e', 'sinuate': 's',
                                   'pores': 'p', 'none': 'f'}

gill_attachment_key_words_dict = {'attach': 'a',  'adnat': 'a', 'adnex': 'x',  'descend': 'd',
                                  'decur': 'd', 'free': 'e', 'sinuat': 's', 'pore': 'p'}

# category 7
gill_spacing_categories_dict = {'close': 'c', 'distant': 'd', 'none': 'f'}

gill_spacing_key_words_dict = {'clos': 'c', 'crowd': 'c', 'distant': 'd', 'wide': 'd'}

# category 11
stem_root_categories_dict = {'bulbous': 'b', 'swollen': 's', 'club': 'c', 'cup': 'u', 'equal': 'e', 'rhizomorphs': 'z',
                             'rooted': 'r'}

stem_root_key_words_dict = {'bulb': 'b', 'club': 'c', 'cup': 'u', 'equal': 'e', 'rhizo': 'z', 'root': 'r'}

# category 12
stem_surface_categories_dict = cap_surface_categories_dict

stem_surface_key_words_dict = cap_surface_key_words_dict

# category 14
veil_type_categories_dict = {'partial': 'p', 'universal': 'u'}

veil_type_key_words_dict = {'part': 'p', 'univ': 'u', 'entir': 'u'}

# category 17
ring_type_categories_dict = {'cobwebby': 'c', 'evanescent': 'e', 'flaring': 'r', 'grooved': 'g'}

ring_type_key_words_dict = {'cobweb': 'c', 'evanescent': 'e', 'transient': 'e', 'flar': 'r', 'shaggy': 'r',
                            'groov': 'g', 'striat': 'g', 'larg': 'l', 'huge': 'l', 'pend': 'p', 'hang': 'p',
                            'sheath': 's', 'zone': 'z'}

# category 18
habitat_categories_dict = {'grasses': 'g', 'leaves': 'l', 'meadows': 'm', 'paths': 'p',
                           'heaths': 'h', 'urban': 'u', 'waste': 'w', 'woods': 'd'}

habitat_key_words_dict = {'grass': 'g', 'leave': 'l', 'meadow': 'm', 'pasture': 'm', 'path': 'p',
                          'heath': 'h', 'urban': 'u', 'waste': 'w', 'wood': 'd', 'forest': 'd'}

# category 18
season_categories_dict = {'spring': 's', 'summer': 'u', 'autumn': 'a', 'winter': 'w'}





