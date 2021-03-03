import data_cat

# general method
def get_category_attributes_list(sentences, category_key_word_list, attributes_key_words_dict):
    """
    Parameters
    ----------
    sentences: list of strs,
    The info text of a mushroom species split into sentences.
    category_key_word_list: list of strs,
    Key words corresponding to a mushroom feature or feature (mostly from dataset_categories.feature_list)
    attributes_key_words_dict: dict of {str: str},
    Key word dict from data_cat.py corresponding to the mushroom feature in category_key_word_list

    Return
    ------
    var name = result_attributes_list: list of strs,
    List of mushroom feature attributes that are in the sentence as the mushroom feature.

    Example
    -------
    sentences[2] = "The entire young fruitbody is enclosed in a white veil which leaves fragments (which may wash off)
        on the shiny red, marginally grooved cap." (the other values of sentences do not contain "cap")
    category_key_word_list = ["cap"]
    attributes_key_words_dict = dataset_categories.cap_surface_key_words_dict

    return:
    ['g', 'h'] with 'g' = 'grooved', 'h' = 'shiny'
    """

    result_attributes_list = []
    for sentence in sentences:
        sentence = sentence.lower()
        for category_str in category_key_word_list:
            if category_str in sentence:
                for attributes_key in attributes_key_words_dict:
                    if attributes_key in sentence and validate_attribute_in_sentence(sentence, attributes_key):
                        result_attributes_list.append(attributes_key_words_dict[attributes_key])
    return result_attributes_list


def get_color_category_dict(sentences):
    """
    Parameters
    ----------
    sentences: list of strs,
    The info text of a mushroom species split into sentences.

    Return
    ------
    var name = feature_color_dict: dict of {str: list of strs},
    Keys are dataset_categories.features_list. Each color in the sentence is matched to the nearest key or feature
    and the list is saved in the dict. The colors are encoded as in dataset_categories.color_categories_dict.

    Example
    -------
    sentences[2] = "The entire young fruitbody is enclosed in a white veil which leaves fragments (which may wash off)
        on the shiny red, marginally grooved cap." (for simplicity only one sentence is considered)

    added to return or feature_color_dict:
    feature_color_dict['cap'] += 'e' (with 'e'='red')
    feature_color_dict['veil'] += 'w' (with 'w'='white')
    """

    color_categories_dict = data_cat.color_categories_dict
    features = data_cat.features_list
    feature_color_dict = {}
    for feature in features:
        feature_color_dict[feature] = []
    for sentence in sentences:
            sentence = sentence.lower()
            colors_in_sentence = []
            if is_feat_in_sentence(sentence, features):
                for color in color_categories_dict:
                    if color in sentence and validate_attribute_in_sentence(sentence, color):
                        colors_in_sentence.append(color)
                for color in colors_in_sentence:
                    feature = find_nearest_feature_to_attribute(sentence, features, color)
                    feature_color_dict[feature] += color_categories_dict[color]
    # feature_color_dict['gill'].append(feature_color_dict['pore'])
    # feature_color_dict.pop('pore')
    return feature_color_dict


def find_nearest_feature_to_attribute(sentence, features, attribute):
    """
    Parameters
    ----------
    sentence: str,
    One sentence from the info text of a mushroom species
    features: list of strs
    List of possible features as in dataset_categories.features_list
    attribute: str,
    Mushroom feature attribute that is in the sentence (e.g. 'red' for 'cap color').

    Return
    ------
    str,
    The feature in features that is closest to attribute in word steps.

    Example
    -------
    sentences[2] = "The entire young fruitbody is enclosed in a white veil which leaves fragments (which may wash off)
        on the shiny red, marginally grooved cap." (for simplicity only one sentence is considered)
    features = dataset_categories.features_list (relevant here: 'cap', 'veil')
    attribute = 'white'

    return:
    'veil' (since 'veil' is closer to 'white' than 'cap')
    """

    min_distance = float('inf')
    min_distance_index = 0
    for i in range(0, len(features)):
        if features[i] in sentence:
            word_distance = get_word_distance(sentence, features[i], attribute)
            if word_distance < min_distance:
                min_distance = word_distance
                min_distance_index = i
    return features[min_distance_index]


def is_feat_in_sentence(sentence, features):
    """
    Parameters
    ----------
    sentence: str,
    One sentence from the info text of a mushroom species
    features: list of strs
    List of possible features as in dataset_categories.features_list

    Return
    ------
    bool,
    True if sentence contains at least one feature from features and else False.
    """

    for feature in features:
        if feature in sentence:
            return True
    return False


def get_word_distance(sentence, feature, attribute):
    """
    Parameters
    ----------
    sentence: str,
    One sentence from the info text of a mushroom species
    feature: str
    One feature from dataset_categories.features_list
    attribute: str,
    Mushroom feature attribute that is in the sentence (e.g. 'red' for 'cap color').

    Return
    ------
    int,
    Word step distance (as a positive number)

    Example
    -------
    sentence = "The entire young fruitbody is enclosed in a white veil which leaves fragments (which may wash off)
        on the shiny red, marginally grooved cap."

    return examples:
    get_word_distance(sentence, 'cap', 'white') => 14
    get_word_distance(sentence, 'veil', 'white') => 1
    """

    sentence_as_words = sentence.split()
    feature_index = None
    attribute_index = None
    for i in range(0, len(sentence_as_words)):
        if(feature.lower() in sentence_as_words[i].lower()):
            feature_index = i
        if(attribute.lower() in sentence_as_words[i].lower()):
            attribute_index = i
        if(not (feature_index is None or attribute_index is None)):
            break
    return abs(feature_index - attribute_index)


def get_has_feature(sentences, features):
    """
    Parameters
    ----------
    sentences: list of strs,
    The info text of a mushroom species split into sentences.
    features: list of strs
    Key words ["bruis", "bleed"] for attribute does-bruise-or-bleed

    Return
    ------
    list of str,
    ['t'] if one of feature in features is in one of the sentences and else ['f'] (dataset encoded boolean)
    """

    for sentence in sentences:
        sentence = sentence.lower()
        for feature in features:
            if feature in sentence:
                return ['t']
    return ['f']


# used for single line information as in habitat and season
def get_attributes_in_sentence_list(line, attributes_dict):
    """
    Parameters
    ----------
    line: str,
    Separate info line for the attribute habitat or season.
    attributes_dict: dict of {str: str},
    dataset_categories.habitat_key_words_dict or dataset_categories.season_categories_dict

    Return
    ------
    var name = attributes_in_sentence_list: list of strs,
    For all of attributes_dict keys found in sentence the corresponding value (one letter encoding).

    Example
    -------
    line = "Late summer to early winter." (season)
    attributes_dict = dataset_categories.season_categories_dict

    return:
    ['u', 'w'] ('u'='summer', 'w'=winter', good example of a mistake since the implicit 'a'='autumn' is missing)
    """

    line = line.lower()
    attributes_in_sentence_list = []
    for attribute_key in attributes_dict:
        if attribute_key in line:
            attribute_val = attributes_dict[attribute_key]
            if attribute_val not in attributes_in_sentence_list:
                attributes_in_sentence_list.append(attribute_val)
    return attributes_in_sentence_list


def validate_attribute_in_sentence(sentence, attribute):
    """
    Parameters
    ----------
    sentence: str,
    One sentence from the info text of a mushroom species.
    attribute: str,
    Mushroom feature attribute that is in the sentence (e.g. 'red' for 'cap color').

    Return
    ------
    bool,
    True if the attribute is preceded by a space or a hyphen and False else (rules out parts of other words)

    Example
    -------
    The 'red' in 'covered' is ignored while the 'white' in 'off-white' is recognized.
    """

    valid_pre_signs = [" ", "-"]
    if sentence.find(attribute) == 0:
        return True
    return sentence[sentence.find(attribute.lower()) - 1] in valid_pre_signs