from "data-cat" import *

# general method
def get_category_attributes_list(sentences, category_key_word_list, attributes_key_words_dict):
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
    color_categories_dict = dataset_categories.color_categories_dict
    features = dataset_categories.features_list
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
    for feat in features:
        if feat in sentence:
            return True
    return False


def get_word_distance(sentence, feature, attribute):
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


# for boolean features has_bruises and has_ring
def get_has_feature(sentences, features):
    for sentence in sentences:
        sentence = sentence.lower()
        for feature in features:
            if feature in sentence:
                return ['t']
    return ['f']


# used for single line information as in habitat and season
def get_attributes_in_sentence_list(sentence, attributes_dict):
    sentence = sentence.lower()
    attributes_in_sentence_list = []
    for attribute_key in attributes_dict:
        if attribute_key in sentence:
            attribute_val = attributes_dict[attribute_key]
            if attribute_val not in attributes_in_sentence_list:
                attributes_in_sentence_list.append(attribute_val)
    return attributes_in_sentence_list


def validate_attribute_in_sentence(sentence, attribute):
    if sentence.find(attribute) == 0:
        return True
    return sentence[sentence.find(attribute.lower()) - 1] in valid_pre_signs()


def valid_pre_signs():
    return ["-", " "]


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
