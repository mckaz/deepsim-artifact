#!/usr/bin/env python
import json
from collections import Counter
import numpy as np
import numpy.random as rng


import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
import os

## label to use for "other" type
OTHER_TYPE = "#other#"

## Randomly generated set of programs that comprises about 5% of dataset.
TEST_SET_PROGS = ['httparty', 'octokit.rb', 'meta-tags', 'mustache', 'meta-tags','reality', 'twitter_ebooks', 'vagrant-aws', 'method_source','faraday', 'wordpress-exploit-framework', 'shoes4', 'mongoid','md2key', 'childprocess', 'psd.rb', 'engine', 'rspec-rails','mongo', 'bh', 'her', 'newrelic_rpm', 'slack-ruby-bot', 'reality','ci', 'active_shipping', 'artoo', 'byebug', 'CompassApp','hamster', 'factory_girl', 'redis-rb', 'factory_bot', 'by_star','airbrake', 'seedbank', 'fog-xml', 'sensu', 'dpl','metasploit-framework']

def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

## Make necessary dirs
ensure_dir('tokenizers')
ensure_dir('models')
ensure_dir('labels')

def create_names_dataset(file_name,test_data=False):
    data_file = json.load(open(file_name, 'r'))
    # dataset will be a list of dicts of structure {'input': 'IDENTIFER_NAME', 'output': 'TYPE', 'program': 'PROGRAM_NAME'}
    dataset = []
    ## dict mapping program names to a set of types observed in that program
    prog_type_dict = {}
    if test_data:
        prog_set = [x for x in data_file.keys() if x in TEST_SET_PROGS]
    else:
        prog_set = [x for x in data_file.keys() if x not in TEST_SET_PROGS]
    #for program,prog_vals in data_file.items():
    for program in prog_set:
        for class_key, class_val in data_file[program].items():
            for method, method_data in class_val.items():
                if method_data.get('return', None) and method_data['return'].get('type', None):
                    ## if we have a return type for this method, use the method's name as input,
                    ## and add the data to the dataset
                    ret_type = method_data['return']['type'].lower()
                    t = {'input': method, 'output': ret_type, 'program': program}
                    dataset.append(t)
                    prog_type_dict.setdefault(program, set()).add(ret_type)
                    for param_name,param_hash in method_data['params'].items():
                        ## for each parameter in the method data, add its data to the dataset
                        if param_hash.get('type', None):
                            param_type = param_hash['type'].lower()
                            t = {'input': param_name, 'output': param_type, 'program': program}
                            dataset.append(t)
                            prog_type_dict.setdefault(program, set()).add(param_type)

    ## below lines of code compute the count of how many programs each type appears in
    ## currently we don't use this count for anything, but it may be useful in the future
    ## for determining which types we want to predict.                            
    return [dataset, prog_type_dict]

def create_comments_dataset(file_name, test_data=False):
    data_file = json.load(open(file_name, 'r'))
    # dataset will be a list of dicts of structure {'input': 'COMMENT', 'output': 'TYPE', 'program': 'PROGRAM_NAME'}    
    dataset = []
    max_doc_size = 500 ## TEMPORARY TO CAP COMMENT SIZE
    prog_type_dict = {} ## dict to keep track of which programs feature which types
    if test_data:
        prog_set = [x for x in data_file.keys() if x in TEST_SET_PROGS]
    else:
        prog_set = [x for x in data_file.keys() if x not in TEST_SET_PROGS]
    #for program,prog_vals in data_file.items():
    for program in prog_set:
        for class_key, class_val in data_file[program].items():
            for method, method_data in class_val.items():
                if method_data.get('return', None) and method_data['return'].get('type', None) and method_data['return'].get('doc', None):
                    ret_type = method_data['return']['type'].lower()
                    ret_doc = method_data['return']['doc']
                    t = {'input': ret_doc[:max_doc_size], 'output': ret_type, 'program': program}
                    dataset.append(t)
                    prog_type_dict.setdefault(program, set()).add(ret_type)
                for param_name,param_hash in method_data['params'].items():
                    if param_hash.get('type', None) and param_hash.get('doc', None):
                        param_type = param_hash['type'].lower()
                        param_doc = param_hash['doc'].lower()
                        t = {'input': param_doc[:max_doc_size], 'output': param_type, 'program': program}
                        dataset.append(t)
                        prog_type_dict.setdefault(program, set()).add(param_type)
    return [dataset, prog_type_dict]


def create_nc_dataset(file_name, delimiter, test_data=False):
    data_file = json.load(open(file_name, 'r'))
    # dataset will be a list of dicts of structure {'input': 'COMMENT', 'output': 'TYPE', 'program': 'PROGRAM_NAME' }    
    dataset = []
    max_doc_size = 500 ## TEMPORARY TO CAP COMMENT SIZE
    prog_type_dict = {} ## dict to keep track of which programs feature which types
    if test_data:
        prog_set = [x for x in data_file.keys() if x in TEST_SET_PROGS]
    else:
        prog_set = [x for x in data_file.keys() if x not in TEST_SET_PROGS]
    #for program,prog_vals in data_file.items():
    for program in prog_set:
        for class_key, class_val in data_file[program].items():
            for method, method_data in class_val.items():
                if method_data.get('return', None) and method_data['return'].get('type', None) and method_data['return'].get('doc', None):
                    ret_type = method_data['return']['type'].lower()
                    ret_doc = method_data['return']['doc']
                    t = {'input': method + delimiter + ret_doc[:max_doc_size].replace(delimiter, ""), 'output': ret_type, 'program': program }
                    dataset.append(t)
                    prog_type_dict.setdefault(program, set()).add(ret_type)
                for param_name,param_hash in method_data['params'].items():
                    if param_hash.get('type', None) and param_hash.get('doc', None):
                        param_type = param_hash['type'].lower()
                        param_doc = param_hash['doc'].lower()
                        t = {'input': param_name + delimiter + param_doc[:max_doc_size].replace(delimiter, ""), 'output': param_type, 'program': program}
                        dataset.append(t)
                        prog_type_dict.setdefault(program, set()).add(param_type)
    return [dataset, prog_type_dict]
    


## below lines of code compute the count of how many programs each type appears in
## currently we don't use this count for anything, but it may be useful in the future
## for determining which types we want to predict.
'''                    
type_progcount = {}
for typ in Counter([x['output'] for x in dataset]).keys():
    prog_count = 0
    for prog_typs in prog_type_dict.values():
        if typ in prog_typs:
            prog_count += 1
    type_progcount[typ] = prog_count

print(sorted(type_progcount.items(), key=itemgetter(1))[::-1])
'''

## print most common 100 types in the dataset
#print(Counter([x['output'] for x in dataset]).most_common(100))


## below is useful for visualizing the distribution of type counts in the dataset
'''
counter = Counter([x['output'] for x in dataset])
type_counts = [counter[x] for x in counter]
plt.hist(type_counts, bins=[10, 50, 100, 300, 500 ])#1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000])
plt.title("Distribution of Types")
plt.show()
'''

def choose_top_labels(dataset, prog_type_dict, label_choice, label_num, min_prognum_labels):
    if (label_choice == "TOP"):
        return [x[0] for x in Counter([x['output'] for x in dataset]).most_common(label_num)]
    elif (label_choice == "PROG"):
        ## first count how many programs each type appears in
        type_progcount = {}
        for typ in Counter([x['output'] for x in dataset]).keys():
            prog_count = 0
            for prog_typs in prog_type_dict.values():
                if typ in prog_typs:
                    prog_count += 1
            type_progcount[typ] = prog_count
        return [key for key,value in type_progcount.items() if value >= min_prognum_labels]
    elif (label_choice == "ALL"):
        [x['output'] for x in dataset]
    else:
        print('unexpected value of label_choice: {}'.format(label_choice))
        raise ValueError

def create_labels(dataset, prog_type_dict, label_choice, use_other, label_num=100, min_prognum_labels=5):
    ## create label indices, and dict to map back from index to label name
    top_labels = choose_top_labels(dataset, prog_type_dict, label_choice, label_num, min_prognum_labels)
    label_to_idx = {x: top_labels.index(x) for x in top_labels}
    idx_to_label = {v: k for k,v in label_to_idx.items()}
    if use_other:
        label_to_idx[OTHER_TYPE] = len(top_labels) + 1
        idx_to_label[len(top_labels)] = OTHER_TYPE

    ## print test below
    #print("string to idx: {}".format(label_to_idx['hash']))
    #print("idx to string: {}".format(idx_to_label[3]))
    return [label_to_idx, idx_to_label]

    
def create_tokenizer(dataset):
    ## create char tokenizer / "fit" to the characters in the inputs
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    lang_tokenizer.fit_on_texts([x['input'] for x in dataset])
    return lang_tokenizer



## test out tokenizer
'''
in_string = dataset[0]['input']
token_string = lang_tokenizer.texts_to_sequences(in_string)
tok_to_string = lang_tokenizer.sequences_to_texts(token_string)

## MULTIPLE INPUTS
#tok_to_string = lang_tokenizer.texts_to_sequences([x['input'] for x in dataset[:5]]) 

print("Input: {}".format(in_string))
print("Tokenized: {}".format(token_string))
print("Output from Tokens: {}".format(tok_to_string))
'''

def split_train_dev(dataset):
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])
    return [train_dataset, dev_dataset]

def prepare_data(dataset, lang_tokenizer, label_to_idx, use_other, test=False):
    ## prepare input/output data
    input_data = []
    output_data = []

    for sample in dataset:
        ## only use data which falls in top N types
        ## TODO: do we want to have an "other" type?
        if label_to_idx.get(sample['output'], -1) > -1:
            input_data.append(lang_tokenizer.texts_to_sequences(sample['input']))
            output_data.append(label_to_idx[sample['output']])
        elif use_other:
            input_data.append(lang_tokenizer.texts_to_sequences(sample['input']))
            output_data.append(label_to_idx[OTHER_TYPE])

    ## pad sequences so they're all same length
    in_data = tf.keras.preprocessing.sequence.pad_sequences(input_data).squeeze()


    if test:
        ## for testing, keep in/out data separate
        return [in_data, output_data]
    else:
        ## otherwise, slice in/out data together to create dataset
        return tf.data.Dataset.from_tensor_slices((in_data, output_data))

def save_labels(d, name):
    with open('labels/'+ name + '.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def get_twin_data(dataset, data_size, lang_tokenizer, label_to_idx, use_other):
    ## prepare input/output data
    input_data = []
    output_data = []

    for sample in dataset:
        ## only use data which falls in top N types
        ## TODO: do we want to have an "other" type?
        if label_to_idx.get(sample['output'], -1) > -1:
            input_seq = lang_tokenizer.texts_to_sequences(sample['input'])
            input_data.append(input_seq)
            output_idx = label_to_idx[sample['output']]
            output_data.append(output_idx)
            #label_input_map.setdefault(output_idx, []).append(input_seq)
        elif use_other:
            input_seq = lang_tokenizer.texts_to_sequences(sample['input'])
            input_data.append(input_seq)
            output_idx = label_to_idx[OTHER_TYPE]
            output_data.append(output_idx)
            #label_input_map.setdefault(output_idx, []).append(input_seq)

    ## pad sequences so they're all same length
    in_data = tf.keras.preprocessing.sequence.pad_sequences(input_data).squeeze()

    ## create mapping from each label to set of inputs of that label
    label_input_map = {}
    for i, val in enumerate(output_data):
        label_input_map.setdefault(val, []).append(in_data[i])

    ## delete any labels for which pair doesn't exist
    to_delete = [key for key in label_input_map if len(label_input_map[key]) < 2]
    for key in to_delete: del label_input_map[key]

    num_data, in_dim = in_data.shape

    ## pairs is list of two arrays. for all i, pairs[0][i] and pairs[1][i] constitute a single datapoint.
    pairs = [np.zeros((data_size, in_dim)) for i in range(2)]

    ## targets is array of labels for data in `pairs`.
    ## Labels are similarity scores, 0 == no similarity, 1 == high similarity.
    ## Make second half of targets all 1s, and we will make pairs match this configuration.
    targets=np.zeros((data_size,))
    targets[data_size//2:] = 1

    for i in range(data_size):
        ## pick first data point
        chosen_type1 = rng.choice(list(label_input_map.keys()))
        #chosen_in1 = label_input_map[chosen_type1].pop(rng.randint(len(label_input_map[chosen_type1])))
        idx_in1 = rng.randint(len(label_input_map[chosen_type1]))
        chosen_in1 = label_input_map[chosen_type1][idx_in1]
        
        pairs[0][i] = chosen_in1

        ## for first half of data, choose dissimilar points. second half choose similar points.
        if i >= data_size // 2:
            chosen_type2 = chosen_type1
            idx_in2 = rng.choice([x for x in range(len(label_input_map[chosen_type2])) if x!=idx_in1])
            chosen_in2 = label_input_map[chosen_type2][idx_in2]
        else:
            chosen_type2 = rng.choice([x for x in label_input_map.keys() if chosen_type1 != x])
            #chosen_in2 = label_input_map[chosen_type2].pop(rng.randint(len(label_input_map[chosen_type2])))
            chosen_in2 = label_input_map[chosen_type2][rng.randint(len(label_input_map[chosen_type2]))]

        pairs[1][i] = chosen_in2


        ## delete keys from dict if number of types is too low
        if len(label_input_map[chosen_type1]) < 2:
            del label_input_map[chosen_type1]
        if (chosen_type2 != chosen_type1) and len(label_input_map[chosen_type2]) < 2:
            del label_input_map[chosen_type2]

    return pairs, targets





## Similar to above method, but here pairs are created from
## types belonging to the same program.
def get_prog_twin_data(dataset, data_size, lang_tokenizer, label_to_idx, use_other):
    ## prepare input/output data
    input_data = []
    output_data = []
    program_labels = []

    for sample in dataset:
        ## only use data which falls in top N types
        ## TODO: do we want to have an "other" type?
        if label_to_idx.get(sample['output'], -1) > -1:
            input_seq = lang_tokenizer.texts_to_sequences(sample['input'])
            input_data.append(input_seq)
            output_idx = label_to_idx[sample['output']]
            output_data.append(output_idx)
            program_labels.append(sample['program'])
            #label_input_map.setdefault(output_idx, []).append(input_seq)
        elif use_other:
            input_seq = lang_tokenizer.texts_to_sequences(sample['input'])
            input_data.append(input_seq)
            output_idx = label_to_idx[OTHER_TYPE]
            output_data.append(output_idx)
            program_labels.append(sample['program'])
            #label_input_map.setdefault(output_idx, []).append(input_seq)

    ## pad sequences so they're all same length
    in_data = tf.keras.preprocessing.sequence.pad_sequences(input_data).squeeze()

    ## create mapping from each program to each label to set of inputs of that label for that program
    prog_label_input_map = {}
    for i, prog in enumerate(program_labels):
        prog_label_input_map.setdefault(prog, {}).setdefault(output_data[i], []).append(in_data[i])


    ## delete types for which pair doesn't exist,
    ## and delete programs which don't have data for
    ## at least two different types
    progs_to_delete = []
    for prog in prog_label_input_map:
        types_to_delete = []
        for typ in prog_label_input_map[prog]:
            if len(prog_label_input_map[prog][typ]) < 2:
                types_to_delete.append(typ)
                #types_to_delete.append([prog, typ])
                #del prog_label_input_map[prog][typ]
        for typ in types_to_delete:
            del prog_label_input_map[prog][typ]
        if len(prog_label_input_map[prog]) < 2:
            progs_to_delete.append(prog)
            #del prog_label_input_map[prog]
        
    #for prog, typ in types_to_delete:
    #    del prog_label_input_map[prog][typ]
    for prog in progs_to_delete:
        del prog_label_input_map[prog]
            
    num_data, in_dim = in_data.shape

    ## pairs is list of two arrays. for all i, pairs[0][i] and pairs[1][i] constitute a single datapoint.
    pairs = [np.zeros((data_size, in_dim)) for i in range(2)]

    ## targets is array of labels for data in `pairs`.
    ## Labels are similarity scores, 0 == no similarity, 1 == high similarity.
    ## Make second half of targets all 1s, and we will make pairs match this configuration.
    targets=np.zeros((data_size,))
    targets[data_size//2:] = 1

    for i in range(data_size):
        ## pick first program
        prog = rng.choice(list(prog_label_input_map.keys()))
        
        ## pick first type
        chosen_type1 = rng.choice(list(prog_label_input_map[prog].keys()))

        ## pick first input
        in1_idx = rng.randint(len(prog_label_input_map[prog][chosen_type1]))
        chosen_in1 = prog_label_input_map[prog][chosen_type1][in1_idx]
                                       
        pairs[0][i] = chosen_in1

        ## for first half of data, choose dissimilar points. second half choose similar points.
        if i >= data_size // 2:
            chosen_type2 = chosen_type1
            in2_idx = rng.choice([x for x in range(len(prog_label_input_map[prog][chosen_type1])) if x!=in1_idx])
            chosen_in2 = prog_label_input_map[prog][chosen_type1][in2_idx]
        else:
            chosen_type2 = rng.choice([x for x in prog_label_input_map[prog].keys() if chosen_type1 != x])
            chosen_in2 = prog_label_input_map[prog][chosen_type2][rng.randint(len(prog_label_input_map[prog][chosen_type2]))]

        pairs[1][i] = chosen_in2

    return pairs, targets
