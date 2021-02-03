"""
Author:         Jianyou (Andre) Wang
Date:           Sep 2020
"""


import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict, Counter
import os
import re
import random
import requests
import pickle
import heapq
import copy
from functools import reduce
import math
import multiprocessing as mp
import time
import pdb
import spacy
from gpt2.src.score import score_model
from gpt2.src.encoder import get_encoder
from .saved_objects.templates import  get_first_line_templates
from .saved_objects.Finer_POS import get_finer_pos_words


class utils:
    def __init__(self,
                 syllables_file='LimGen/saved_objects/cmudict-0.7b.txt',
                 postag_file='LimGen/saved_objects/postag_dict_all.p',
                 model_name='345M'):
        with open('LimGen/saved_objects/total_vocab.pickle',"rb") as f:
            self.total_vocab=pickle.load(f)
        with open('LimGen/saved_objects/clean_rhyming_dictionary.pickle',"rb") as f:
            self.rhyming_dictionary=pickle.load(f)
        self.api_url = 'https://api.datamuse.com/words'
        self.ps = nltk.stem.PorterStemmer()
        self.punct = re.compile(r'[^\w\s]')
        # punctuations
        self.punctuation={"second":True,"third":True,"fourth":True,"fifth":True}
        self.sentence_to_punctuation={"second":".","third":",","fourth":",","fifth":"."}
        # gpt2 model
        self.model_name = model_name
        self.enc = get_encoder(self.model_name)
        # load spacy word embeddings
        self.spacy_nlp = spacy.load("en_core_web_lg")
        # specify parameters for look ahead score
        self.word_embedding_alpha = 0.5
        self.word_embedding_coefficient = 0
        # for multiprocessing
        self.cpu=mp.cpu_count()
        # create variables for hard constraints, syllable, meter and rhyme, people names.
        self.create_syll_dict(syllables_file)
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]
        self.create_pos_syllables()
        self.create_templates_dict(postag_dict[0])   
        self.filtered_names_rhymes = "LimGen/saved_objects/filtered_names_rhymes.pkl"
        with open(self.filtered_names_rhymes, "rb") as hf:
            self.names_rhymes_list = pickle.load(hf)
        self.female_name_list, self.male_name_list = pickle.load(open("LimGen/saved_objects/name_list.p", "rb"))
        # filtering out unfavorable words
        with open("LimGen/saved_objects/filtered_nouns_verbs.txt", "r") as hf:
            self.filtered_nouns_verbs = [line.strip() for line in hf.readlines()]
            self.filtered_nouns_verbs += self.pos_to_words["IN"] + self.pos_to_words["PRP"]
        self.verb_repeat_whitelist = set(['be', 'is', 'am', 'are', 'was', 'were',
        'being', 'do', 'does', 'did', 'have', 'has', 'had'])

    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        if word not in self.words_to_pos:
            return None
        # Special case
        if word.upper() in self.special_words:
            return set([word.upper()])
        return set(self.words_to_pos[word])



    def split_chunks(self, data):
        data_list=[]
        chuck_len = len(data)//self.cpu + 1
        flag=0
        while_flag=True
        while (while_flag):
            if flag+chuck_len<len(data):
                data_list.append(data[flag:flag+chuck_len])
            else:
                data_list.append(data[flag:])
                while_flag=False
            flag+=chuck_len
        return data_list

    def get_madlib_verbs(self, prompt, pos_list, n_return=20):
        # dictionary {pos: set()}
        try:
            pickle_in=open("LimGen/saved_objects/spacy_prompt_to_madlib_verbs.pickle","rb")
            mydict=pickle.load(pickle_in)
            pickle_in.close()
        except:
            pickle_in=open("LimGen/saved_objects/spacy_prompt_to_madlib_verbs.pickle","wb")
            mydict={}
            pickle.dump(mydict,pickle_in)
            pickle_in.close()
        if prompt in mydict.keys() and mydict[prompt]!=None:
            return mydict[prompt]
        else:
            ret={pos: self.get_similar_word([prompt], n_return=n_return, word_set=set(self.pos_to_words[pos]))for pos in pos_list}
            mydict[prompt]=ret
            with open("LimGen/saved_objects/spacy_prompt_to_madlib_verbs.pickle","wb") as pickle_in:
                pickle.dump(mydict,pickle_in)
            return ret

    def encodes_align(self,previous_data):
        """
        Different lines have different encodes length. We force the encodes to
        have the same length as the minimal length encodes so that the encodes
        is a matrix that GPT2 accepts.
        """
        encodes_length=[len(i[0]) for i in previous_data]
        encodes=[i[0][-min(encodes_length):] for i in previous_data]
        temp=[]
        for i,j in enumerate(previous_data):
            temp.append((encodes[i],j[1],j[2],j[3],j[4],j[5]))
        return temp

    def create_last_word_dict(self, w1s_rhyme_dict, w3s_rhyme_dict):
        """
        Given the rhyme sets, extract all possible last words from the rhyme set
        dictionaries.

        Parameters
        ----------
        w1s_rhyme_dict: dictionary
            Format is {w1: [w2/w5s]}
        w3s_rhyme_dict: dictionary
            Format is {w3: [w4s]}

        Returns
        -------
        dictionary
            Format is {'second': ['apple', 'orange'], 'third': ['apple', orange] ... }

        """
        last_word_dict={}
        for i in ["second","third","fourth","fifth"]:
            temp=[]
            if i== "fifth":
                for k in w1s_rhyme_dict.keys():
                    temp+=w1s_rhyme_dict[k]
            if i=="second":
                for k in w1s_rhyme_dict.keys():
                    temp.append(k)
            if i== "fourth":
                for k in w3s_rhyme_dict.keys():
                    temp+=w3s_rhyme_dict[k]
            if i=="third":
                for k in w3s_rhyme_dict.keys():
                    temp.append(k)
            last_word_dict[i]=[*{*temp}]
        return last_word_dict

    def sylls_bounds(self,partial_template, rhyme_set_curr):
        """
        Return upper and lower bounds of syllables in a POS template.
        """
        def sylls_bounds_last_word(pos, rhyme_set_curr):
            my_list=[len(self.dict_meters[r][0]) for r in rhyme_set_curr if r in self.dict_meters.keys()]
            return max(my_list), min(my_list)

        threshold=0.38
        sylls_up=0
        sylls_lo=0
        if len(partial_template)==1:
            x,y=sylls_bounds_last_word(partial_template[0],rhyme_set_curr)
            return x,y
        else:
            for t in partial_template[:-1]:
                x=[j[0] for j in self.pos_sylls_mode[t] if j[1]>=min(threshold,self.pos_sylls_mode[t][0][1])]
                if len(x)==0:
                    sylls_up+=0
                    sylls_lo+=0
                else:
                    sylls_up+=max(x)
                    sylls_lo+=min(x)
        x,y=sylls_bounds_last_word(partial_template[-1],rhyme_set_curr)
        sylls_up+=x
        sylls_lo+=y
        return sylls_up, sylls_lo

    def there_is_template_new(self,last_word_info,num_sylls, which_line):
        """
        Return a list of possible templates given last word's POS, last word's syllabes
        and which line it is.
        """
        threshold=0.1
        pos=last_word_info[0]
        sylls=last_word_info[1]
        dataset=self.templates[which_line]
        possible=[]
        if pos in dataset.keys():
            for i,_,_ in dataset[pos]:
                sylls_up=0
                sylls_lo=0
                for t in i[:-1]:
                    x=[j[0] for j in self.pos_sylls_mode[t] if j[1]>=min(threshold,self.pos_sylls_mode[t][0][1])]
                    sylls_up+=max(x)
                    sylls_lo+=min(x)
                if num_sylls-sylls>=sylls_lo and num_sylls-sylls<=sylls_up:
                    possible.append(i)
        return possible

    def get_all_templates(self, num_sylls, which_line, last_word_set):
        """
        Given number of syllables a line has, which line it is and all possible last
        words, return all possible POS templates
        """

        last_word_info_set=set()
        temp=[]
        for i in last_word_set:
            if i in self.words_to_pos.keys() and i in self.dict_meters.keys():
                for j in self.get_word_pos(i):
                    last_word_info_set.add((j,len(self.dict_meters[i][0])))
        for i in last_word_info_set:
            temp+=self.there_is_template_new(i, num_sylls, which_line)
        temp=[x for x in set(tuple(x) for x in temp)]
        return temp

    def template_sylls_checking(self, pos_set, sylls_set, template_curr, num_sylls_curr, possible, num_sylls, rhyme_set_curr):
        """
        Check whether the current word could fit into our template with given syllables constraint

        Parameters
        ----------
        pos_set: set
            POS of the current word
        sylls_set: set
            Possible number of syllabes of the current word
        template_curr: list
            Partial, unfinished POS template of the current line (e.g. [NN, VB, NN])
        num_sylls_curr: int
            Syllable count of the partially constructed sentence
        possible: list
            All possible POS templates associated with the current line
        num_sylls: int
            predefined number of syllables the current line should have (e.g. 6,9)

        Returns
        -------
        list
            Format is [(POS, sylls)], a combination of possible POS
            and number of syllables of the current word
        """
        continue_flag=set()
        for t in possible:
            if t[:len(template_curr)]==template_curr and len(t)>len(template_curr)+1:
                for pos in pos_set:
                    if pos==t[len(template_curr)]:
                        for sylls in sylls_set:
                            sylls_up, sylls_lo=self.sylls_bounds(t[len(template_curr)+1:], rhyme_set_curr)
                            if num_sylls-num_sylls_curr-sylls>=sylls_lo and num_sylls-num_sylls_curr-sylls<=sylls_up:
                                continue_flag.add((pos,sylls))
        if len(continue_flag)==0: continue_flag=False
        return continue_flag

    def end_template_checking(self, pos_set, sylls_set, template_curr, num_sylls_curr, possible, num_sylls, relax_story_line):
        """
        Check whether the current word could fit into a template as the last word
        of the line with given syllables constraint

        Parameters
        ----------
        pos_set: set
            POS of the current word
        sylls_set: set
            Possible number of syllabes of the current word
        template_curr: list
            Partial, unfinished POS template of the current line (e.g. [NN, VB, NN])
        num_sylls_curr: int
            Syllable count of the partially constructed sentence
        possible: list
            All possible POS templates associated with the current line
        num_sylls: int
            predefined number of syllables the current line should have (e.g. 6,9)

        Returns
        -------
        list
            Format is [(POS, sylls)], a combination of possible POS
            and number of syllables of the current word
        """

        end_flag=set()
        if self.mode=="multi" and relax_story_line==False:
            for t in possible:
                if t[:len(template_curr)]==template_curr and len(t)==len(template_curr)+1:
                    for pos in pos_set:
                        if pos==t[len(template_curr)]:
                            for sylls in sylls_set:
                                if num_sylls_curr+sylls==num_sylls:
                                    end_flag.add((pos,sylls))

        # this version, does not check last word pos.
        else:
            for t in possible:
                if t[:len(template_curr)]==template_curr and len(t)==len(template_curr)+1:
                    for sylls in sylls_set:
                        if num_sylls_curr+sylls==num_sylls:
                            pos=t[len(template_curr)]
                            end_flag.add((pos,sylls))
        if len(end_flag)==0:
            end_flag=False
        return end_flag

    def different_gender(self,gender):
        temp=[]
        for i in self.names_rhymes_list:
            male=[]
            female=[]
            for j in i[0]:
                if j in self.male_name_list:
                    male.append(j)
                if j in self.female_name_list:
                    female.append(j)
            if gender=="male":
                if len(male)>0:
                    temp.append((male,i[1]))
            if gender=="female":
                if len(female)>0:
                    temp.append((female,i[1]))
        self.names_rhymes_list=temp





    def create_w1s_rhyme_dict(self,prompt):
        self.sum_rhyme=[]
        self.w1s_rhyme_dict=defaultdict(list)
        self.words_to_names_rhyme_dict=defaultdict(list)
        for item in self.names_rhymes_list:
            item_name, item_rhyme= item[0],item[1]
            self.sum_rhyme+=item_rhyme
        if self.storyline=="story":
            self.storyline_second_words=self.get_similar_word([prompt], n_return=50, word_set=set(self.sum_rhyme))
        else:
            self.storyline_second_words=self.rhyming_dictionary.keys()

        #print(self.storyline_second_words)
        for item in self.names_rhymes_list:
            item_name, item_rhyme= item[0],item[1]             
            for i in item_rhyme:
                if i in self.storyline_second_words and i in self.rhyming_dictionary.keys():
                    temp=list(self.rhyming_dictionary[i])
                    #temp=item_rhyme
                    self.w1s_rhyme_dict[i]+=temp
                    self.words_to_names_rhyme_dict[i]+=item_name
        for k in self.w1s_rhyme_dict.keys():
            if k in self.w1s_rhyme_dict[k]:
                self.w1s_rhyme_dict[k].remove(k)
            if len(self.w1s_rhyme_dict[k])==0:
                del self.w1s_rhyme_dict[k]


    def finer_pos_category(self):
        self.special_words= get_finer_pos_words()
        if self.mode=="multi":
            with open("LimGen/saved_objects/templates_processed_tuple.pickle","rb") as pickle_in:
                data=pickle.load(pickle_in)
                data['fifth']=self.random_split(data['fifth'],percent=0.5)
        else:
            with open("LimGen/saved_objects/unified_poems.pickle","rb") as pickle_in:
                data=pickle.load(pickle_in)
                data=random.choice(data)
        temp_data={}
        for k in data.keys():
            temp_line=defaultdict(list)
            for i in data[k].keys():
                for j in data[k][i]:
                    temp_j=[]
                    flag=False
                    if len(j[1])!=len(j[0]): continue
                    for w in range(len(j[1])):
                        if j[1][w].upper() in self.special_words:
                            temp_j.append(j[1][w].upper())
                            if w==len(j[1])-1: flag=True
                        else:
                            temp_j.append(j[0][w])
                    if flag:
                        temp_line[j[1][-1].upper()].append((tuple(temp_j),j[1],j[2]))
                    else:
                        temp_line[i].append((tuple(temp_j),j[1],j[2]))
                    #if (tuple(temp_j),j[1],j[2]) != j:
                        #temp_line[i].append(j)
            temp_data[k]=temp_line
        with open("LimGen/saved_objects/templates_processed_more_tuple.pickle","wb") as pickle_in:
            pickle.dump(temp_data,pickle_in)
        with open("LimGen/saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
            self.templates= pickle.load(pickle_in)
        template_to_line=defaultdict(list)
        for i in ["second","third","fourth","fifth"]:
            for j in self.templates[i].keys():
                for k in self.templates[i][j]:
                    template_to_line[" ".join(k[0])].append(k[1])
        with open("LimGen/saved_objects/template_to_line.pickle","wb") as pickle_in:
            pickle.dump(template_to_line,pickle_in)
        with open("LimGen/saved_objects/template_to_line.pickle","rb") as pickle_in:
            self.template_to_line=pickle.load(pickle_in)
        with open("LimGen/saved_objects/pos_sylls_mode.p","rb") as pickle_in:
            self.pos_sylls_mode= pickle.load(pickle_in)
        with open("LimGen/saved_objects/blacklist_index.p","rb") as pickle_in:
            self.blacklist_index= pickle.load(pickle_in)

        for i in self.special_words:
            try:
                self.pos_sylls_mode[i]=[(len(self.dict_meters[i.lower()][0]),1.0)]
            except:
                self.pos_sylls_mode[i]=[1,1.0]
    def helper(self,prompt):
        if self.storyline=="story":
            try:
                with open("LimGen/saved_objects/spacy_prompt_to_w3s_rhyme_dict.pickle","rb") as pickle_in:
                    mydict=pickle.load(pickle_in)
            except:
                with open("LimGen/saved_objects/spacy_prompt_to_w3s_rhyme_dict.pickle","wb") as pickle_in:
                    mydict={}
                    pickle.dump(mydict,pickle_in)
            if prompt not in mydict.keys():
                w3s = self.get_similar_word([prompt], n_return=200, word_set=set(self.filtered_nouns_verbs))
                #w3s_rhyme_dict = {w3: {word for word in self.get_rhyming_words_one_step(w3) if self.filter_common_word(word, fast=True)} for w3 in w3s}
                w3s_rhyme_dict = {w3: self.rhyming_dictionary[w3] for w3 in w3s if w3 in self.rhyming_dictionary.keys()}
                mydict[prompt]=w3s_rhyme_dict
            self.w3s_rhyme_dict=mydict[prompt]
            #print(self.w3s_rhyme_dict)
            with open("LimGen/saved_objects/spacy_prompt_to_w3s_rhyme_dict.pickle","wb") as pickle_in:
                pickle.dump(mydict,pickle_in)
        else:
            self.w3s_rhyme_dict=dict()
            for k in self.rhyming_dictionary:
                if len(self.rhyming_dictionary[k])>=10:
                    self.w3s_rhyme_dict[k]=self.rhyming_dictionary[k]



    def get_spacy_similarity(self, word1, word2):
        return self.spacy_nlp(word1).similarity(self.spacy_nlp(word2))

    def random_split(self,data,percent=0.5):
        ret=defaultdict(list)
        for i in data.keys():
            ret[i]=[]
            for j in data[i]:
                if random.uniform(0,1)>=0.5:
                    ret[i].append(j)
            if len(ret[i])==0:
                del ret[i]
        return ret


    def create_syll_dict(self, fname):
        """
        Using the cmudict file, returns a dictionary mapping words to their
        intonations (represented by 1's and 0's). Assumed to be larger than the
        corpus of words used by the model.

        Parameters
        ----------
        fname : str
            The name of the file containing the mapping of words to their
            intonations.
        """
        with open(fname, encoding='UTF-8') as f:
            lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
            self.dict_meters = {}
            for i in range(len(lines)):
                line = lines[i]
                newLine = [line[0].lower()]
                if("(" in newLine[0] and ")" in newLine[0]):
                    newLine[0] = newLine[0][:-3]
                chars = ""
                for word in line[1:]:
                    for ch in word:
                        if(ch in "012"):
                            if(ch == "2"):
                                chars += "1"
                            else:
                                chars += ch
                newLine += [chars]
                lines[i] = newLine
                if(newLine[0] not in self.dict_meters):  # THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                    self.dict_meters[newLine[0]] = [chars]
                else:
                    if(chars not in self.dict_meters[newLine[0]]):
                        self.dict_meters[newLine[0]] += [chars]
            self.dict_meters[','] = ['']
            self.dict_meters['.'] = ['']

    def create_pos_syllables(self):
        """
        Creates a mapping from every pos encountered in the corpus to the all of
        the possible number of syllables across all of the words tagged with
        the given pos.
        """
        self.pos_syllables = {}
        for k, v in self.pos_to_words.items():
            self.pos_syllables[k] = set()
            for w in v:
                try:
                    self.pos_syllables[k].add(len(self.dict_meters[w][0]))
                except:
                    continue
        self.pos_syllables[','].add(0)
        self.pos_syllables['.'].add(0)

    def create_templates_dict(self, templates):
        """
        Creates a mapping from every (pos, length of line) encountered in the
        corpus to a list of templates ending with that pos and length.

        Parameters
        ----------
        templates : dict
            A dictionary mapping a pairing of pos to templates containing both
            those pos's (used in previous poem generating algorithms).
        """
        self.templates_dict = {}
        for l in templates.values():
            for t, _ in l:
                if len(t) > 15:
                    continue
                ending_pos = t[-1]
                if (ending_pos, len(t)) not in self.templates_dict:
                    self.templates_dict[(ending_pos, len(t))] = []
                self.templates_dict[(ending_pos, len(t))].append(t)


    def is_duplicate_in_previous_words(self, word, previous):
        """
        Given a new word and previous words, if the word is a noun or adjective,
        and has appeared previously in the poem, return true. Otherwise, return
        false.
        Parameters
        ----------
        word : str
            New word we want to put into the poem.
        previous : list of str
            previous words in the poem.
        Returns
        -------
        bool
            Whether the word is a duplicate.
        """

        if word not in self.words_to_pos:
            return False
        word_pos = self.words_to_pos[word]
        if len(word_pos) == 0:
            return False
        if 'VB' in word_pos[0]:
            return (word in previous and word not in self.verb_repeat_whitelist)
        return ('JJ' == word_pos[0] or 'NN' == word_pos[0]) and word in previous




    def get_similar_word(self, words, seen_words=[], weights=1, n_return=1, word_set=None):
        """
        Given a list of words, return a list of words of a given number most similar to this list.
        <arg>:
        words: a list of words (prompts)
        seen_words: words not to repeat (automatically include words in arg <words> in the following code)
        weights: weights for arg <words>, default to be all equal
        n_return: number of words in the return most similar list
        word_set: a set of words to choose from, default set to the set of words extracted from the definitions of arg <word> in gensim
        <measure of similarity>:
        similarity from gensim squared and weighted sum by <arg> weights
        <return>:
        a list of words of length arg <n_return> most similar to <arg> words
        """
        seen_words_set = set(seen_words) | set(self.ps.stem(word) for word in words)
        '''
        if word_set is None:
            word_set = set()

            for word in words:
                for synset in wn.synsets(word):
                    clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
                    word_set.update(clean_def.lower().split())
                word_set.update({dic["word"] for dic in requests.get(self.api_url, params={'rel_syn': "grace"}).json()})
        '''
        if weights == 1:
            weights = [1] * len(words)

        def cal_score(words, weights, syn):
            score = 0
            for word, weight in zip(words, weights):
                score += max(self.get_spacy_similarity(word, syn), 0) ** 0.5 * weight
            return score / sum(weights)

        syn_score_list = [(syn, cal_score(words, weights, syn)) for syn in word_set if self.ps.stem(syn) not in seen_words_set]
        syn_score_list.sort(key=lambda x: x[1], reverse=True)

        return [e[0] for e in syn_score_list[:n_return]]



    
    def get_rhyming_words_one_step(self, word, max_syllables=3):
        """
        get the rhyming words of <arg> word returned from datamuse api
        <args>:
        word: any word
        <return>:
        a set of words
        """
        return set(d['word'] for d in requests.get(self.api_url, params={'rel_rhy': word}).json() if " " not in d['word'] and d['numSyllables'] <= max_syllables and d['word'] in self.total_vocab)
    


    def fill_in(self, end_word, pos_list=["VB"], seen_words_set=set(), n_return=10, return_score=False):
        """
        <args>:
        end_word: a given last word;
        pos: pos_tag(s) of words to fill in; note that pos="VB" includes all pos_tags "VBP", "VBD", etc; similar holds for "NN", etc;
        seen_words_set: words not to appear;
        n_return: # of mad-libs choices to return;
        return_score: True / False, determine whether or not to return the similarity score between each mad-libs choice and end_word;
        <desc>:
        find mad-libs choices satisfying a given pos_tag(s) for a given end_word, and return similarity scores if needed;
        <return>:
        a list of length <arg> n_return,
        if return_score is False, the list contains <str>, i.e. mad-libs choices,
        if return_score is True, the list contains <tuple>, i.e. (mad-libs choice, its similarity score with end_word)'s;
        """
        words_list_from_pos = [self.pos_to_words[pos_i] for pos_i in self.pos_to_words if pos_i[:2] in pos_list or pos_i in pos_list]
        words_set = set(reduce(lambda x, y: x + y, words_list_from_pos))

        for seen_word in seen_words_set | {end_word}:
            if seen_word in words_set:
                words_set.remove(seen_word)
        related_words_list = [(w, self.get_spacy_similarity(w, end_word)) for w in words_set]
        top_related_words_list = sorted(related_words_list, reverse=True, key=lambda x: x[1])[:n_return]

        return top_related_words_list if return_score else [tup[0] for tup in top_related_words_list]

    def score_averaging(self, scores, method="log"):
        """
        <args>:
        scores: a list / arr of floats;
        method: "log" or "sqrt", i.e. the averaging weight;
        <desc>:
        compute weighted average of <arg> scores,
        the weight is decaying, inverse proportional to the log or sqrt of index;
        <return>:
        return a float, i.e. the weighted average;
        """
        if method == "log":
            weight_arr = np.log(np.arange(2, 2 + len(scores), 1))

        elif method == "sqrt":
            weight_arr = np.sqrt(np.arange(1, 1 + len(scores), 1))

        return np.sum(np.array(scores) * weight_arr) / np.sum(weight_arr)


    def filter_common_word(self, word, fast=False, threshold=0.3):
        if fast:
            pos_list = ["VBP"]
        else:
            pos_list = ["VB", "NN"]
        fill_in_return = self.fill_in(word, pos_list=pos_list, n_return=10, return_score=True)

        if self.score_averaging([tup[1] for tup in fill_in_return]) >= threshold:
            return True
        return False




    

    def load_city_list(self):
        city_list_file = 'LimGen/saved_objects/city_names.txt'
        l = []
        with open(city_list_file, 'rb') as cities:
            for line in cities:
                l.append(line.rstrip().decode('utf-8').lower())
        return l

    # For instance, if correct meter is: da DUM da da DUM da da DUM, pass in
    # stress = [1,4,7] to enforce that the 2nd, 5th & 8th syllables have stress.
    def is_correct_meter(self, template, num_syllables=[8], stress=[1, 4, 7]):
        meter = []
        n = 0
        for x in template:
            if x not in self.dict_meters:
                return False
            n += len(self.dict_meters[x][0])
            curr_meter = self.dict_meters[x]
            for i in range(max([len(j) for j in curr_meter])):
                curr_stress = []
                for possible_stress in curr_meter:
                    if len(possible_stress)>=i+1:
                        curr_stress.append(possible_stress[i])
                meter.append(curr_stress)
        return (not all(('1' not in meter[i]) for i in stress)) \
            and (n in num_syllables)

    def gen_first_line_new(self, last_word, contains_adjective=True, strict=False, search_space=100, seed=None):
        """
        Generetes all possible first lines of a Limerick by going through a
        set of template. Number of syllables is always 8 or 9.

        Parameters
        ----------
        w1 : str
            The last word in the line, used to generate backwards from.
        last_word : str
            The last word of the first_line sentence specified by the user.
        strict : boolean, optional
            Set to false by default. If strict is set to false, this method
            will look for not only sentences that end with last_word, but also
            sentences that end with a word that rhyme with last_word.

        Returns
        -------
        first_lines : list
            All possible first line sentences.
        """

        def get_num_sylls(template):
            n = 0
            for x in template:
                if x not in self.dict_meters:
                    return 0
                n += len(self.dict_meters[x][0])
            return n

        female_name_list, male_name_list = self.female_name_list, self.male_name_list
        city_name_list = self.load_city_list()
        templates, placeholders, dict = get_first_line_templates()

        if strict:
            if last_word not in female_name_list and \
                last_word not in male_name_list and \
                last_word not in city_name_list:
                raise Exception('last word ' + last_word + ' is not a known name or location')
            last_word_is_location = last_word in city_name_list
            last_word_is_male = last_word in male_name_list
            last_word_is_female = last_word in female_name_list

        w_response = {last_word}
        candidate_sentences = []
        candidate_sentences_with_place = []

        # Get top 5 that is related to the seed word
        if seed is not None:
            adj_dict_with_distances = [(self.get_spacy_similarity(word, seed), word) for word in dict['JJ'] if word in self.words_to_pos]
            adj_dict_with_distances = heapq.nlargest(5, adj_dict_with_distances, key=lambda x: x[0])
            adj_dict_with_distances = [a[1] for a in adj_dict_with_distances]

            person_with_distances = []
            for gender in dict['PERSON']:
                person_with_distances += [(self.get_spacy_similarity(word, seed), word, gender) for word in dict['PERSON'][gender]]
            person_with_distances = heapq.nlargest(5, person_with_distances, key=lambda x: x[0])
            person_with_distances_dict = defaultdict(list)
            for person in person_with_distances:
                person_with_distances_dict[person[2]].append(person[1])

        for template in templates:
            if strict and last_word_is_location and template[-1] != 'PLACE':
                continue
            if strict and (last_word_is_male or last_word_is_female) and \
                template[-1] != 'NAME':
                continue
            if not contains_adjective and ('JJ' in template):
                continue
            candidates = []
            for word in template:
                if word not in placeholders:
                    continue
                if word == 'PERSON':
                    person_dict = dict['PERSON']
                    if seed is not None:
                        person_dict = person_with_distances_dict
                    if len(candidates) == 0:
                        candidates = [{'PERSON': p, 'GENDER': 'MALE'} for p in person_dict['MALE']] \
                            + [{'PERSON': p, 'GENDER': 'FEMALE'} for p in person_dict['FEMALE']] \
                            + [{'PERSON': p, 'GENDER': 'NEUTRAL'} for p in person_dict['NEUTRAL']]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for gender in person_dict:
                                for p in person_dict[gender]:
                                    new_d = copy.deepcopy(d)
                                    new_d['PERSON'] = p
                                    new_d['GENDER'] = gender
                                    new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'JJ':
                    adj_dict = dict['JJ']
                    if seed is not None:
                        adj_dict = adj_dict_with_distances

                    if len(candidates) == 0:
                        candidates = [{'JJ': w} for w in adj_dict]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for adj in adj_dict:
                                new_d = copy.deepcopy(d)
                                new_d['JJ'] = adj
                                new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'IN':
                    in_dict = dict['IN']
                    new_candidates = []
                    for d in candidates:
                        for i in in_dict:
                            new_d = copy.deepcopy(d)
                            new_d['IN'] = i
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'PLACE':
                    if strict and last_word_is_location:
                        for d in candidates:
                            d['PLACE'] = last_word
                    new_candidates = []
                    for d in candidates:
                        for city in city_name_list:
                            new_d = copy.deepcopy(d)
                            new_d['PLACE'] = city
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'NAME':
                    # Only select candidates with the correct gender as the name
                    if strict and (last_word_is_male or last_word_is_female):
                        new_candidates = []
                        for d in candidates:
                            if d['GENDER'] == 'FEMALE' and last_word_is_female:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'MALE' and last_word_is_male:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'NEUTRAL':
                                d['NAME'] = last_word
                                new_candidates.append(d)
                        candidates = new_candidates
                        continue

                    new_candidates = []
                    for d in candidates:
                        if d['GENDER'] == 'MALE' or d['GENDER'] == 'NEUTRAL':
                            for name in male_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                        if d['GENDER'] == 'FEMALE' or d['GENDER'] == 'NEUTRAL':
                            for name in female_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                    candidates = new_candidates

            is_template_with_place = ('PLACE' in template)
            for candidate in candidates:
                if candidate[template[-1]] not in w_response:
                    continue
                new_sentence = copy.deepcopy(template)
                for i in range(len(new_sentence)):
                    if new_sentence[i] in placeholders:
                        new_sentence[i] = candidate[new_sentence[i]]
                # First line always has 8 or 9 syllables
                if self.is_correct_meter(new_sentence, num_syllables=[8, 9]):
                    if is_template_with_place:
                        candidate_sentences_with_place.append(new_sentence)
                    else:
                        candidate_sentences.append(new_sentence)

        random.shuffle(candidate_sentences)
        random.shuffle(candidate_sentences_with_place)
        return candidate_sentences[:int(search_space*0.7)] \
        + candidate_sentences_with_place[:min(int(search_space*0.3), int(len(candidate_sentences)*0.3))]
        return candidate_sentences[:search_space]
    def gen_first_line_deepspeare(self, last_word, contains_adjective=True, strict=False, search_space=1, seed=None):
        """
        similar to gen_first_line_new
        this function writes the first line for the existing lines of limericks from deep speare.
        """

        def get_num_sylls(template):
            n = 0
            for x in template:
                if x not in self.dict_meters:
                    return 0
                n += len(self.dict_meters[x][0])
            return n

        female_name_list, male_name_list = self.female_name_list, self.male_name_list
        city_name_list = self.load_city_list()
        templates, placeholders, dict = get_first_line_templates()

        if strict:
            if last_word not in female_name_list and \
                last_word not in male_name_list and \
                last_word not in city_name_list:
                raise Exception('last word ' + last_word + ' is not a known name or location')
            last_word_is_location = last_word in city_name_list
            last_word_is_male = last_word in male_name_list
            last_word_is_female = last_word in female_name_list

        w_response = {last_word}
        candidate_sentences = []
        candidate_sentences_with_place = []

        # Get top 5 that is related to the seed word
        if seed is not None:
            adj_dict_with_distances = [(self.get_spacy_similarity(word, seed), word) for word in dict['JJ'] if word in self.words_to_pos]
            adj_dict_with_distances = heapq.nlargest(5, adj_dict_with_distances, key=lambda x: x[0])
            adj_dict_with_distances = [a[1] for a in adj_dict_with_distances]

            person_with_distances = []
            for gender in dict['PERSON']:
                person_with_distances += [(self.get_spacy_similarity(word, seed), word, gender) for word in dict['PERSON'][gender]]
            person_with_distances = heapq.nlargest(5, person_with_distances, key=lambda x: x[0])
            person_with_distances_dict = defaultdict(list)
            for person in person_with_distances:
                person_with_distances_dict[person[2]].append(person[1])

        for template in templates:
            if strict and last_word_is_location and template[-1] != 'PLACE':
                continue
            if strict and (last_word_is_male or last_word_is_female) and \
                template[-1] != 'NAME':
                continue
            if not contains_adjective and ('JJ' in template):
                continue
            candidates = []
            for word in template:
                if word not in placeholders:
                    continue
                if word == 'PERSON':
                    person_dict = dict['PERSON']
                    if seed is not None:
                        person_dict = person_with_distances_dict
                    if len(candidates) == 0:
                        candidates = [{'PERSON': p, 'GENDER': 'MALE'} for p in person_dict['MALE']] \
                            + [{'PERSON': p, 'GENDER': 'FEMALE'} for p in person_dict['FEMALE']] \
                            + [{'PERSON': p, 'GENDER': 'NEUTRAL'} for p in person_dict['NEUTRAL']]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for gender in person_dict:
                                for p in person_dict[gender]:
                                    new_d = copy.deepcopy(d)
                                    new_d['PERSON'] = p
                                    new_d['GENDER'] = gender
                                    new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'JJ':
                    adj_dict = dict['JJ']
                    if seed is not None:
                        adj_dict = adj_dict_with_distances

                    if len(candidates) == 0:
                        candidates = [{'JJ': w} for w in adj_dict]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for adj in adj_dict:
                                new_d = copy.deepcopy(d)
                                new_d['JJ'] = adj
                                new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'IN':
                    in_dict = dict['IN']
                    new_candidates = []
                    for d in candidates:
                        for i in in_dict:
                            new_d = copy.deepcopy(d)
                            new_d['IN'] = i
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'PLACE':
                    if strict and last_word_is_location:
                        for d in candidates:
                            d['PLACE'] = last_word
                    new_candidates = []
                    for d in candidates:
                        for city in city_name_list:
                            new_d = copy.deepcopy(d)
                            new_d['PLACE'] = city
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'NAME':
                    # Only select candidates with the correct gender as the name
                    if strict and (last_word_is_male or last_word_is_female):
                        new_candidates = []
                        for d in candidates:
                            if d['GENDER'] == 'FEMALE' and last_word_is_female:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'MALE' and last_word_is_male:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'NEUTRAL':
                                d['NAME'] = last_word
                                new_candidates.append(d)
                        candidates = new_candidates
                        continue

                    new_candidates = []
                    for d in candidates:
                        if d['GENDER'] == 'MALE' or d['GENDER'] == 'NEUTRAL':
                            for name in male_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                        if d['GENDER'] == 'FEMALE' or d['GENDER'] == 'NEUTRAL':
                            for name in female_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                    candidates = new_candidates

            is_template_with_place = ('PLACE' in template)
            for candidate in candidates:
                if candidate[template[-1]] not in w_response:
                    continue
                new_sentence = copy.deepcopy(template)
                for i in range(len(new_sentence)):
                    if new_sentence[i] in placeholders:
                        new_sentence[i] = candidate[new_sentence[i]]
                # First line always has 8 or 9 syllables
                if self.is_correct_meter(new_sentence, num_syllables=[8, 9]):
                    if is_template_with_place:
                        candidate_sentences_with_place.append(new_sentence)
                    else:
                        candidate_sentences.append(new_sentence)

        random.shuffle(candidate_sentences)
        random.shuffle(candidate_sentences_with_place)
        return candidate_sentences[:search_space]

 