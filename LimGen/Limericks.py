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
from .utils import utils
from .saved_objects.Finer_POS import get_finer_pos_words

class LimGen(utils):
	def __init__(self, gender,prompt, search_space, retain_space, 
		stress=True, prob_threshold=-10, mode="multi", 
		relax_story_line=False, beam_search=None, multiprocessing=True, storyline="story"):
		"""
		Generate poems with multiple templat es given a seed word (prompt) and GPT2
		search space.
        Parameters
        ----------
		prompt: str
			A seed word used to kickstart poetry generation.
        search_space : int
            Search space of the sentence finding algorithm.
            The larger the search space, the more sentences the network runs
            in parallel to find the best one with the highest score.
        retain_space : int
            How many sentences per template to keep.
		stress: bool
			Whether we enforce stress.
		prob_threshold: float
			If the probability of a word is lower than this threshold we will not consider
			this word. Set it to None to get rid of it.
		"""
		print("=================== Initializing ==================================")
		super(LimGen,self).__init__()
		self.storyline=storyline
		if self.storyline=="story":
			self.madlib_flag=True
		else:
			self.madlib_flag=False
		self.multiprocessing=multiprocessing
		self.pdb_flag=True
		self.which_line_dict={0:"second", 1:"third",2:"fourth",3:"fifth"}
		self.beam_search=beam_search
		self.mode=mode
		self.relax_story_line=relax_story_line
		self.prob_threshold = prob_threshold
		self.enforce_stress = stress
		if self.mode!="multi":
			self.relax_story_line=True
			self.prob_threshold = None
		self.finer_pos_category()
		self.pos_weight()
		if gender=="male":
			self.temp_name="Robert"
		if gender=="female":
			self.temp_name="Sarah"
		self.different_gender(gender)
		self.create_w1s_rhyme_dict(prompt)
		self.n_w25_threshold=10
		#print("===============================   helper       ==============================================")
		self.helper(prompt)
		#print("===============================   end helper       ==============================================")
		self.madlib_verbs = self.get_madlib_verbs(prompt,["VBD", "VBN", "VB", "VBZ", "VBP", "VBG"])
		# get rid of common words
		if "was" in self.madlib_verbs["VBD"]:
			self.madlib_verbs["VBD"].remove("was")
			#print("remove was \n")
		# self.madlib_verbs = self.get_madlib_verbs(prompt,["NN","NNS"])
		#print("------- Madlib Verbs ------")
		#print(self.madlib_verbs)
		self.last_word_dict=self.create_last_word_dict(self.w1s_rhyme_dict,self.w3s_rhyme_dict)
		self.prompt=prompt
		self.search_space=search_space
		self.retain_space=retain_space
		print("=================== Finished Initializing ==================================")

	def gen_poem(self):
		previous_data=[]
		candidates=self.gen_first_line_new(self.temp_name.lower(),search_space=5,strict=True,seed=self.prompt)
		assert len(candidates)>0, "no first line"
		#print(candidates)
		for text in candidates:
			first_line_encodes = self.enc.encode(" ".join(text))
			previous_data.append((tuple(first_line_encodes),(0,),tuple(text)+("\n",), (text[-1],"\n"),("",""),(0,)))
		for which_line, num_sylls in zip(["second","third","fourth","fifth"],[9,6,6,9]):

			print("====================================================================================")
			print("====================================================================================")
			print("======================= starting {} line generation =============================".format(which_line))
			print("====================================================================================")
			print("====================================================================================")
			last_word_set=self.last_word_dict[which_line]
			possible=self.get_all_templates(num_sylls,which_line,last_word_set)
			previous_data=self.gen_line_flexible(previous_data=previous_data, possible=possible,num_sylls=num_sylls, search_space=self.search_space,retain_space=self.retain_space, which_line=which_line)
		if self.beam_search=="candidate_rank":
			previous_data, _ = self.diversity_sort(data=previous_data,last=True, diversity=False)
		if self.beam_search=="MTBS":
			previous_data, _ = self.diversity_sort_MTBS(data=previous_data,last=True, which_line=which_line)
		return previous_data, self.template_to_line, self.words_to_names_rhyme_dict

	def pos_weight(self):
		'''
		calculate the weight for each pos for each specific line
		'''
		def softmax(data1):
			data=[d/sum(data1) for d in data1]
			return data
			temp=[math.exp(d) for d in data]
			sum_temp=sum(temp)
			return [t/sum(temp) for t in temp]
		self.pos_weight_dict={}
		for key in self.templates.keys():
			temp=[]
			for k in self.templates[key].keys():
				for item in self.templates[key][k]:
					temp+=list(item[0])
			occur=Counter(temp)
			key_list=list(occur.keys())
			temp2=[]
			for k in key_list:
				temp2.append(len(temp)/occur[k])
			ret=softmax(temp2)
			occur2={}
			for ii, k in enumerate(key_list):
				occur2[k]=ret[ii]
			self.pos_weight_dict[key]=occur2


	def diversity_sort_MTBS(self,search_space=None, retain_space=None,data=None, finished=None, last=False, which_line=None):
		def discount_mean(data_list):
			ret=0
			gamma=0.9
			count=0
			for i in data_list[::-1]:
				ret+=i*(gamma**count)
				count+=1
			return ret/len(data_list)
		def lined_template(temp):
			lines_template=[]
			line_template=[]
			for i in temp:
				if i!="\n":
					line_template.append(i)
				else:
					lines_template.append(line_template)
					line_template=[]
			return lines_template
		def total_weighted_Hamming_distance(temp1,temp2):
			dist=[]
			diversity_factor=1
			lines_template_1=lined_template(temp1)[1:]
			lines_template_2=lined_template(temp2)[1:]
			'''
			if len(lines_template_2)==4 and self.pdb_flag:
				pdb.set_trace()
				self.pdb_flag=False
			'''
			for i in range(len(lines_template_1)):
				dist.append(weighted_Hamming_distance(lines_template_1[i],lines_template_2[i],self.which_line_dict[i]))
			dist=dist[::-1] # reverse dist
			ret=0
			for i in range(len(dist)):
				ret+=dist[i]*(diversity_factor**i)
			return dist


		def weighted_Hamming_distance(template1, template2, which_line):
			dist=0

			for i in range(max(len(template1),len(template2))):
				if i<len(template1) and i<len(template2):
					if template1[i]!=template2[i]:
						if template1[i] in self.pos_weight_dict[which_line].keys() and template2[i] in self.pos_weight_dict[which_line].keys():
							dist+=max(self.pos_weight_dict[which_line][template1[i]],self.pos_weight_dict[which_line][template2[i]])
				if i>=len(template1):
					if template2[i] in self.pos_weight_dict[which_line].keys():
						dist+=self.pos_weight_dict[which_line][template2[i]]
				if i>=len(template2):
					if template1[i] in self.pos_weight_dict[which_line].keys():
						dist+=self.pos_weight_dict[which_line][template1[i]]
			return dist
		def find_best_template(temp_data,already_selected_templates, which_line):
			ret={}
			if len(already_selected_templates)==0:
				for t in temp_data.keys():
					#ret[t]=np.mean([np.mean(item[1][-len(item[4]):]) for item in temp_data[t]])
					ret[t]=np.mean([np.mean(item[1]) for item in temp_data[t]])
					#ret[t]=np.mean([discount_mean(item[1]) for item in temp_data[t]])

			else:
				for t in temp_data.keys():
					dist=np.sum([total_weighted_Hamming_distance(t,tt) for tt in already_selected_templates])
					#ret[t]=np.mean([np.mean(item[1][-len(item[4]):]) for item in temp_data[t]])*dist
					ret[t]=np.mean([np.mean(item[1]) for item in temp_data[t]])*dist
					#ret[t]=np.mean([discount_mean(item[1]) for item in temp_data[t]])*dist
			return sorted(ret.items(),key=lambda item: item[1],reverse=True)[0][0]

		if last:
			data_new=heapq.nlargest(len(data), data, key=lambda x: np.mean(x[1]))
			return data_new,0

		temp_data=defaultdict(set)
		# Key is "template; current_line_template". For each key we only keep retain_space sentences
		for n in data:
			if not finished:
				key=n[3]+n[4]
			else:
				key=n[3] # because the curr is already merged.
			temp_data[key].add(n)
		pre_select_data={}
		selected_data=[]
		already_selected_templates=[]
		for k in temp_data.keys():
			#temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1][-len(x[4]):]))
			temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1]))
			#temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: discount_mean(x[1]))
			pre_select_data[k]=temp
		count=0
		while count<search_space and len(pre_select_data.keys())>0:
			template_chosen=find_best_template(pre_select_data,already_selected_templates,which_line)
			already_selected_templates.append(template_chosen)
			selected_data+=pre_select_data[template_chosen]
			del pre_select_data[template_chosen]
			count+=1

		

		return selected_data, len(temp_data.keys())




	def diversity_sort(self,search_space=None, retain_space=None,data=None, finished=None, last=False, diversity=True):
		"""
		Given a list of sentences, put them in bins according to their templates, get
		retain_space sentences from each bin and form a list, and get top search_space sentences from
		the list.

        Parameters
        ----------
		search_space: int
			Number of sentences returned
		data: list
			Input sentences
		finished: bool
			Whether the current sentence is completed
		"""
		if last:
			data_new=heapq.nlargest(len(data), data, key=lambda x: np.mean(x[1]))
			return data_new,0
		if diversity:
			temp_data=defaultdict(set)
			# Key is "template; current_line_template". For each key we only keep retain_space sentences
			for n in data:
				if not finished:
					key=n[3]+n[4]
				else:
					key=n[3] # because the curr is already merged.
				temp_data[key].add(n)
			data=[]
			list_of_keys=list(temp_data.keys())
			x=random.sample(list_of_keys, len(list_of_keys))
			for k in x:
				if not finished:
					temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1]))
					data.append((temp,np.max([np.mean(m[1]) for m in temp])))
				else:
					temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1]))
					data.append((temp,np.max([np.mean(m[1]) for m in temp])))
			data=heapq.nlargest(min(len(data),search_space),data, key = lambda x: x[1])
			data_new=[]
			for k in data:
				data_new+=k[0]
			data=data_new
		else:
			# this is normal beam search
			if not finished:
				data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]))
			else:
				data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]))
			return data_new, 0

		return data_new, len(temp_data.keys())


	def gen_line_flexible(self, previous_data, possible, num_sylls, search_space, retain_space,which_line):
		'''
		Generate a line using multiple templates.

        Parameters
        ----------
		previous_data: list of tuples
			Each element has a tuple structure (encodes, score, text, template, (w1,w3)).
			encodes: list of int
				encodes are gpt-index for words
			score: double
				 Score is the probability of the line
			text: list of str
				the text corresponding to encodes
			template: list of POS
				template is all existing templates, e.g. if we are genrating third line right now, template is ["somename","second line templates"].
			(w1,w3):  tuple,
				It records the rhyme word in this sense, second line and fifth line last word have to be
				in the w1s_rhyme_dict[w1], the fourth line last word have to be in w3s_rhyme_dict[w3]. Note if we are only at line2,
				then w3 is '', because it hasn't happened yet.
		possible: list
			Possible templates for current line.
		search_space: int
			We generate search_space lines and sort them by probability to find out the bes line.
		num_sylls: int
			Number of syllables of current line
		which_line: int
			which line it is (1,2,3,4 or 5)
		'''
		previous_data=self.encodes_align(previous_data)
		sentences=[]
		for i in previous_data:
			template_curr=()
			num_sylls_curr=0
			sentences.append([i[0],i[1],i[2],i[3],template_curr,num_sylls_curr,i[4], i[5]])
		# sentences is a tuple, each element looks like (encodes, score, text, template, current_line_template, how_many_syllabus_used_in_current_line, (w1,w3), moving average/word similarity list)
		# curren_line_template is a partial template of the currently developing line. template is all the POS of the developing poem, with lines separated by "\n".
		finished_sentences=[]
		iteration=0
		new_sentences=[1]
		while(len(new_sentences)>0):
			iteration+=1
			context_token=[s[0] for s in sentences]
			m=len(context_token)
			context_token=np.array(context_token).reshape(m,-1)
			print("******************************** gpt2 Starts Processing Next Word **********************************")
			logits = score_model(model_name=self.model_name, context_token = context_token)
			print("******************************** gpt2 Finished Processing Next Word **********************************")
			if self.multiprocessing:
				logits_list= self.split_chunks(logits)
				sentences_list=self.split_chunks(sentences)
				manager = mp.Manager()
				output=manager.Queue()
				processes = [mp.Process(target=self.batch_process_word, args=(which_line, possible,num_sylls,logits_list[mp_index], sentences_list[mp_index], output, True,retain_space)) for mp_index in range(len(logits_list)) ]
				print("******************************** multiprocessing starts with {} processes *************************************".format(len(processes)))
				for p in processes:
					p.start()
				for p in processes:
					p.join()
				print("********************************** multiprocessing ends *****************************************************")
				results = [output.get() for p in processes]
				new_sentences, quasi_finished_sentences = [], []
				for result in results:
					new_sentences += result[0]
					quasi_finished_sentences += result[1]
			else:
				new_sentences, quasi_finished_sentences= self.batch_process_word(which_line, possible, num_sylls, logits, sentences)
			if self.punctuation[which_line]:
				if len(quasi_finished_sentences)>0:
					if self.beam_search=="candidate_rank":
						quasi_finished_sentences, diversity=self.diversity_sort(search_space,retain_space,quasi_finished_sentences, finished=True, diversity=False)
					if self.beam_search=="MTBS":
						quasi_finished_sentences, diversity=self.diversity_sort_MTBS(search_space,retain_space,quasi_finished_sentences, finished=True, which_line=which_line)
					context_token=[s[0] for s in quasi_finished_sentences]
					m=len(context_token)
					context_token=np.array(context_token).reshape(m,-1)
					print("################################## gpt2 Starts Adding Punctuation #############################")
					logits = score_model(model_name=self.model_name, context_token = context_token)
					print("################################## gpt2 Finished Adding Punctuation #############################")
					for i,j in enumerate(logits):
						sorted_index=np.argsort(-1*j)
						for index in sorted_index:
							word = self.enc.decode([index]).lower().strip()
							if word==self.sentence_to_punctuation[which_line]:
								finished_sentences.append((quasi_finished_sentences[i][0] + (index,),
															quasi_finished_sentences[i][1] + (np.log(j[index]),),
															quasi_finished_sentences[i][2]+(word,),
															quasi_finished_sentences[i][3]+(word,),
															quasi_finished_sentences[i][4],
															quasi_finished_sentences[i][5]))
								break
			else:
				for q in quasi_finished_sentences:
					finished_sentences.append(q)
			print("========================= iteration {} ends =============================".format(iteration))
			if self.beam_search=="candidate_rank":
				sentences, diversity=self.diversity_sort(search_space,retain_space,new_sentences, finished=False, diversity=False)
			if self.beam_search=="MTBS":
				sentences, diversity=self.diversity_sort_MTBS(search_space,retain_space,new_sentences, finished=False, which_line=which_line)
			print("{} sentences before diversity_sort, {} sentences afterwards, diversity {}, this iteration has {} quasi_finished_sentences,  now {} finished_sentences \n".format(len(new_sentences),len(sentences), diversity, len(quasi_finished_sentences),len(finished_sentences)))
		assert len(sentences)==0, "something wrong"
		if self.beam_search=="candidate_rank":
			previous_data_temp, _=self.diversity_sort(search_space,retain_space,finished_sentences, finished=True,diversity=False)
		if self.beam_search=="MTBS":
			previous_data_temp, _=self.diversity_sort_MTBS(search_space,retain_space,finished_sentences, finished=True, which_line=which_line)
		previous_data=[(i[0],i[1],i[2]+("\n",),i[3]+("\n",),i[4],i[5]+(0,)) for i in previous_data_temp]
		return previous_data


	def batch_process_word(self, which_line, possible, num_sylls, logits, sentences, output=None, madlib_flag=None, retain_space=None):
		'''
		Batch process the new possible word of a group of incomplete sentences.

        Parameters
        ----------
		possible: list
			list of possible templates
		num_sylls: int
			we generate search_space lines and sort them by probability to find out the bes line.
		which_line: int
			which line it is (1,2,3,4 or 5)
		num_sylls: int
			wumber of syllables of current line.
		logits: list
			Logits is the output of GPT model.
		sentences: list
			List of sentences that we currently are generating.
		'''

		new_sentences = []
		quasi_finished_sentences = []
		for i,j in enumerate(logits):
			if self.beam_search=="candidate_rank":
				new_sentences_per_beam=[]
				quasi_finished_sentences_per_beam=[]
			sorted_index=np.argsort(-1*j)
			word_list_against_duplication=[]
			# sentences is a tuple, each element looks like (encodes, score, text, template, current_line_template, how_many_syllabus_used_in_current_line, (w1,w3), moving average)
			# curren_line_template is a partial template of the currently developing line.
			#template is all the POS of the developing poem, with lines separated by "\n".
			template_curr=sentences[i][4]
			num_sylls_curr=sentences[i][5]
			moving_avg_curr=sentences[i][7][-1]
			rhyme_set_curr = set()
			if which_line=="second":
				rhyme_set_curr = self.w1s_rhyme_dict.keys()
				rhyme_word="second_line_special_case"
			if which_line=="fifth":
				rhyme_set_curr=self.w1s_rhyme_dict[sentences[i][6][0]]
				rhyme_word=sentences[i][6][0]
			if which_line=="third":
				rhyme_set_curr = self.w3s_rhyme_dict.keys()
				rhyme_word="third_line_special_case"
			if which_line=="fourth":
				rhyme_set_curr = self.w3s_rhyme_dict[sentences[i][6][1]]
				rhyme_word=sentences[i][6][1]
			assert len(rhyme_set_curr)>0
			# If it is the fifth line, the current template has to corresponds to the fourth line template
			# because they are usually one sentence

			for ii,index in enumerate(sorted_index):
				# Get current line's template, word embedding average, word, rhyme set, etc.
				relax_story_line=False
				word = self.enc.decode([index]).lower().strip()
				if word not in self.total_vocab:
					continue
				if self.prob_threshold is not None and np.log(j[index]) < self.prob_threshold:
					relax_story_line=True # when we are considering low probability words, we relax the last word pos constraints.
					if len(new_sentences)>=10 or len(quasi_finished_sentences)>0:
						break
				if word in word_list_against_duplication:
					continue
				elif len(word)==0:
					continue
				# note that both , and . are in these keys()
				elif word not in self.words_to_pos.keys() or word not in self.dict_meters.keys():
					continue
				else:
					if index in self.blacklist_index:
						continue
					pos_set=self.get_word_pos(word)
					sylls_set=set([len(m) for m in self.dict_meters[word]])
					if len(pos_set)==0 or len(sylls_set)==0:
						continue
					# If the word is a noun or adjective and has appeared
					# previously, we discard the sentence.
					if self.is_duplicate_in_previous_words(word, sentences[i][2]):
						continue

					# If stress is incorrect, continue
					if self.enforce_stress:
						possible_syllables = self.dict_meters[word]
						word_length = min(sylls_set)

						stress = [1, 4] if (which_line == "third" or which_line == "fourth") else [1, 4, 7]
						correct_stress = True
						# There is a stress on current word
						for stress_position in stress:
							if num_sylls_curr <= stress_position and num_sylls_curr + word_length > stress_position:
								stress_syllable_pos = stress_position - num_sylls_curr
								if all(s[stress_syllable_pos] != '1' for s in possible_syllables):
									correct_stress = False
								break
						if not correct_stress:
							continue

					# end_flag is the (POS, Sylls) of word if word can be the last_word for a template, False if not
					# continue_flag is (POS,Sylls) if word can be in a template and is not the last word. False if not
					continue_flag=self.template_sylls_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls, rhyme_set_curr=rhyme_set_curr)
					end_flag=self.end_template_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls, relax_story_line=relax_story_line)
					# placeholder code, no effect, only to resolve compatibility issue.
					tuple_of_wema=tuple([m for m in sentences[i][7][:-1]])+(0,)
					if continue_flag:
						word_list_against_duplication.append(word)
						for continue_sub_flag in continue_flag:

							# If current word POS is VB, current line is second line and word is not in our
							# precomputed list, throw away the sentence
							if self.madlib_flag:
								curr_vb_pos = continue_sub_flag[0]
								if 'VB' in curr_vb_pos and which_line == 'second' \
									and not any('VB' in pos_tag for pos_tag in template_curr):
									if word not in self.madlib_verbs[curr_vb_pos]:
										continue
							word_tuple = (sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3],
												sentences[i][4]+(continue_sub_flag[0],),
												sentences[i][5]+continue_sub_flag[1],
												sentences[i][6],
												tuple_of_wema)
							if self.beam_search=="candidate_rank":
								if len(new_sentences_per_beam)<retain_space:
									new_sentences_per_beam.append(word_tuple)
							else:
								new_sentences.append(word_tuple)
					if end_flag:
						for end_sub_flag in end_flag:
							if which_line=="second":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												(word,""),
												tuple_of_wema)
									if self.beam_search=="candidate_rank":
										if len(quasi_finished_sentences_per_beam)<retain_space:
											quasi_finished_sentences_per_beam.append(word_tuple)
									else:
										quasi_finished_sentences.append(word_tuple)
							if which_line=="third":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												(sentences[i][6][0],word),
												tuple_of_wema)
									if self.beam_search=="candidate_rank":
										if len(quasi_finished_sentences_per_beam)<retain_space:
											quasi_finished_sentences_per_beam.append(word_tuple)
									else:
										quasi_finished_sentences.append(word_tuple)
							if which_line=="fourth" or which_line=="fifth":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												sentences[i][6],
												tuple_of_wema)
									if self.beam_search=="candidate_rank":
										if len(quasi_finished_sentences_per_beam)<retain_space:
											quasi_finished_sentences_per_beam.append(word_tuple)
									else:
										quasi_finished_sentences.append(word_tuple)
			if self.beam_search=="candidate_rank":
				new_sentences+=new_sentences_per_beam
				quasi_finished_sentences+=quasi_finished_sentences_per_beam
		if self.multiprocessing:
			output.put((new_sentences, quasi_finished_sentences))
		else:
			return new_sentences, quasi_finished_sentences
