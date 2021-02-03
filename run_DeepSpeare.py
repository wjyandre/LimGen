"""
Author:         Jianyou (Andre) Wang
Date:           Sep 2020
"""

from LimGen.utils import utils
import pickle
import pdb
import random
import os
with open("deepspeare/deepspeare_results.pickle","rb") as f:
	quatrains=pickle.load(f)
lg=utils()
new_quatrains=[]
male=set(["him","himself","his","he","man","boy","guy"])
female=set(["her","herself","hers","she","woman","girl","gal","lady"])
for qua in quatrains:
	words=[]
	for line in qua:
		words+=line.split(" ")
	gender="neutral"
	if len(male.intersection(set(words)))>0 and len(female.intersection(set(words)))==0:
		gender="male"
	if len(male.intersection(set(words)))==00 and len(female.intersection(set(words)))>0:
		gender="female"
	name="not-name"
	#rhymes=lg.get_rhyming_words_one_step(qua[0].split(" ")[-1], max_syllables=3)
	if qua[0].split(" ")[-1] in lg.rhyming_dictionary.keys():
		rhymes=lg.rhyming_dictionary[qua[0].split(" ")[-1]]
	else:
		print("cannot find rhyming name for the last four lines of this limerick")
		continue
	for item in lg.names_rhymes_list:
		if len(set(rhymes).intersection(set(item[1])))>0:
			x=set(item[0]).intersection(set(lg.female_name_list))
			y=set(item[0]).intersection(set(lg.male_name_list))
			if gender=="female":
				if len(x)>0:
					name=random.choice(list(x))
					break
			if gender=="male":
				if len(y)>0:
					name=random.choice(list(y))
					break
			if gender=="neutral":
				if len(x.union(y))>0:
					name=random.choice(list(x.union(y)))
					break
	if name != "not-name":
		for line in qua:
			for word in line.split(" "):
				if word in lg.words_to_pos:
					if "JJ" in lg.words_to_pos[word]:
						prompt=word
						break
		first_line=lg.gen_first_line_deepspeare(last_word=name, contains_adjective=True, strict=False, search_space=1, seed=prompt)
		new_qua=[]
		for ii, q in enumerate(qua):
			if ii==0 or ii==3:
				new_qua.append(q+" .")
			else:
				new_qua.append(q+" ,")
		new_qua=[" ".join(first_line[0])]+qua
		new_quatrains.append(new_qua)
if not os.path.exists("deepspeare_limericks"):
	os.makedirs("deepspeare_limericks")
with open("deepspeare_limericks/deepspeare_limericks.txt","a+") as f:
	f.write("======================== new trial ==================================\n")
	for qua in new_quatrains:
		for line in qua:
			f.write(line+"\n")
		f.write("\n")
