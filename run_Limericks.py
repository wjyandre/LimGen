"""
Author:         Jianyou (Andre) Wang
Date:           Sep 2020
"""


import tensorflow as tf
import nltk
import pdb
import os
import pickle
import numpy as np
from collections import defaultdict
import heapq
import random
import argparse
from LimGen.Limericks import LimGen

def init_parser():
    parser = argparse.ArgumentParser(description='Evaluate which epoch')
    parser.add_argument('--prompt', '-p', default='love', type=str, dest='prompt', help= 'prompt word')
    parser.add_argument("--saved_directory",'-dir', default='testing',type=str,dest='saved_directory')
    parser.add_argument("--search_space",'-ser', default=5,type=int,dest='search_space', help = 'beam size for each template')
    parser.add_argument("--retain_space",'-re', default=3,type=int,dest='retain_space', help ='max number of templates to retain')
    parser.add_argument("--mode",'-m', default='multi',type=str,dest='mode', help ='multi mode enalbles AMTC, adaptive multi-templated constraints')
    parser.add_argument("--massproduce",'-mas', default=False,type=bool,dest='massproduce', help= 'if true, will generate poems for multiple prompts, if false, generate for one specific prompt, see __main__')
    parser.add_argument("--multiprocessing",'-mp', default=True,type=bool,dest='multiprocessing', help = 'enable multiprocessing')
    parser.add_argument("--gender",'-g',default="female",type=str,dest='gender', help = 'specify the gender of the protagonist in the poem')
    parser.add_argument("--beam_search",'-bs',default="MTBS",type=str,dest='beam_search', help ="MTBS is for multi-templated beam search, candidate_rank is for candidate rank beam search")
    parser.add_argument("--storyline",'-sl',default="story",type=str,dest='storyline', help = "enable storyline algorithms")
    return parser.parse_args()

def printing(data, f, f_final, f_final_best, template_to_line,words_to_names_rhyme_dict,f_all,f_single_all,prompt):
	'''
	The printing function will print poems out in nice format. 
	'''
	try:
		with open(f_final+".pickle","rb") as pickle_in:
			data_old=pickle.load(pickle_in)
	except:
		with open(f_final+".pickle","wb") as pickle_in:
			data_old={"score":[]}
			pickle.dump(data_old,pickle_in)
	try:
		with open(f_final_best+".pickle","rb") as pickle_in:
			data_old_best=pickle.load(pickle_in)
	except:
		with open(f_final_best+".pickle","wb") as pickle_in:
			data_old_best={"score":[]}
			pickle.dump(data_old_best,pickle_in)
	data_curr_score=[]
	temp_data=defaultdict(list)
	for line in data:
		temp_data[" ".join(line[3])].append(line)

	for t,k in enumerate(temp_data.keys()):
		lines=[]
		num_of_words_each_line=[0]
		for pp in temp_data[k]:
			count=0
			for ppp in pp[3]:
				if ppp=="\n":
					count+=1
					num_of_words_each_line.append(0)
				else:
					num_of_words_each_line[count]+=1
			break
		num_of_words_each_line=num_of_words_each_line[1:-1]
		for i in k.split("\n")[1:]:
			i=i.strip()
			if len(i)!=0:
				i_list=i.split(" ")
				try:
					line=list(template_to_line[" ".join(i_list)][0])+["\n"]
				except:
					line=list(template_to_line[" ".join(i_list[:-1])][0])+["\n"]
				lines+=line

		f.write("======================= template: {} ============================  \n".format(t+1))
		f.write(k)
		f.write("----------------------- original sentences ------------------------------------ \n")
		f.write(" ".join(lines))
		for jj,j in enumerate(temp_data[k]):
			if jj==1: break
			score=np.mean(j[1])
			data_curr_score.append(score)
			f.write("-------------------------score:  {}----------------------- \n".format(score))
			limerick=list(j[2])
			limerick[limerick.index("\n")-1]=random.choice(words_to_names_rhyme_dict[j[4][0]])
			if jj<1:
				f_all.write("{}:{}".format(prompt,score)+"\n")
				f_all.write(" ".join(limerick)+"\n")
			if jj<1 and t==0:
				f_single_all.write("{}:{}".format(prompt,score)+"\n")
				f_single_all.write(" ".join(limerick)+"\n")
			f.write(" ".join(limerick))
			f.write("------------------------- score breakdown ------------------------ \n")
			count_w=j[2].index("\n")+1
			count_s=1
			for s in range(4):
				temp_list=[]
				for ww,w in enumerate(j[2][count_w:count_w+num_of_words_each_line[s]]):
					f.write("({} {:03.2f})".format(w,j[1][count_s+ww]))
					temp_list.append(j[1][count_s+ww])
				count_s+=ww
				count_w+=ww+2
				f.write(" line score is : {:04.03f}, look ahead score is : {:04.03f}".format(np.mean(temp_list),j[5][s]))
				f.write("\n")
	data_old_best_score=data_old_best["score"]
	data_curr_best_score=heapq.nlargest(min(len(data_curr_score),5), data_curr_score, key=lambda x: x)
	data_curr_best_score+=data_old_best_score
	data_curr_best={"score":data_curr_best_score}
	data_old_score=data_old["score"]
	data_curr_score+=data_old_score
	data_curr={}
	data_curr["score"]=data_curr_score
	with open(f_final+".pickle","wb") as pickle_in:
		pickle.dump(data_curr,pickle_in)
	with open(f_final_best+".pickle","wb") as pickle_in:
		pickle.dump(data_curr_best,pickle_in)
	return len(temp_data.keys())
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',prompt="blood",args=None):
	if not args.massproduce:
		prompt=args.prompt
	gender=args.gender
	beam_search=args.beam_search
	saved_directory=args.saved_directory
	search_space=args.search_space
	retain_space=args.retain_space
	multiprocessing=args.multiprocessing
	mode=args.mode
	storyline=args.storyline
	lg = LimGen(gender=gender,prompt=prompt,search_space=search_space, retain_space=retain_space, mode=mode, beam_search=beam_search, multiprocessing=multiprocessing, storyline=storyline)
	saved_directory=saved_directory
	f_final=saved_directory +"/"+"results_"+str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)
	f_final_best=saved_directory +"/"+"best_results_"+str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)
	f_single_all_path=saved_directory +"/"+"single_best_results_"+str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)
	f1_path=saved_directory+"/"+"success.txt"
	f2_path=saved_directory+"/"+"success.pickle"
	f3_path=saved_directory+"/"+"diversity.pickle"
	if saved_directory not in os.listdir(os.getcwd()):
		os.mkdir(saved_directory)
	result_file_path = saved_directory +"/"+ prompt+"_" + gender + '_' +str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)+"_"+str(beam_search)
	all_result_file_path=saved_directory +"/" + str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)+"_"+str(beam_search)
	previous_data, template_to_line,words_to_names_rhyme_dict=lg.gen_poem()
	with open(result_file_path+".pickle","wb") as f3:
		pickle.dump(previous_data,f3)
	#with open(result_file_path+"template_to_line"+".pickle","wb") as f4:
		#pickle.dump(template_to_line,f4)
	with open(result_file_path+".txt","a+") as f:
		with open(all_result_file_path+".txt","a+") as f_all:
			with open(f_single_all_path+".txt","a+") as f_single_all:
				div=printing(previous_data,f, f_final,f_final_best, template_to_line, words_to_names_rhyme_dict,f_all,f_single_all,prompt)
	if len(previous_data)>0:
		with open(f1_path,"a+") as f1:
			f1.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(mode)+"_"+str(storyline)+"_"+str(beam_search)+"and diversity is {}".format(div) +"\n")
		'''
		try:
			with open(f2_path,"rb") as f2:
				data=pickle.load(f2)
				data.append(prompt)
			with open(f2_path,"wb") as f2:
				pickle.dump(data,f2)
		except:
			with open(f2_path,"wb") as f2:
				pickle.dump([],f2)
		'''
		if os.path.exists(f2_path):
			with open(f2_path,"rb") as f2:
				data=pickle.load(f2)
				data.append(prompt)
			with open(f2_path,"wb") as f2:
				pickle.dump(data, f2)
		else:
			with open(f2_path,"wb") as f2:
				pickle.dump([prompt], f2)
		if os.path.exists(f3_path):
			with open(f3_path,"rb") as f3:
				data=pickle.load(f3)
				data.append(div)
			with open(f3_path,"wb") as f3:
				pickle.dump(data, f3)
		else:
			with open(f3_path,"wb") as f3:
				pickle.dump([div], f3)
	
if __name__ == '__main__':
	# the code below is for mass producing limericks with different prompt word for SLURM.
	
	data1="born, shaken, restore, laugh, tears, surprise, kindness, humiliation, victory, wedding, alien, holiday, christmas, thanksgiving, birthday, injury, pillow, fiance, dawn, traffic, heartbreak, wine, beer, musuem, mountain, river, memory, mud, spider, rain, season, winter, throne, politics, promise, beach, bank, money, limerick"
	data2="love, cunning, dog, blood, death, war, disease, world, planet, fire, water, sports, love, car, animal, violent, opera, monster, library, market, noble, doctor, funeral, ball, body, smart, exercise, gun, art, music, boxing, forest, philosophy, night, scary, creativity, evil, angry, pride, law, school, light, rich, color, leader, park, airplane, loss, weight, useful, applaud, home, union, child, working, cheat, fall, time, hope, flower, random, impressive"
	prompt_list=list(data1.split(", ")+data2.split(", "))
	slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
	prompt=prompt_list[slurm_task_id]
	limericks_generation_gpt(prompt=prompt, args=init_parser())


	# uncomment the line below, and comment out the lines above if wish to generate limerick for a single prompt word. You will need to specify -prompt [user input] in command line, or in shell file.
	#limericks_generation_gpt(args=init_parser())
	
	
