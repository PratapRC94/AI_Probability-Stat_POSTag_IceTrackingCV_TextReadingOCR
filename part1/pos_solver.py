###################################
# CS B551 Fall 2021, Assignment #3
#
# Your names and user ids: Pratap Roy Choudhury[prroyc]-Tanu kansal[takansal]-Parth Ravindra Rao[partrao]
#
# (Based on skeleton code by D. Crandall)
#
# MCMC reading and implementation references::::::::
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9427263
# http://hua-zhou.github.io/teaching/st758-2014fall/ST758-2014-Fall-Pre-LecNotes.pdf

import random
import math
import pandas as pd
import numpy as np
from collections import Counter,defaultdict


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # creating list variables for 12 POS - initial tags and final tags
    initial_states = [0]*12
    final_states = [0]*12
    # dictionary to store emission count and emission probability
    emission_count={}
    emission_probability={}
    # variable to store all the 12 POS tags
    pos_types = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
    # list variables to store the transition count and transition probability
    transition_count=[[0 ]*12 for i in range(12)]
    transition_probability=[[0]*12 for i in range(12)]
    # initialized value to the word occuring first time for the initial state probability
    count = 10**(-20) #float(1/100000000000000000)
    count1 = 0
    # generating 1000 samples for complex MCMC
    kKl = 1000
    # table for transitional probability calculation of Gibbs sampling in order to sample values from the posterior distribution
    trans_s1_to_sn=[[0]*12 for i in range(12)]
    trans_si_to_sn=[[0]*12 for i in range(12)]
    list_of_mcmc=[]
 
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):

        if model == "Simple":
            # return the sum of all log posterior probabilities from simplified model 
            p=self.simplified(sentence)[1]
            return(sum(p))

        elif model == "Complex":
            # calculate the posterior probability of a POS tag given a word as emission count * emission probability * transition probability
            posterior_prob = 0.0
            for i in range(len(sentence)):
                if sentence[i] in self.emission_probability:
                    p = self.emission_count[sentence[i]][Solver.pos_types.index(label[i])] * self.emission_probability[sentence[i]][Solver.pos_types.index(label[i])]
                    posterior_prob += math.log(p)
                else:
                    posterior_prob += math.log(1e-20)
            for i in range(len(label)-1):
                p = self.transition_probability[Solver.pos_types.index(label[i])][Solver.pos_types.index(label[i+1])]
                posterior_prob += math.log(p)        
            return posterior_prob

        elif model == "HMM":
            # return the log posterior probability from hmm_viterbi model as it already calculates the max posterior
            return(self.hmm_viterbi(sentence)[1])
        else:
            print("Unknown algo!")


    # Do the training!
    #
    def train(self, data):
        initial_count=[0]*12
        first_word_count=[0]*12
        last_word_count=[0]*12
        self.total_words_count=0
        total_sentences_count=0
        self.prior_count=[]
        dict_1={}
        self.emission_count=self.emission_probability={j:[Solver.count]*12 for i in data for j in i[0]}
        for i in range(len(data)):
            self.total_words_count+=len(data[i][0])
            total_sentences_count+=1
            # first and last word of a sentence count to generate the initial tag probabilities
            first_word_count[Solver.pos_types.index(data[i][1][0])]+=1
            last_word_count[Solver.pos_types.index(data[i][1][-1])]+=1
            for j in range(len(data[i][1])):
                index=Solver.pos_types.index(data[i][1][j])
                initial_count[index]+=1
                #Counts used for calculating transition probability
                if(j<len(data[i][1])-1):
                    Solver.transition_count[index][Solver.pos_types.index(data[i][1][j+1])]+=1
                #Counts used for calculating emission probability
                self.emission_count[data[i][0][j]][index]+=1


        self.prior_probability=[]
        for i in range(12):
            #Initial Probability distribution
            if(first_word_count[i]==0):
                Solver.initial_states[i]=Solver.count
            else:
                Solver.initial_states[i]=float(first_word_count[i])/len(data)
            self.prior_probability.append(float(initial_count[i])/self.total_words_count)

            if(last_word_count[i]==0):
                Solver.final_states[i]=Solver.count
            else:
                Solver.final_states[i]=float(last_word_count[i]/len(data))
            #Calculating transition probabilities
            for j in range(12):
                if(sum(Solver.transition_count[i])==0 or Solver.transition_count[i][j]==0):
                    Solver.transition_probability[i][j]=Solver.count
                else:
                    Solver.transition_probability[i][j]=float(Solver.transition_count[i][j])/sum(Solver.transition_count[i])


        # calculate the emission probability
        for i in self.emission_probability:
            for j in range(12):
                if(initial_count[j]==0 or self.emission_count[i][j]==0):
                    self.emission_probability[i][j]=Solver.count
                else:
                    self.emission_probability[i][j]=float(self.emission_count[i][j])/initial_count[j]


        dict_1=dict_2={j:[Solver.count1]*12 for j in Solver.pos_types}
       
        for i in range(len(Solver.pos_types)):
            dict_1={j:[Solver.count1]*12 for j in Solver.pos_types}
            for k in range(len(data)):
                if data[k][1][len(data[k][0])-1]==Solver.pos_types[i]:
                    dict_1[data[k][1][0]][Solver.pos_types.index(data[k][1][len(data[k][0])-2])]+=1
            Solver.list_of_mcmc.append(dict_1)
      

        for k in range(len(data)):
            dict_2[data[k][1][0]][Solver.pos_types.index(data[k][1][len(data[k][0])-2])]+=1
        for i in range(0,len(Solver.list_of_mcmc)):
            for j in Solver.pos_types:
                for k in range(0,12):
                    if Solver.list_of_mcmc[i][j][k]==0:
                        Solver.list_of_mcmc[i][j][k]=float(Solver.count)
                    else:
                        Solver.list_of_mcmc[i][j][k]=float(Solver.list_of_mcmc[i][j][k]/dict_2[j][k])

        # generate transition probability table for Gibbs sampling
        for i in range(len(data)):
            Solver.trans_s1_to_sn[Solver.pos_types.index(data[i][1][0])][Solver.pos_types.index(data[i][1][len(data[i][0])-1])]+=1
            Solver.trans_si_to_sn[Solver.pos_types.index(data[i][1][len(data[i][0])-2])][Solver.pos_types.index(data[i][1][len(data[i][0])-1])]+=1
        for i in range(12):
            for j in range(12):
                if(sum(Solver.trans_s1_to_sn[i])==0 or Solver.trans_s1_to_sn[i][j]==0):
                    Solver.trans_s1_to_sn[i][j]=Solver.count
                else:
                    Solver.trans_s1_to_sn[i][j]=float(Solver.trans_s1_to_sn[i][j])/sum(Solver.trans_s1_to_sn[i])
                if(sum(Solver.trans_si_to_sn[i])==0 or Solver.trans_si_to_sn[i][j]==0):
                    Solver.trans_si_to_sn[i][j]=Solver.count
                else:
                    Solver.trans_si_to_sn[i][j]=float(Solver.trans_si_to_sn[i][j])/sum(Solver.trans_si_to_sn[i])



    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        
        pos = []
        posterior_probability = []
        for i in range(len(sentence)):
            self.most_probable_pos_proba=[0]*12
            if(sentence[i] not in self.emission_probability):
                # If a word occuring first time in the sentence, give high probability for the word to be Noun
                self.emission_probability[sentence[i]]=[Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,1-(Solver.count)*11,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count]
            for j in range(12):
                # calculate the probabilities of all probable pos
                self.most_probable_pos_proba[j]= math.log(self.emission_probability[sentence[i]][j])+math.log(self.prior_probability[j])
            # get the most probable pos which has the maximum probability
            pos.append(Solver.pos_types[self.most_probable_pos_proba.index(max(self.most_probable_pos_proba))])
            posterior_probability.append(max(self.most_probable_pos_proba))

        return  (pos,posterior_probability)

    
    def hmm_viterbi(self, sentence):
     
        L = len(sentence)
        # Viterbi table will have L rows for all words in a sentence and 12 columns for pos
        v_table = [[0 for i in range(12)]for l in range(L)]
        # table to track which viterbi of previous pos maximises the current viterbi
        max_decision_table = [[0 for l in range(L)]for j in range(12)]

        for l in range(L):
            for j in range(12):
                if l==0: # initial viterbi for each state
                    v_table[l][j] = math.log(Solver.initial_states[j]) + math.log(self.emission_probability[sentence[l]][j])
                else:
                    cost = [v_table[l - 1][i]+math.log(Solver.transition_probability[i][j] )for i in range(12)]
                    # select which viterbi state is maximum
                    max_cost = max(cost)
                    v_table[l][j] = math.log(self.emission_probability[sentence[l]][j]) + max_cost
                    max_decision_table[j][l] = cost.index(max_cost)
                    
        probable_states = []
        idx_max = v_table[L - 1].index(max(v_table[L - 1]))
        probable_states.append(Solver.pos_types[idx_max])
        # backtrack the max_decision_table to pick the maximum probable pos tag
        for s in range(L-1,0,-1):
            idx_max = max_decision_table[idx_max][s]
            probable_states.append(Solver.pos_types[idx_max])
        
        return (probable_states[::-1],max(v_table[L - 1]))

        
    
    def complex_mcmc(self, sentence):
        
        L = len(sentence)
        # initialize the sample pos tag as Noun
        initial_sample = ["noun"] * L
        top_samples = []
        posterior = []
        for noi in range(Solver.kKl): # 1000 sample generation
            for i in range(0,L):
                prob_val = []
                prob_sum = 0
                for j in Solver.pos_types:
                    if len(sentence) > 2:

                        if i==0:
                            init_prob = Solver.initial_states[Solver.pos_types.index(j)]
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            trans_prob=self.transition_probability[Solver.pos_types.index(j)][Solver.pos_types.index(initial_sample[i+1])]
                            mcmc_prob=self.list_of_mcmc[Solver.pos_types.index(initial_sample[L-1])][j][Solver.pos_types.index(initial_sample[L-2])]
                            p=float(init_prob*em_prob*trans_prob*mcmc_prob)

                        if i >0 and i <L-2:
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            trans_prob=self.transition_probability[Solver.pos_types.index(j)][Solver.pos_types.index(initial_sample[i+1])]
                            trans_init_prob=self.transition_probability[Solver.pos_types.index(initial_sample[i-1])][Solver.pos_types.index(j)]
                            p=float(em_prob*trans_prob*trans_init_prob)
                        if i==L-1:
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            mcmc_prob=self.list_of_mcmc[Solver.pos_types.index(j)][initial_sample[0]][Solver.pos_types.index(initial_sample[L-2])]
                            p=float(em_prob*mcmc_prob)
                        if i== L-2:
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            mcmc_prob=self.list_of_mcmc[Solver.pos_types.index(initial_sample[L-1])][initial_sample[0]][Solver.pos_types.index(j)]
                            trans_init_prob=self.transition_probability[Solver.pos_types.index(initial_sample[L-3])][Solver.pos_types.index(j)]
                            p=float(em_prob*mcmc_prob*trans_init_prob)

                    if len(sentence) == 1:
                        init_prob = Solver.initial_states[Solver.pos_types.index(j)] 
                        em_prob = self.emission_probability[sentence[i]][Solver.pos_types.index(j)] 
                        p=float(init_prob*em_prob)
                        
                    if len(sentence) == 2:
                        if i==0:
                            init_prob = Solver.initial_states[Solver.pos_types.index(j)]
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            trans_prob=self.transition_probability[Solver.pos_types.index(j)][Solver.pos_types.index(initial_sample[i+1])]
                            p=float(init_prob*em_prob*trans_prob)
                        if i==1:
                            trans_prob=self.transition_probability[Solver.pos_types.index(initial_sample[i-1])][Solver.pos_types.index(j)]
                            em_prob=self.emission_probability[sentence[i]][Solver.pos_types.index(j)]
                            p=float(trans_prob*em_prob)
                            
                    prob_sum += p
                    prob_val.append(p)

                # end of for loop j (for 12 pos)
                
                cumulative_prob=0
                random_bias = random.uniform(0.00,1.00)

                for w in range(0, len(prob_val)):
                    prob_val[w] =(prob_val[w]/prob_sum)
                    cumulative_prob += prob_val[w]
                    prob_val[w] = cumulative_prob
                    if random_bias < prob_val[w]:
                        pos_index = w
                        break
                initial_sample[i]=Solver.pos_types[pos_index]
                
            # end of for loop i (lenght of sentence)
            top_samples.append(initial_sample)
            
        # end of kKl loop

        posterior = list(Counter(col).most_common(1)[0][0] for col in zip(*top_samples))
        
        return(posterior)

    
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)[0]
        elif model == "HMM":
            return self.hmm_viterbi(sentence)[0]
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

    

