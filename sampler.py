import numpy as np
import fmqa
import dimod
import os
import subprocess
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from typing import Tuple, Dict
import collections
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from itertools import groupby
from functools import reduce
import random
from modlamp.descriptors import GlobalDescriptor
from modlamp.descriptors import PeptideDescriptor

#Pareto front identification
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)
            is_efficient[i] = True
    return is_efficient

#Elimination of sequences with non-AA characters
def seq_fix():
    lines_d = open('.../MOQA/model_output/binary/decoded.txt','r')
    Lines_d = lines_d.readlines()
    seq_d=[]
    for i in Lines_d:
        seq_d.append(i.replace(" ", "").rstrip())

    matching_d1=[s for s in seq_d if "<pad>" in s]
    matching_d2=[s for s in seq_d if len(s)==0]
    matching_d3=[s for s in seq_d if "<unk>" in s]

    for ii in range(len(matching_d1)):
        for i in range(len(seq_d)):
            if seq_d[i]==matching_d1[ii]:
                seq_d[i]="SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    for ii in range(len(matching_d2)):
        for i in range(len(seq_d)):
            if seq_d[i]==matching_d2[ii]:
                seq_d[i]="SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    for ii in range(len(matching_d3)):
        for i in range(len(seq_d)):
            if seq_d[i]==matching_d3[ii]:
                seq_d[i]="SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    return seq_d

#Objective function evaluation
def objectives(seq_d):
    desc_charge_density = GlobalDescriptor(seq_d)
    desc_charge_density.charge_density()
    charge_density_together=desc_charge_density.descriptor
    charge_density_together=charge_density_together.ravel()

    desc_instability = GlobalDescriptor(seq_d)
    desc_instability.instability_index()
    instability_together=desc_instability.descriptor
    instability_together=instability_together.ravel()
    instability_together=-1.*instability_together

    desc_boman = GlobalDescriptor(seq_d)
    desc_boman.boman_index()
    boman_together=desc_boman.descriptor
    boman_together=boman_together.ravel()
    boman_together=-1.*boman_together
    
    return charge_density_together, instability_together, boman_together

#Hypervolume calculation
def ParetoV(charge_density_instability_boman):
    
    front_charge_density_instability_boman_0=is_pareto_efficient(charge_density_instability_boman)
    
    front_charge_density_instability_boman=[]
    front_index_charge_density_instability_boman_original=[]
    for i in range(len(front_charge_density_instability_boman_0)):
        if front_charge_density_instability_boman_0[i]== True:
            front_charge_density_instability_boman.append(charge_density_instability_boman[i])
            front_index_charge_density_instability_boman_original.append(i)

    temp0 = []
    for i in front_charge_density_instability_boman:
        temp0.append(i)
    
    x0 = torch.from_numpy(np.array(temp0))
    xref = torch.from_numpy(np.array([0.,0.,0.]))
    hv = Hypervolume(xref)
    pv0 = hv.compute(x0)
    
    oo = open('.../MOQA/model_output/binary/hv_score.txt','a')
    oo.write(str(pv0) + '\n')
    oo.close()

#Non-dominated sorting procedure with the set number of layers
def NSPareto(charge_density_instability_boman,charge_density_instability_boman_original,population_size,NLayers):

    SLayers=np.ones(population_size)*(-1)*10

    for ii in range(NLayers):
    
        front_charge_density_instability_boman_0=is_pareto_efficient(charge_density_instability_boman)
        
        front_charge_density_instability_boman=[]
        front_index_charge_density_instability_boman=[]
        for i in range(len(front_charge_density_instability_boman_0)):
            if front_charge_density_instability_boman_0[i]== True:
                front_charge_density_instability_boman.append(charge_density_instability_boman[i])
                front_index_charge_density_instability_boman.append(i)
    
        index_out=[]
        for i in front_charge_density_instability_boman:
            index = np.where(i == charge_density_instability_boman_original)
            index_out.append(index[0][0])
    
        for i in index_out:
            SLayers[i]=1./(ii+1.)
    
        charge_density_instability_boman = np.delete(charge_density_instability_boman, front_index_charge_density_instability_boman, axis=0)
    
    return SLayers


#Decode initial binary vector set
os.system('python3 decode_file.py experiment_configs/binary.json')

c0 = [x.split(' ')[:] for x in open('.../MOQA/model_output/binary/vectors.txt').readlines()]

vectors_all=[]
for k in range(len(c0)):
    vectors=[]
    for i in c0[k]:
        vectors.append(int(float(i)))
    vectors_all.append(vectors)

vectors_all=np.array(vectors_all)

#Get rid of wrong sequences
seq_d_all=seq_fix()

#Calculate objective functions
scores_all=objectives(seq_d_all)
scores_all_charge_density=scores_all[0]
scores_all_instability=scores_all[1]
scores_all_boman=scores_all[2]

#Output best metrics in the initial set
oo = open('.../MOQA/model_output/binary/all_points_best.txt','a')
oo.write(str(np.max(scores_all_charge_density)) + ' ' + str(np.min(-1.*scores_all_instability)) + ' ' + str(np.min(-1.*scores_all_boman)) + '\n')
oo.close()

charge_density_instability_boman=[]
for i in range(len(scores_all_charge_density)):
    charge_density_instability_boman.append([scores_all_charge_density[i],scores_all_instability[i],scores_all_boman[i]])

charge_density_instability_boman_original=np.array(charge_density_instability_boman)
charge_density_instability_boman=charge_density_instability_boman_original

NLayers=20 #define a number of non-dominated layers to consider

population_size=len(scores_all_charge_density)

#Calculate initial Pareto hypervolume
ParetoV(charge_density_instability_boman)

#Non-dominated sorting procedure
SLayers=NSPareto(charge_density_instability_boman,charge_density_instability_boman_original,population_size,NLayers)
scores=(-1.)*SLayers

#FM training
model = fmqa.FMBQM.from_data(vectors_all, scores)
#Simulated annealing for sampling
sampler = dimod.samplers.SimulatedAnnealingSampler()

#Sampling/Evaluation/Training
for _ in range(200):
    
    #Sample
    res = sampler.sample(model, num_reads=10)
    
    vectors_all = np.r_[vectors_all, res.record['sample']]
    vectors_sample = np.r_[res.record['sample']]
        
    vectors_sample_out=[]
    for i in vectors_sample:
        vectors_sample_out.append(''.join(str(i)[1:-1].splitlines()))
    
    os.system('rm .../MOQA/model_output/binary/vectors.txt')
    oo = open('.../MOQA/model_output/binary/vectors.txt','a')
    for i in vectors_sample_out:
        oo.write(str(i) + '\n')
    oo.close()

    os.system('rm .../MOQA/model_output/binary/decoded.txt')
    os.system('python .../MOQA/decode_file.py experiment_configs/binary.json')
    seq_d=seq_fix()
    
    seq_d_all = np.r_[seq_d_all, seq_d]
    
    #Calculate objective functions for sampled sequences
    scores_sample=objectives(seq_d)
    
    scores_sample_charge_density=scores_sample[0]
    scores_all_charge_density = np.r_[scores_all_charge_density, scores_sample_charge_density]
    
    scores_sample_instability=scores_sample[1]
    scores_all_instability = np.r_[scores_all_instability, scores_sample_instability]
    
    scores_sample_boman=scores_sample[2]
    scores_all_boman = np.r_[scores_all_boman, scores_sample_boman]

    charge_density_instability_boman=[]
    for i in range(len(scores_all_charge_density)):
        charge_density_instability_boman.append([scores_all_charge_density[i],scores_all_instability[i],scores_all_boman[i]])

    charge_density_instability_boman_original=np.array(charge_density_instability_boman)
    charge_density_instability_boman=charge_density_instability_boman_original
    
    #Calculate Pareto hypervolume for the updated population
    ParetoV(charge_density_instability_boman)
    
    #Non-dominated sorting procedure with simulated annealing for the layer number reduction
    #Geometric annealing schedule
    STEP_GEOM=0.982
    ii= _
    if ii==0:
        NLayerss=20.
    else:
        NLayerss=STEP_GEOM*NLayerss
    
    NLayers=round(NLayerss)
    population_size=len(scores_all_charge_density)
    
    SLayers=NSPareto(charge_density_instability_boman,charge_density_instability_boman_original,population_size,NLayers)
    scores_all=(-1.)*SLayers

    #Output best metrics for the updated population
    oo = open('.../MOQA/model_output/binary/all_points_best.txt','a')
    oo.write(str(np.max(scores_all_charge_density)) + ' ' + str(np.min(-1.*scores_all_instability)) + ' ' + str(np.min(-1.*scores_all_boman)) + '\n')
    oo.close()

    #FM re-training
    model.train(vectors_all, scores_all)
