import numpy as np
import fmqa
import dimod
import greedy
import os
import subprocess
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system import LeapHybridSampler
from typing import Tuple, Dict
import collections
from pymoo.factory import get_performance_indicator
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

#Decode initial data set
    
os.system('python3 decode_file.py experiment_configs/binary.json')


c0 = [x.split(' ')[:] for x in open('.../MOQA/model_output/binary/vectors.txt').readlines()]

vectors_all=[]
for k in range(len(c0)):
    vectors=[]
    for i in c0[k]:
        vectors.append(int(float(i)))
    vectors_all.append(vectors)

lines_d = open('.../MOQA/model_output/binary/decoded.txt','r')
Lines_d = lines_d.readlines()
seq_d=[]
for i in Lines_d:
    seq_d.append(i.replace(" ", "").rstrip())

#Assigning dummy names to unrealistic sequences

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

seq_d_all=seq_d

vectors_all=np.array(vectors_all)

#Calculating objective functions for optimization

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

scores_all_charge_density=charge_density_together
scores_all_instability=instability_together
scores_all_boman=boman_together

oo = open('.../MOQA/model_output/binary/all_points_best.txt','a')
oo.write(str(np.max(scores_all_charge_density)) + ' ' + str(np.min(-1.*scores_all_instability)) + ' ' + str(np.min(-1.*scores_all_boman)) + '\n')
oo.close()

#Pareto front estimation and evaluation

charge_density_instability_boman=[]
for i in range(len(scores_all_charge_density)):
    charge_density_instability_boman.append([scores_all_charge_density[i],scores_all_instability[i],scores_all_boman[i]])

charge_density_instability_boman_original=np.array(charge_density_instability_boman)
charge_density_instability_boman=charge_density_instability_boman_original

xref = torch.from_numpy(np.array([0.,0.,0.]))
hv = Hypervolume(xref)

NLayers=20 #number of non-dominated layers to consider

#Non-dominated sorting procedure

SLayers=np.ones(len(scores_all_charge_density))*(-1)*10

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

#FM training

scores=(-1.)*SLayers

model = fmqa.FMBQM.from_data(vectors_all, scores)

sampler = dimod.samplers.SimulatedAnnealingSampler()

#Sampling/Evaluation/Training

for _ in range(200):
    
    #Sampling
    
    res = sampler.sample(model, num_reads=10)
    
    vectors_all = np.r_[vectors_all, res.record['sample']]
    
    #Decode sampled binary vectors
    
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
    
    os.system('scp .../MOQA/model_output/binary/decoded.txt .../MOQA/model_output/binary/decoded_%d.txt' % (_))
    
    lines_d = open('.../MOQA/model_output/binary/decoded.txt','r')
    Lines_d = lines_d.readlines()
    seq_d=[]
    for i in Lines_d:
        seq_d.append(i.replace(" ", "").rstrip())
    
    #Assigning dummy names to unrealistic sequences
    
    matching_d1=[]
    matching_d2=[]
    matching_d3=[]
    matching_d=[]
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
    
    seq_d_all = np.r_[seq_d_all, seq_d]
    
    #Calculating objective functions for optimization
    
    desc_charge_density = GlobalDescriptor(seq_d)
    desc_charge_density.charge_density()
    charge_density_sample=desc_charge_density.descriptor
    charge_density_sample=charge_density_sample.ravel()

    desc_instability = GlobalDescriptor(seq_d)
    desc_instability.instability_index()
    instability_sample=desc_instability.descriptor
    instability_sample=instability_sample.ravel()
    instability_sample=-1.*instability_sample
    
    desc_boman = GlobalDescriptor(seq_d)
    desc_boman.boman_index()
    boman_sample=desc_boman.descriptor
    boman_sample=boman_sample.ravel()
    boman_sample=-1.*boman_sample
    
    scores_sample_charge_density=charge_density_sample
    scores_all_charge_density = np.r_[scores_all_charge_density, scores_sample_charge_density]
    
    scores_sample_instability=instability_sample
    scores_all_instability = np.r_[scores_all_instability, scores_sample_instability]
    
    scores_sample_boman=boman_sample
    scores_all_boman = np.r_[scores_all_boman, scores_sample_boman]
    
    #Pareto front estimation and evaluation

    charge_density_instability_boman=[]
    for i in range(len(scores_all_charge_density)):
        charge_density_instability_boman.append([scores_all_charge_density[i],scores_all_instability[i],scores_all_boman[i]])

    charge_density_instability_boman_original=np.array(charge_density_instability_boman)
    charge_density_instability_boman=charge_density_instability_boman_original
    
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
    pv0 = hv.compute(x0)
    
    #Non-dominated sorting procedure with simulated annealing
    
    #Geometric schedule
    STEP_GEOM=0.982
    ii= _
    if ii==0:
        NLayerss=20.
    else:
        NLayerss=STEP_GEOM*NLayerss
    
    NLayers=round(NLayerss)
    
    SLayers=np.ones(len(scores_all_charge_density))*(-1)*10
    front_charge_density_instability_boman_0=[]
    
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

    
    scores_all=(-1.)*SLayers

    #Output
        
    oo = open('.../MOQA/model_output/binary/hv_score.txt','a')
    oo.write(str(pv0) + '\n')
    oo.close()
        
    oo = open('.../MOQA/model_output/binary/pareto_front.txt','a')
    for i in front_index_charge_density_instability_boman_original:
        oo.write(str(scores_all_charge_density[i]) + ' ' + str(-1.*scores_all_instability[i]) + ' ' + str(-1.*scores_all_boman[i]) + '\n')
    oo.close()
    
    os.system('scp .../MOQA/model_output/binary/pareto_front.txt .../MOQA/model_output/binary/pareto_front_%d.txt' % (_))
    
    os.system('rm .../MOQA/model_output/binary/pareto_front.txt')

    
    oo = open('.../MOQA/model_output/binary/pareto_seq.txt','a')
    for i in front_index_charge_density_instability_boman_original:
        oo.write(str(seq_d_all[i]) +  '\n')
    oo.close()

    os.system('scp .../MOQA/model_output/binary/pareto_seq.txt .../MOQA/model_output/binary/pareto_seq_%d.txt' % (_))
    
    os.system('rm .../MOQA/model_output/binary/pareto_seq.txt')

    oo = open('.../MOQA/model_output/binary/all_points_best.txt','a')
    oo.write(str(np.max(scores_all_charge_density)) + ' ' + str(np.min(-1.*scores_all_instability)) + ' ' + str(np.min(-1.*scores_all_boman)) + '\n')
    oo.close()

    #FM re-training

    model.train(vectors_all, scores_all)
