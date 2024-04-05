
import torch
from torch.utils.data import Dataset
import sys
import pandas as pd
import os
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


class PatientDataset(Dataset):
  def __init__(self,root,csv_file):
    self.root = root
    self.csv_file = csv_file

    # read csv file
    self.csv_df = pd.read_excel(os.path.join(self.root,self.csv_file))
    self.csv_df = self.csv_df.drop([20,21,22,28],axis=0).reset_index(drop=True)
    
    self.subdir = ["Strokes","Healthies"]
    self.stroke_dir = [f for f in os.listdir(os.path.join(self.root,"Strokes")) if not f.startswith('.')]
    self.healthy_dir = [f for f in os.listdir(os.path.join(self.root,"Healthies")) if not f.startswith('.')]

    self.all_files = []
    for patient in self.stroke_dir:
      files = [os.path.join(self.root,"Strokes",patient,f) for f in os.listdir(os.path.join(self.root,"Strokes",patient)) if not f.startswith('.')]
      self.all_files += files
    
    for patient in self.healthy_dir:
      files = [os.path.join(self.root,"Healthies",patient,f)  for f in os.listdir(os.path.join(self.root,"Healthies",patient)) if not f.startswith('.')]
      self.all_files += files
    

    self.imp_joints= {'jRightShoulder':7,'jLeftShoulder':8,'jRightElbow':9,'jLeftElbow':11,'jRightWrist':12,'jLeftWrist':13,'jRightHip':14,'jLeftHip':18}

  def _get_task_files(self):
    tasks = defaultdict(list)

    for f in self.all_files:
      name = Path(f).stem.upper()
      name_split = name.split("_")
      p_id , t_id, h_id = name_split[0], name_split[1], name_split[2]
      tasks[t_id].append(f)
    
    return tasks


  def __len__(self):
    return len(self.all_files)

  def __getitem__(self,index): 
    return self._get_data_from_file(self.all_files[index])
  

  def _get_data_from_file(self,filename):
    _file = h5py.File(filename)
    filename = Path(filename).stem

    if filename.find("L") != -1 or filename.find("l") != -1: # moving left body part
      id = self.imp_joints["jLeftElbow"]

    else: # moving right body part
      id = self.imp_joints["jRightElbow"]

    x = self.MinMaxNorm(_file["jointAngle"][:,id*3])
    y = self.MinMaxNorm(_file["jointAngle"][:,id*3 + 1])
    z = x = self.MinMaxNorm(_file["jointAngle"][:,id*3 + 2])
    elbow_xyz = np.stack([x,y,z],axis=1)
    return elbow_xyz

  def MinMaxNorm(self,sequence):
    seq_len = len(sequence)
    _min = np.min(sequence)
    _max = np.max(sequence)

    sequence = (sequence - _min) / (_max - _min)
    return sequence



def TaskDataset(Dataset):
  def __init__(self,filename):
    data = np.load(filename)
    
    self.num_samples, self.sequence_len, self.features = data.shape

    
