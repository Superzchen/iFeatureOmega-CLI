#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re
from networkx.algorithms import cluster
from networkx.algorithms.tree.recognition import is_tree
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
sys.path.append(os.path.join(pPath, 'chem'))
import json
import random
import math
import cmath
import pickle
import itertools
import numpy as np
import networkx as nx
import pandas as pd
import warnings
from collections import Counter
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.HSExposure import HSExposureCA, HSExposureCB
from Bio.PDB.PDBList import PDBList
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN,
                             AgglomerativeClustering, SpectralClustering, MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from rdkit import Chem
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from chem import *
import copy

plt.style.use('ggplot')
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Sequence(object):
    def __init__(self, file):
        self.file = file                              # whole file path
        self.fasta_list = []                          # 2-D list [sampleName, fragment, label, training or testing]
        self.sample_purpose = None                    # 1-D ndarray, sample used as training dataset (True) or testing dataset(False)
        self.sequence_number = 0                      # int: the number of samples
        self.sequence_type = ''                       # DNA, RNA or Protein
        self.is_equal = False                         # bool: sequence with equal length?
        self.minimum_length = 1                       # int
        self.maximum_length = 0                       # int
        self.minimum_length_without_minus = 1         # int
        self.maximum_length_without_minus = 0         # int
        self.error_msg = ''                           # string

        self.fasta_list, self.sample_purpose, self.error_msg = self.read_fasta(self.file)
        self.sequence_number = len(self.fasta_list)

        if self.sequence_number > 0:
            self.is_equal, self.minimum_length, self.maximum_length, self.minimum_length_without_minus, self.maximum_length_without_minus = self.sequence_with_equal_length()
            self.sequence_type = self.check_sequence_type()
        else:
            self.error_msg = 'File format error.'

    def read_fasta(self, file):
        """
        read fasta sequence
        :param file:
        :return:
        """
        msg = ''
        if not os.path.exists(self.file):
            msg = 'Error: file %s does not exist.' % self.file
            return [], None, msg
        with open(file) as f:
            records = f.read()
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWY-]', '-', ''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            label = header_array[1] if len(header_array) >= 2 else '0'
            label_train = header_array[2] if len(header_array) >= 3 else 'training'
            fasta_sequences.append([name, sequence, label, label_train])
        sample_purpose = np.array([item[3] == 'training' for item in fasta_sequences])
        return fasta_sequences, sample_purpose, msg

    def sequence_with_equal_length(self):
        """
        Check if fasta sequence is in equal length
        :return:
        """
        length_set = set()
        length_set_1 = set()
        for item in self.fasta_list:
            length_set.add(len(item[1]))
            length_set_1.add(len(re.sub('-', '', item[1])))

        length_set = sorted(length_set)
        length_set_1 = sorted(length_set_1)
        if len(length_set) == 1:
            return True, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]
        else:
            return False, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]

    def check_sequence_type(self):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return:
        """
        tmp_fasta_list = []
        if len(self.fasta_list) < 100:
            tmp_fasta_list = self.fasta_list
        else:
            random_index = random.sample(range(0, len(self.fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(self.fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item[1]

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in self.fasta_list:
                line[1] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line[1])
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('U', 'T', line[1])
            return 'RNA'
        else:
            return 'Unknown'

class iProtein(Sequence):
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> protein = iFeatureOmegaCLI.iProtein("./data_examples/peptide_sequences.txt")

    # display available feature descriptor methods
    >>> protein.display_feature_types()

    # import parameters for feature descriptors (optimal)
    >>> protein.import_parameters('parameters/Protein_parameters_setting.json')

    # calculate feature descriptors. Take "AAC" as an example.
    >>> protein.get_descriptor("AAC")

    # display the feature descriptors
    >>> print(protein.encodings)

    # save feature descriptors
    >>> protein.to_csv("AAC.csv", "index=False", header=False)
    """

    def __init__(self, file):
        super(iProtein, self).__init__(file=file)
        self.__default_para_dict = {
            'EAAC': {'sliding_window': 5},
            'CKSAAP': {'kspace': 3},
            'EGAAC': {'sliding_window': 5},
            'CKSAAGP': {'kspace': 3},
            'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
            'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'KSCTriad': {'kspace': 3},
            'SOCNumber': {'nlag': 3},
            'QSOrder': {'nlag': 3, 'weight': 0.05},
            'PAAC': {'weight': 0.05, 'lambdaValue': 3},
            'APAAC': {'weight': 0.05, 'lambdaValue': 3},
            'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
            'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 3},
            'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 4},
            'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 4},
            'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},            
        }
        self.__default_para = {
            'sliding_window': 5,
            'kspace': 3,            
            'nlag': 3,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 
        }
        self.encodings = None       # pandas dataframe
        self.__cmd_dict ={
            'AAC': 'self._AAC()',
            'EAAC': 'self._EAAC()',
            'CKSAAP': 'self._CKSAAP()',
            'DPC': 'self._DPC()',
            'DDE': 'self._DDE()',
            'TPC': 'self._TPC()',
            'binary': 'self._binary()',
            'binary_6bit': 'self._binary_6bit()',
            'binary_5bit type 1': 'self._binary_5bit_type_1()',
            'binary_5bit type 2': 'self._binary_5bit_type_2()',
            'binary_3bit type 1': 'self._binary_3bit_type_1()',
            'binary_3bit type 2': 'self._binary_3bit_type_2()',
            'binary_3bit type 3': 'self._binary_3bit_type_3()',
            'binary_3bit type 4': 'self._binary_3bit_type_4()',
            'binary_3bit type 5': 'self._binary_3bit_type_5()',
            'binary_3bit type 6': 'self._binary_3bit_type_6()',
            'binary_3bit type 7': 'self._binary_3bit_type_7()',
            'AESNN3': 'self._AESNN3()',
            'GAAC': 'self._GAAC()',
            'EGAAC': 'self._EGAAC()',
            'CKSAAGP': 'self._CKSAAGP()',
            'GDPC': 'self._GDPC()',
            'GTPC': 'self._GTPC()',
            'AAIndex': 'self._AAIndex()',
            'ZScale': 'self._ZScale()',
            'BLOSUM62': 'self._BLOSUM62()',
            'NMBroto': 'self._NMBroto()',
            'Moran': 'self._Moran()',
            'Geary': 'self._Geary()',
            'CTDC': 'self._CTDC()',
            'CTDT': 'self._CTDT()',
            'CTDD': 'self._CTDD()',
            'CTriad': 'self._CTriad()',
            'KSCTriad': 'self._KSCTriad()',
            'SOCNumber': 'self._SOCNumber()',
            'QSOrder': 'self._QSOrder()',
            'PAAC': 'self._PAAC()',
            'APAAC': 'self._APAAC()',
            'OPF_10bit': 'self._OPF_10bit()',
            'OPF_10bit type 1': 'self._OPF_10bit_type_1()',
            'OPF_7bit type 1': 'self._OPF_7bit_type_1()',
            'OPF_7bit type 2': 'self._OPF_7bit_type_2()',
            'OPF_7bit type 3': 'self._OPF_7bit_type_3()',
            'ASDC': 'self._ASDC()',
            'DistancePair': 'self._DistancePair()',
            'AC': 'self._AC()',
            'CC': 'self._CC()',
            'ACC': 'self._ACC()',
            'PseKRAAC type 1': 'self._PseKRAAC_type_1()',
            'PseKRAAC type 2': 'self._PseKRAAC_type_2()',
            'PseKRAAC type 3A': 'self._PseKRAAC_type_3A()',
            'PseKRAAC type 3B': 'self._PseKRAAC_type_3B()',
            'PseKRAAC type 4': 'self._PseKRAAC_type_4()',
            'PseKRAAC type 5': 'self._PseKRAAC_type_5()',
            'PseKRAAC type 6A': 'self._PseKRAAC_type_6A()',
            'PseKRAAC type 6B': 'self._PseKRAAC_type_6B()',
            'PseKRAAC type 6C': 'self._PseKRAAC_type_6C()',
            'PseKRAAC type 7': 'self._PseKRAAC_type_7()',
            'PseKRAAC type 8': 'self._PseKRAAC_type_8()',
            'PseKRAAC type 9': 'self._PseKRAAC_type_9()',
            'PseKRAAC type 10': 'self._PseKRAAC_type_10()',
            'PseKRAAC type 11': 'self._PseKRAAC_type_11()',
            'PseKRAAC type 12': 'self._PseKRAAC_type_12()',
            'PseKRAAC type 13': 'self._PseKRAAC_type_13()',
            'PseKRAAC type 14': 'self._PseKRAAC_type_14()',
            'PseKRAAC type 15': 'self._PseKRAAC_type_15()',
            'PseKRAAC type 16': 'self._PseKRAAC_type_16()',
        }

    def import_parameters(self, file):
        if os.path.exists(file):
            with open(file) as f:
                records = f.read().strip()
            try:
                self.__default_para_dict = json.loads(records)
                print('File imported successfully.')
            except Exception as e:
                print('Parameter file parser error.')

    def get_descriptor(self, descriptor='AAC'):
        # copy parameters
        if descriptor in self.__default_para_dict:
            for key in self.__default_para_dict[descriptor]:
                self.__default_para[key] = self.__default_para_dict[descriptor][key]       
            
        if descriptor in self.__cmd_dict:
            cmd = self.__cmd_dict[descriptor]
            status = eval(cmd)            
        else:
            print('The descriptor type does not exist.')

    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        AAC                                                Amino acid composition
        EAAC                                               Enhanced amino acid composition
        CKSAAP                                             Composition of k-spaced amino acid pairs
        DPC                                                Dipeptide composition
        TPC                                                Tripeptide composition
        CTDC                                               Composition
        CTDT                                               Transition
        CTDD                                               Distribution
        CTriad                                             Conjoint triad
        KSCTriad                                           Conjoint k-spaced triad
        ASDC                                               Adaptive skip dipeptide composition
        DistancePair                                       PseAAC of distance-pairs and reduced alphabe
        GAAC                                               Grouped amino acid composition
        EGAAC                                              Enhanced grouped amino acid composition
        CKSAAGP                                            Composition of k-spaced amino acid group pairs
        GDPC                                               Grouped dipeptide composition
        GTPC                                               Grouped tripeptide composition
        Moran                                              Moran
        Geary                                              Geary
        NMBroto                                            Normalized Moreau-Broto
        AC                                                 Auto covariance
        CC                                                 Cross covariance
        ACC                                                Auto-cross covariance
        SOCNumber                                          Sequence-order-coupling number
        QSOrder                                            Quasi-sequence-order descriptors
        PAAC                                               Pseudo-amino acid composition
        APAAC                                              Amphiphilic PAAC
        PseKRAAC type 1                                    Pseudo K-tuple reduced amino acids composition type 1
        PseKRAAC type 2                                    Pseudo K-tuple reduced amino acids composition type 2
        PseKRAAC type 3A                                   Pseudo K-tuple reduced amino acids composition type 3A
        PseKRAAC type 3B                                   Pseudo K-tuple reduced amino acids composition type 3B
        PseKRAAC type 4                                    Pseudo K-tuple reduced amino acids composition type 4
        PseKRAAC type 5                                    Pseudo K-tuple reduced amino acids composition type 5
        PseKRAAC type 6A                                   Pseudo K-tuple reduced amino acids composition type 6A
        PseKRAAC type 6B                                   Pseudo K-tuple reduced amino acids composition type 6B
        PseKRAAC type 6C                                   Pseudo K-tuple reduced amino acids composition type 6C
        PseKRAAC type 7                                    Pseudo K-tuple reduced amino acids composition type 7
        PseKRAAC type 8                                    Pseudo K-tuple reduced amino acids composition type 8
        PseKRAAC type 9                                    Pseudo K-tuple reduced amino acids composition type 9
        PseKRAAC type 10                                   Pseudo K-tuple reduced amino acids composition type 10
        PseKRAAC type 11                                   Pseudo K-tuple reduced amino acids composition type 11
        PseKRAAC type 12                                   Pseudo K-tuple reduced amino acids composition type 12
        PseKRAAC type 13                                   Pseudo K-tuple reduced amino acids composition type 13
        PseKRAAC type 14                                   Pseudo K-tuple reduced amino acids composition type 14
        PseKRAAC type 15                                   Pseudo K-tuple reduced amino acids composition type 15
        PseKRAAC type 16                                   Pseudo K-tuple reduced amino acids composition type 16
        binary                                             Binary
        binary_6bit                                        Binary
        binary_5bit type 1                                 Binary
        binary_5bit type 2                                 Binary
        binary_3bit type 1                                 Binary
        binary_3bit type 2                                 Binary
        binary_3bit type 3                                 Binary
        binary_3bit type 4                                 Binary
        binary_3bit type 5                                 Binary
        binary_3bit type 6                                 Binary
        binary_3bit type 7                                 Binary
        AESNN3                                             Learn from alignments
        OPF_10bit                                          Overlapping property features - 10 bit
        OPF_7bit type 1                                    Overlapping property features - 7 bit type 1
        OPF_7bit type 2                                    Overlapping property features - 7 bit type 2
        OPF_7bit type 3                                    Overlapping property features - 7 bit type 3
        AAIndex                                            AAIndex
        BLOSUM62                                           BLOSUM62
        ZScale                                             Z-Scales index

        Note: the first column is the names of availables feature types while the second column is description.  
        
        '''

        print(info)

    def _AAC(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            header = ['SampleName']
            encodings = []
            for i in AA:
                header.append('AAC_{0}'.format(i))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name]
                for aa in AA:
                    code.append(count[aa])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _EAAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EAAC descriptor need fasta sequence with equal length.'
                self.encodings = None
            else:
                AA = 'ARNDCQEGHILKMFPSTWYV'
                encodings = []
                header = ['SampleName']
                for w in range(1, len(self.fasta_list[0][1]) - self.__default_para['sliding_window'] + 2):
                    for aa in AA:
                        header.append('EAAC_SW.' + str(w) + '.' + aa)
                encodings.append(header)
                for i in self.fasta_list:
                    name, sequence, label = i[0], i[1], i[2]
                    code = [name]
                    for j in range(len(sequence)):
                        if j < len(sequence) and j + self.__default_para['sliding_window'] <= len(sequence):
                            count = Counter(sequence[j:j + self.__default_para['sliding_window']])
                            for key in count:
                                count[key] = count[key] / len(sequence[j:j + self.__default_para['sliding_window']])
                            for aa in AA:
                                code.append(count[aa])
                    encodings.append(code)
                encodings = np.array(encodings)
                self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
                return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _CKSAAP(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)
            header = ['SampleName']
            gap = self.__default_para['kspace']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append('CKSAAP_' + aa + '.gap' + str(g))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '' , i[1]), i[2]
                code = [name]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False
  
    def _DPC(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            diPeptides = ['DPC_' + aa1 + aa2 for aa1 in AA for aa2 in AA]
            header = ['SampleName'] + diPeptides
            encodings.append(header)
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 400
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DDE(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                        }
            encodings = []
            diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            header = ['SampleName'] + ['DDE_{0}'.format(i) for i in diPeptides]
            encodings.append(header)
            myTM = []
            for pair in diPeptides:
                myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))
            
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]                
                code = [name]
                tmpCode = [0] * 400
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                        sequence[j + 1]]] + 1                
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]               
                
                myTV = []
                for j in range(len(myTM)):
                    myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))
                for j in range(len(tmpCode)):
                    tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])
                code = code + tmpCode                
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _TPC(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            triPeptides = ['TPC_' + aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            header = ['SampleName'] + triPeptides
            encodings.append(header)
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 8000
                for j in range(len(sequence) - 3 + 1):                    
                    tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j + 1]] * 20 + AADict[sequence[j + 2]]] = tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j + 1]] * 20 + AADict[sequence[j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            AA = 'ARNDCQEGHILKMFPSTWYV'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 20 + 1):
                header.append('Binary_' + str(i))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_6bit(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'HRK',
                'DENQ',
                'C',
                'STPAG',
                'MILV',
                'FYW',
            ]
            encodings = []
            header = ['SampleName']
            header += ['Binary6_p%s_g%s' % (i + 1, j + 1) for i in range(len(self.fasta_list[0][1])) for j in
                    range(len(self.AA_group_list))]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_5bit_type_1(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'GAVLMI',
                'FYW',
                'KRH',
                'DE',
                'STCPNQ',
            ]
            self.AA_group_index = ['alphatic', 'aromatic', 'postivecharge', 'negativecharge', 'uncharge']
            encodings = []
            header = ['SampleName']
            header += ['Binary5_t1_p%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_5bit_type_2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            aa_dict = {
                'A': [0, 0, 0, 1, 1],
                'C': [0, 0, 1, 0, 1],
                'D': [0, 0, 1, 1, 0],
                'E': [0, 0, 1, 1, 1],
                'F': [0, 1, 0, 0, 1],
                'G': [0, 1, 0, 1, 0],
                'H': [0, 1, 0, 1, 1],
                'I': [0, 1, 1, 0, 0],
                'K': [0, 1, 1, 0, 1],
                'L': [0, 1, 1, 1, 0],
                'M': [1, 0, 0, 0, 1],
                'N': [1, 0, 0, 1, 0],
                'P': [1, 0, 0, 1, 1],
                'Q': [1, 0, 1, 0, 0],
                'R': [1, 0, 1, 0, 1],
                'S': [1, 0, 1, 1, 0],
                'T': [1, 1, 0, 0, 0],
                'V': [1, 1, 0, 0, 1],
                'W': [1, 1, 0, 1, 0],
                'Y': [1, 1, 1, 0, 0],
            }
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 5 + 1):
                header.append('Binary5_t2_' + str(i))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    if aa in aa_dict:
                        code += aa_dict[aa]
                    else:
                        code += [0, 0, 0, 0, 0]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_1(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'RKEDQN',
                'GASTPHY',
                'CLVIMFW',
            ]
            self.AA_group_index = ['Polar', 'Neutral', 'Hydrophobicity']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t1_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'GASTPD',
                'NVEQIL',
                'MHKFRYW',
            ]
            self.AA_group_index = ['Volume_range(0-2.78)', 'Volumn_range(2.95-4.0)', 'Volumn_range(4.03-8.08)']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t2_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_3(self):
        try:            
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'RKEDQN',
                'GASTPHY',
                'CLVIMFW',
            ]
            self.AA_group_index = ['PolarityValue(4.9-6.2)', 'PolarityValue(8.0-9.2)', 'PolarityValue(10.4-13.0)']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t3_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_4(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'GASDT',
                'CPNVEQIL',
                'KMHFRYW',
            ]
            self.AA_group_index = ['PolarizabilityValue(0-0.108)', 'PolarizabilityValue(0.128-0.186)',
                            'PolarizabilityValue(0.219-0.409)']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t4_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_5(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'KR',
                'ANCQGHILMFPStWYV',
                'DE',
            ]
            self.AA_group_index = ['Positive', 'Neutral', 'Negative']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t5_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_6(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'EALMQKRH',
                'VIYCWFT',
                'GNPSD',
            ]
            self.AA_group_index = ['Helix', 'Strand', 'Coil']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t6_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary_3bit_type_7(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            self.AA_group_list = [
                'ALFCGIVW',
                'PKQEND',
                'MPSTHY',
            ]
            self.AA_group_index = ['Buried', 'Exposed', 'Intermediate']
            encodings = []
            header = ['SampleName']
            header += ['Binary3_t7_p%s_g%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in self.AA_group_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    for j in self.AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _AESNN3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'AESNN3 descriptor need fasta sequence with equal length.'
                return False
            AESNN3_dict = {
                'A': [-0.99, -0.61,  0.00],
                'R': [ 0.28, -0.99, -0.22],
                'N': [ 0.77, -0.24,  0.59],
                'D': [ 0.74, -0.72, -0.35],
                'C': [ 0.34,  0.88,  0.35],
                'Q': [ 0.12, -0.99, -0.99],
                'E': [ 0.59, -0.55, -0.99],
                'G': [-0.79, -0.99,  0.10],
                'H': [ 0.08, -0.71,  0.68],
                'I': [-0.77,  0.67, -0.37],
                'L': [-0.92,  0.31, -0.99],
                'K': [-0.63,  0.25,  0.50],
                'M': [-0.80,  0.44, -0.71],
                'F': [ 0.87,  0.65, -0.53],
                'P': [-0.99, -0.99, -0.99],
                'S': [ 0.99,  0.40,  0.37],
                'T': [ 0.42,  0.21,  0.97],
                'W': [-0.13,  0.77, -0.90],
                'Y': [ 0.59,  0.33, -0.99],
                'V': [-0.99,  0.27, -0.52],
                '-': [    0,     0,     0],
            }
            encodings = []
            header = ['SampleName']
            for p in range(1, len(self.fasta_list[0][1]) + 1):
                for z in ('1', '2', '3'):
                    header.append('AESNN3_' + 'p' + str(p) + 'z' + z)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code += AESNN3_dict.get(aa, [0, 0, 0])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _GAAC(self):
        try:
            group = {
                'alphatic': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharge': 'KRH',
                'negativecharge': 'DE',
                'uncharge': 'STCPNQ'
            }
            groupKey = group.keys()
            encodings = []
            header = ['SampleName']
            for key in groupKey:
                header.append('GAAC_{0}'.format(key))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                count = Counter(sequence)
                myDict = {}
                for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]
                for key in groupKey:
                    code.append(myDict[key] / len(sequence))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _EGAAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EGAAC descriptor need fasta sequence with equal length.'
                return False
            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }
            groupKey = group.keys()
            encodings = []
            header = ['SampleName']
            window = self.__default_para['sliding_window']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for g in groupKey:
                    header.append('EGAAC_SW' + str(w) + '.' + g)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence)):
                    if j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        myDict = {}
                        for key in groupKey:
                            for aa in group[key]:
                                myDict[key] = myDict.get(key, 0) + count[aa]
                        for key in groupKey:
                            code.append(myDict[key] / window)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generateGroupPairs(self, groupKey):
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair

    def _CKSAAGP(self):
        try:
            gap = self.__default_para['kspace']
            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }
            AA = 'ARNDCQEGHILKMFPSTWYV'
            groupKey = group.keys()
            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key
            gPairIndex = []
            for key1 in groupKey:
                for key2 in groupKey:
                    gPairIndex.append(key1 + '.' + key2)
            encodings = []
            header = ['SampleName']
            for g in range(gap + 1):
                for p in gPairIndex:
                    header.append('CKSAAGP_' + p + '.gap' + str(g))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                for g in range(gap + 1):
                    gPair = self.generateGroupPairs(groupKey)
                    sum = 0
                    for p1 in range(len(sequence)):
                        p2 = p1 + g + 1
                        if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                            gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[
                                                                                        index[sequence[p1]] + '.' + index[
                                                                                            sequence[p2]]] + 1
                            sum = sum + 1
                    if sum == 0:
                        for gp in gPairIndex:
                            code.append(0)
                    else:
                        for gp in gPairIndex:
                            code.append(gPair[gp] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _GDPC(self):
        try:
            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }
            groupKey = group.keys()
            baseNum = len(groupKey)
            dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]
            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key
            encodings = []
            header = ['SampleName'] + ['GDPC_{0}'.format(i) for i in dipeptide]        
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                myDict = {}
                for t in dipeptide:
                    myDict[t] = 0
                sum = 0
                for j in range(len(sequence) - 2 + 1):
                    myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] + 1
                    sum = sum + 1
                if sum == 0:
                    for t in dipeptide:
                        code.append(0)
                else:
                    for t in dipeptide:
                        code.append(myDict[t] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _GTPC(self):
        try:
            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }
            groupKey = group.keys()
            baseNum = len(groupKey)
            triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]
            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key
            encodings = []
            header = ['SampleName'] + ['GTPC_{0}'.format(i) for i in triple]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                myDict = {}
                for t in triple:
                    myDict[t] = 0
                sum = 0
                for j in range(len(sequence) - 3 + 1):
                    myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] = myDict[index[
                                                                                                                        sequence[
                                                                                                                            j]] + '.' +
                                                                                                                    index[
                                                                                                                        sequence[
                                                                                                                            j + 1]] + '.' +
                                                                                                                    index[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                    sum = sum + 1
                if sum == 0:
                    for t in triple:
                        code.append(0)
                else:
                    for t in triple:
                        code.append(myDict[t] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _AAIndex(self):
        try:
            props = self.__default_para['aaindex'].split(';')
            self.encoding_array = np.array([])
            if not self.is_equal:
                self.error_msg = 'AAIndex descriptor need fasta sequence with equal length.'
                return False
            AA = 'ARNDCQEGHILKMFPSTWYV'            
            fileAAindex = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAindex.txt')
            with open(fileAAindex) as f:
                records = f.readlines()[1:]
            AAindex = []
            AAindexName = []
            for i in records:
                AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
                AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i
            #  use the user inputed properties
            if props:
                tmpIndexNames = []
                tmpIndex = []
                for p in props:
                    if AAindexName.index(p) != -1:
                        tmpIndexNames.append(p)
                        tmpIndex.append(AAindex[AAindexName.index(p)])
                if len(tmpIndexNames) != 0:
                    AAindexName = tmpIndexNames
                    AAindex = tmpIndex
            encodings = []
            header = ['SampleName']
            for pos in range(1, len(self.fasta_list[0][1]) + 1):
                for idName in AAindexName:
                    header.append('AAindex_' + 'p.' + str(pos) + '.' + idName)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    if aa == '-':
                        for j in AAindex:
                            code.append(0)
                        continue
                    for j in AAindex:
                        code.append(j[index[aa]])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ZScale(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ZScale descriptor need fasta sequence with equal length.'
                return False
            zscale = {
                'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
                'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
                'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
                'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
                'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
                'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
                'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
                'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
                'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
                'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
                'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
                'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
                'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
                'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
                'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
                'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
                'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
                'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
                'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
                'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
                '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
            }
            encodings = []
            header = ['SampleName']
            for p in range(1, len(self.fasta_list[0][1]) + 1):
                for z in ('1', '2', '3', '4', '5'):
                    header.append('ZScale_' + 'p' + str(p) + '.z' + z)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code = code + zscale[aa]
                encodings.append(code)            
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _BLOSUM62(self):
        try:
            if not self.is_equal:
                self.error_msg = 'BLOSUM62 descriptor need fasta sequence with equal length.'
                return False
            blosum62 = {
                'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
                'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
                'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
                'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
                'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
                'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
                'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
                'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
                'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
                'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
                'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
                'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
                'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
                'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
                'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
                'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
                'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
                'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
                'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
                'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
                '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
            }
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 20 + 1):
                header.append('blosum62_' + str(i))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code = code + blosum62[aa]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _NMBroto(self):
        try:
            props = self.__default_para['aaindex'].split(';')
            nlag = self.__default_para['nlag']
            AA = 'ARNDCQEGHILKMFPSTWYV'            
            fileAAidx = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAidx.txt')
            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]
            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None
            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))
            pstd = np.std(AAidx, axis=1)
            pmean = np.average(AAidx, axis=1)
            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - pmean[i]) / pstd[i]
            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i
            encodings = []
            header = ['SampleName']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append('NMBroto_' + p + '.lag' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                N = len(sequence)
                for prop in range(len(props)):
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            rn = sum(
                                [AAidx[prop][index.get(sequence[j], 0)] * AAidx[prop][index.get(sequence[j + n], 0)] for j
                                in range(len(sequence) - n)]) / (N - n)
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Moran(self):
        try:
            props = self.__default_para['aaindex'].split(';')
            nlag = self.__default_para['nlag']
            AA = 'ARNDCQEGHILKMFPSTWYV'           
            fileAAidx = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAidx.txt')
            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]
            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None
            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))
            propMean = np.mean(AAidx, axis=1)
            propStd = np.std(AAidx, axis=1)
            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]
            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i
            encodings = []
            header = ['SampleName']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append('Moran_' + p + '.lag' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                N = len(sequence)
                for prop in range(len(props)):
                    xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            fenzi = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) * (
                                    AAidx[prop][index.get(sequence[j + n], 0)] - xmean) for j in
                                        range(len(sequence) - n)]) / (N - n)
                            fenmu = sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))]) / N
                            rn = fenzi / fenmu
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Geary(self):
        try:           
            props = self.__default_para['aaindex'].split(';')
            nlag = self.__default_para['nlag']            
            fileAAidx = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAidx.txt')            
            AA = 'ARNDCQEGHILKMFPSTWYV'
            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]
            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None
            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))
            propMean = np.mean(AAidx, axis=1)
            propStd = np.std(AAidx, axis=1)
            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]
            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i
            encodings = []
            header = ['SampleName']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append('Geary_' + p + '.lag' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                N = len(sequence)
                for prop in range(len(props)):
                    xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            rn = (N - 1) / (2 * (N - n)) * ((sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)]) ** 2
                                for
                                j in range(len(sequence) - n)])) / (sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generatePropertyPairs(self, myPropertyName):
            pairs = []
            for i in range(len(myPropertyName)):
                for j in range(i + 1, len(myPropertyName)):
                    pairs.append([myPropertyName[i], myPropertyName[j]])
                    pairs.append([myPropertyName[j], myPropertyName[i]])
            return pairs

    def _AC(self):
        try:
            property_name = self.__default_para['aaindex'].split(';')
            nlag = self.__default_para['nlag']
            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False
            try:                
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAindex.data')
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]
            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for i in range(nlag):
                    header.append('AC_%s.lag%s' %(p_name, i+1))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                L = len(sequence)
                for p_name in property_name:
                    xmean = sum([property_dict[p_name][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        ac = 0
                        try:
                            ac = sum([(property_dict[p_name][AA_order_dict[sequence[j]]] - xmean) * (property_dict[p_name][AA_order_dict[sequence[j+lag]]] - xmean) for j in range(L - lag)])/(L-lag)
                        except Exception as e:
                            ac = 0
                        code.append(ac)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _CC(self):
        try:
            property_name = self.__default_para['aaindex'].split(';')
            if len(property_name) < 2:
                self.error_msg = 'More than two property should be selected for this descriptor.'
                return False
            nlag = self.__default_para['nlag']
            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False
            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAindex.data')
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]
            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i
            property_pairs = self.generatePropertyPairs(property_name)
            encodings = []
            header = ['SampleName']
            header += ['CC_' + p[0] + '_' + p[1] + '_lag.' + str(lag) for p in property_pairs for lag in range(1, nlag + 1)]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]                
                L = len(sequence)
                for pair in property_pairs:
                    mean_p1 = sum([property_dict[pair[0]][AA_order_dict[aa]] for aa in sequence]) / L
                    mean_p2 = sum([property_dict[pair[1]][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        cc = 0
                        try:
                            cc = sum([(property_dict[pair[0]][AA_order_dict[sequence[j]]] - mean_p1) * (property_dict[pair[1]][AA_order_dict[sequence[j+lag]]] - mean_p2) for j in range(L - lag)]) / (L - lag)
                        except Exception as e:
                            cc = 0
                        code.append(cc)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ACC(self):
        try:
            property_name = self.__default_para['aaindex'].split(';')
            if len(property_name) < 2:
                self.error_msg = 'More than two property should be selected for this descriptor.'
                return False
            nlag = self.__default_para['nlag']
            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False
            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'AAindex.data')
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]
            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i
            property_pairs = self.generatePropertyPairs(property_name)
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for i in range(nlag):
                    header.append('ACC_%s.lag%s' % (p_name, i + 1))
            header += ['ACC_' + p[0] + '_' + p[1] + '_lag.' + str(lag) for p in property_pairs for lag in range(1, nlag + 1)]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                L = len(sequence)
                for p_name in property_name:
                    xmean = sum([property_dict[p_name][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        ac = 0
                        try:
                            ac = sum([(property_dict[p_name][AA_order_dict[sequence[j]]] - xmean) * (property_dict[p_name][AA_order_dict[sequence[j+lag]]] - xmean) for j in range(L - lag)])/(L-lag)
                        except Exception as e:
                            ac = 0
                        code.append(ac)
                for pair in property_pairs:
                    mean_p1 = sum([property_dict[pair[0]][AA_order_dict[aa]] for aa in sequence]) / L
                    mean_p2 = sum([property_dict[pair[1]][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        cc = 0
                        try:
                            cc = sum([(property_dict[pair[0]][AA_order_dict[sequence[j]]] - mean_p1) * (
                                        property_dict[pair[1]][AA_order_dict[sequence[j + lag]]] - mean_p2) for j in
                                    range(L - lag)]) / (L - lag)
                        except Exception as e:
                            cc = 0
                        code.append(cc)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def _CTDC(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }
            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
            encodings = []
            header = ['SampleName']
            for p in property:
                for g in range(1, len(groups) + 1):
                    header.append('CTDC_' + p + '.G' + str(g))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                for p in property:
                    c1 = self.Count(group1[p], sequence) / len(sequence)
                    c2 = self.Count(group2[p], sequence) / len(sequence)
                    c3 = 1 - c1 - c2
                    code = code + [c1, c2, c3]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _CTDT(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }
            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
            encodings = []
            header = ['SampleName']
            for p in property:
                for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                    header.append('CTDT_' + p + '.' + tr)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
                for p in property:
                    c1221, c1331, c2332 = 0, 0, 0
                    for pair in aaPair:
                        if (pair[0] in group1[p] and pair[1] in group2[p]) or (
                                pair[0] in group2[p] and pair[1] in group1[p]):
                            c1221 = c1221 + 1
                            continue
                        if (pair[0] in group1[p] and pair[1] in group3[p]) or (
                                pair[0] in group3[p] and pair[1] in group1[p]):
                            c1331 = c1331 + 1
                            continue
                        if (pair[0] in group2[p] and pair[1] in group3[p]) or (
                                pair[0] in group3[p] and pair[1] in group2[p]):
                            c2332 = c2332 + 1
                    code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code

    def _CTDD(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }
            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
            encodings = []
            header = ['SampleName']
            for p in property:
                for g in ('1', '2', '3'):
                    for d in ['0', '25', '50', '75', '100']:
                        header.append('CTDD_' + p + '.' + g + '.residue' + d)
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                for p in property:
                    code = code + self.Count1(group1[p], sequence) + self.Count1(group2[p], sequence) + self.Count1(
                        group3[p], sequence)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CalculateKSCTriad(self, sequence, gap, features, AADict):
        res = []
        for g in range(gap + 1):
            myDict = {}
            for f in features:
                myDict[f] = 0

            for i in range(len(sequence)):
                if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                    fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                        sequence[i + 2 * g + 2]]
                    myDict[fea] = myDict[fea] + 1
           
            maxValue, minValue = np.max(list(myDict.values())), np.min(list(myDict.values()))            
            for f in features:
                res.append((myDict[f] - minValue) / maxValue)
        return res

    def _CTriad(self):
        # try:
        if self.minimum_length_without_minus < 3:
            self.error_msg = 'CTriad descriptor need fasta sequence with minimum length > 3.'
            return False
        AAGroup = {
            'g1': 'AGV',
            'g2': 'ILFP',
            'g3': 'YMTS',
            'g4': 'HNQW',
            'g5': 'RK',
            'g6': 'DE',
            'g7': 'C'
        }
        myGroups = sorted(AAGroup.keys())
        AADict = {}
        for g in myGroups:
            for aa in AAGroup[g]:
                AADict[aa] = g
        features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
        encodings = []
        header = ['SampleName']
        for f in features:
            header.append('CTriad_{0}'.format(f))
        encodings.append(header)
        for i in self.fasta_list:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            code = [name]
            code = code + self.CalculateKSCTriad(sequence, 0, features, AADict)
            encodings.append(code)
        encodings = np.array(encodings)
        self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
        return True
        # except Exception as e:
        #     self.error_msg = str(e)
        #     return False

    def _KSCTriad(self):
        try:
            gap = self.__default_para['kspace']
            if self.minimum_length_without_minus < 2 * gap + 3:
                self.error_msg = 'KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3).'
                return False
            AAGroup = {
                'g1': 'AGV',
                'g2': 'ILFP',
                'g3': 'YMTS',
                'g4': 'HNQW',
                'g5': 'RK',
                'g6': 'DE',
                'g7': 'C'
            }
            myGroups = sorted(AAGroup.keys())
            AADict = {}
            for g in myGroups:
                for aa in AAGroup[g]:
                    AADict[aa] = g
            features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
            encodings = []
            header = ['SampleName']
            for g in range(gap + 1):
                for f in features:
                    header.append('KSCTriad_' + f + '.gap' + str(g))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                if len(sequence) < 2 * gap + 3:
                    self.error_msg = 'Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3).'
                    return 0
                code = code + self.CalculateKSCTriad(sequence, gap, features, AADict)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _SOCNumber(self):
        try:
            nlag = self.__default_para['nlag']            
            dataFile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'Schneider-Wrede.txt')            
            dataFile1 = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'Grantham.txt')
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA1 = 'ARNDCQEGHILKMFPSTWYV'
            DictAA = {}
            for i in range(len(AA)):
                DictAA[AA[i]] = i
            DictAA1 = {}
            for i in range(len(AA1)):
                DictAA1[AA1[i]] = i
            with open(dataFile) as f:
                records = f.readlines()[1:]
            AADistance = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance.append(array)
            AADistance = np.array([float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))
            with open(dataFile1) as f:
                records = f.readlines()[1:]
            AADistance1 = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance1.append(array)
            AADistance1 = np.array([float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape((20, 20))
            encodings = []
            header = ['SampleName']
            for n in range(1, nlag + 1):
                header.append('SOCNumber_Schneider.lag' + str(n))
            for n in range(1, nlag + 1):
                header.append('SOCNumber_gGrantham.lag' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                for n in range(1, nlag + 1):
                    code.append(sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (len(sequence) - n))
                for n in range(1, nlag + 1):
                    code.append(sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (len(sequence) - n))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _QSOrder(self):
        try:
            nlag = self.__default_para['nlag']
            w = self.__default_para['weight']
            if nlag > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lag value is out of range.'
                return False            
            dataFile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'Schneider-Wrede.txt')           
            dataFile1 = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'Grantham.txt')
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA1 = 'ARNDCQEGHILKMFPSTWYV'
            DictAA = {}
            for i in range(len(AA)):
                DictAA[AA[i]] = i
            DictAA1 = {}
            for i in range(len(AA1)):
                DictAA1[AA1[i]] = i
            with open(dataFile) as f:
                records = f.readlines()[1:]
            AADistance = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance.append(array)
            AADistance = np.array([float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))
            with open(dataFile1) as f:
                records = f.readlines()[1:]
            AADistance1 = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance1.append(array)
            AADistance1 = np.array([float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape((20, 20))
            encodings = []
            header = ['SampleName']
            for aa in AA1:
                header.append('QSOrder_Schneider.Xr.' + aa)
            for aa in AA1:
                header.append('QSOrder_Grantham.Xr.' + aa)
            for n in range(1, nlag + 1):
                header.append('QSOrder_Schneider.Xd.' + str(n))
            for n in range(1, nlag + 1):
                header.append('QSOrder_Grantham.Xd.' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                arraySW = []
                arrayGM = []
                for n in range(1, nlag + 1):
                    arraySW.append(
                        sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in
                            range(len(sequence) - n)]))
                    arrayGM.append(sum(
                        [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                        range(len(sequence) - n)]))
                myDict = {}
                for aa in AA1:
                    myDict[aa] = sequence.count(aa)
                for aa in AA1:
                    code.append(myDict[aa] / (1 + w * sum(arraySW)))
                for aa in AA1:
                    code.append(myDict[aa] / (1 + w * sum(arrayGM)))
                for num in arraySW:
                    code.append((w * num) / (1 + w * sum(arraySW)))
                for num in arrayGM:
                    code.append((w * num) / (1 + w * sum(arrayGM)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Rvalue(self, aa1, aa2, AADict, Matrix):
        return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

    def _PAAC(self):
        try:
            lambdaValue = self.__default_para['lambdaValue']
            if lambdaValue > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lambda value is out of range.'
                return False
            w = self.__default_para['weight']
            dataFile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'PAAC.txt')
            with open(dataFile) as f:
                records = f.readlines()
            AA = ''.join(records[0].rstrip().split()[1:])
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            AAProperty = []
            AAPropertyNames = []
            for i in range(1, len(records)):
                array = records[i].rstrip().split() if records[i].rstrip() != '' else None
                AAProperty.append([float(j) for j in array[1:]])
                AAPropertyNames.append(array[0])
            AAProperty1 = []
            for i in AAProperty:
                meanI = sum(i) / 20
                fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
                AAProperty1.append([(j - meanI) / fenmu for j in i])
            encodings = []
            header = ['SampleName']
            for aa in AA:
                header.append('PAAC_Xc1.' + aa)
            for n in range(1, lambdaValue + 1):
                header.append('PAAC_Xc2.lambda' + str(n))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                theta = []
                for n in range(1, lambdaValue + 1):
                    theta.append(
                        sum([self.Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                            range(len(sequence) - n)]) / (
                                len(sequence) - n))
                myDict = {}
                for aa in AA:
                    myDict[aa] = sequence.count(aa)
                code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
                code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _APAAC(self):
        try:
            lambdaValue = self.__default_para['lambdaValue']
            if lambdaValue > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lambda value is out of range.'
                return False
            w = self.__default_para['weight']            
            dataFile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', 'PAAC.txt')
            with open(dataFile) as f:
                records = f.readlines()
            AA = ''.join(records[0].rstrip().split()[1:])
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            AAProperty = []
            AAPropertyNames = []
            for i in range(1, len(records) - 1):
                array = records[i].rstrip().split() if records[i].rstrip() != '' else None
                AAProperty.append([float(j) for j in array[1:]])
                AAPropertyNames.append(array[0])
            AAProperty1 = []
            for i in AAProperty:
                meanI = sum(i) / 20
                fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
                AAProperty1.append([(j - meanI) / fenmu for j in i])
            encodings = []
            header = ['SampleName']
            for i in AA:
                header.append('APAAC_Pc1.' + i)
            for j in range(1, lambdaValue + 1):
                for i in AAPropertyNames:
                    header.append('APAAC_Pc2.' + i + '.' + str(j))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                theta = []
                for n in range(1, lambdaValue + 1):
                    for j in range(len(AAProperty1)):
                        theta.append(
                            sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                range(len(sequence) - n)]) / (len(sequence) - n))
                myDict = {}
                for aa in AA:
                    myDict[aa] = sequence.count(aa)

                code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
                code = code + [w * value / (1 + w * sum(theta)) for value in theta]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _OPF_10bit(self):
        try:
            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False
            physicochemical_properties_list = [
                'FYWH',
                'DE',
                'KHR',
                'NQSDECTKRHYW',
                'AGCTIVLKHFYWM',
                'IVL',
                'ASGC',
                'KHRDE',
                'PNDTCAGSV',
                'P',
            ]
            physicochemical_properties_index = ['Aromatic', 'Negative', 'Positive', 'Polar', 'Hydrophobic', 'Aliphatic',
                                                'Tiny', 'Charged', 'Small', 'Proline']
            header = ['SampleName']
            encodings = []
            header += ['OPF_p%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _OPF_7bit_type_1(self):
        try:
            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False
            physicochemical_properties_list = [
                'ACFGHILMNPQSTVWY',
                'CFILMVW',
                'ACDGPST',
                'CFILMVWY',
                'ADGST',
                'DGNPS',
                'ACFGILVW',
            ]
            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']
            header = ['SampleName']
            encodings = []
            header += ['OPF7_t1_p%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _OPF_7bit_type_2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False
            physicochemical_properties_list = [
                'DE',
                'AGHPSTY',
                'EILNQV',
                'AGPST',
                'CEILNPQV',
                'AEHKLMQR',
                'HMPSTY',
            ]
            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']
            header = ['SampleName']
            encodings = []
            header += ['OPF7_t2_p%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _OPF_7bit_type_3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False
            physicochemical_properties_list = [
                'KR',
                'DEKNQR',
                'FHKMRWY',
                'DEHKNQR',
                'FHKMRWY',
                'CFITVWY',
                'DEKNRQ',
            ]
            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']
            header = ['SampleName']
            encodings = []
            header += ['OPF7_t3_p%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True           
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ASDC(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)
            header = ['SampleName']
            header += ['ASDC_' +aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DistancePair(self):
        try:
            cp20_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'L',
                'M': 'M',
                'N': 'N',
                'P': 'P',
                'Q': 'Q',
                'R': 'R',
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'W',
                'Y': 'Y',
            }
            cp19_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'L',
                'M': 'M',
                'N': 'N',
                'P': 'P',
                'Q': 'Q',
                'R': 'R',
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'W',
                'Y': 'F',  # YF
            }
            cp14_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'H',      # HRKQ
                'L': 'L',
                'M': 'I',      # IMV
                'N': 'N',
                'P': 'P',
                'Q': 'H',      # HRKQ
                'R': 'H',      # HRKQ
                'S': 'S',
                'T': 'T',
                'V': 'I',      # IMV
                'W': 'W',
                'Y': 'W',      # WY
            }
            cp13_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'I',   # IL
                'M': 'F',   # FM
                'N': 'N',
                'P': 'H',   # HPQWY
                'Q': 'H',   # HPQWY
                'R': 'K',   # KR
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'H',   # HPQWY
                'Y': 'H',   # HPQWY
            }
            cp20_AA = 'ACDEFGHIKLMNPQRSTVWY'
            cp19_AA = 'ACDEFGHIKLMNPQRSTVW'
            cp14_AA = 'ACDEFGHILNPSTW'
            cp13_AA = 'ACDEFGHIKNSTV'
            distance = self.__default_para['distance']
            cp = self.__default_para['cp']
            if self.minimum_length_without_minus < distance + 1:
                self.error_msg = 'The distance value is too large.'
                return False
            AA = cp20_AA
            AA_dict = cp20_dict
            if cp == 'cp(19)':
                AA = cp19_AA
                AA_dict = cp19_dict
            if cp == 'cp(14)':
                AA = cp14_AA
                AA_dict = cp14_dict
            if cp == 'cp(13)':
                AA = cp13_AA
                AA_dict = cp13_dict            
            encodings = []
            pair_dict = {}
            single_dict = {}
            for aa1 in AA:
                single_dict[aa1] = 0
                for aa2 in AA:
                    pair_dict[aa1+aa2] = 0
            header = ['SampleName']
            for d in range(distance+1):
                if d == 0:
                    for key in sorted(single_dict):
                        header.append('DP_{0}'.format(key))
                else:
                    for key in sorted(pair_dict):
                        header.append('DP_%s.distance%s' %(key, d))
            encodings.append(header)
            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                for d in range(distance + 1):
                    if d == 0:
                        tmp_dict = single_dict.copy()
                        for i in range(len(sequence)):
                            tmp_dict[AA_dict[sequence[i]]] += 1
                        for key in sorted(tmp_dict):
                            code.append(tmp_dict[key]/len(sequence))
                    else:
                        tmp_dict = pair_dict.copy()
                        for i in range(len(sequence) - d):
                            tmp_dict[AA_dict[sequence[i]] + AA_dict[sequence[i+d]]] += 1
                        for key in sorted(tmp_dict):
                            code.append(tmp_dict[key]/(len(sequence) -d))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
            
    def gapModel(self, fastas, myDict, gDict, gNames, ktuple, glValue, ttype):
        encodings = []
        header = ['SampleName']
        if ktuple == 1:
            header = header + [ttype + '_' + g + '_gap' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [ttype + '_' + g1 + '_' + g2 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [ttype + '_' + g1 + '_' + g2 + '_' + g3 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence) and j + 2 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                            myDict[sequence[j + 2]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                                myDict[sequence[j + 2]]], 0) + 1
                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    def lambdaModel(self, fastas, myDict, gDict, gNames, ktuple, glValue, ttype):
        encodings = []
        header = ['SampleName']
        if ktuple == 1:
            header = header + [ttype + '_' + g + '_LC' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence)):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [ttype + '_' + g1 + '_' + g2 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [ttype + '_' + g1 + '_' + g2 + '_' + g3 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in
                               gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence) and j + 2 * glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                            myDict[sequence[j + 2 * glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                                myDict[sequence[j + 2 * glValue]]], 0) + 1
                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    def _PseKRAAC_type_1(self):
        try:
            AAGroup = {
                2: ['CMFILVWY', 'AGTSNQDEHRKP'],
                3: ['CMFILVWY', 'AGTSP', 'NQDEHRK'],
                4: ['CMFWY', 'ILV', 'AGTS', 'NQDEHRKP'],
                5: ['WFYH', 'MILV', 'CATSP', 'G', 'NQDERK'],
                6: ['WFYH', 'MILV', 'CATS', 'P', 'G', 'NQDERK'],
                7: ['WFYH', 'MILV', 'CATS', 'P', 'G', 'NQDE', 'RK'],
                8: ['WFYH', 'MILV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                9: ['WFYH', 'MI', 'LV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                10: ['WFY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'DE', 'QRK'],
                11: ['WFY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                12: ['WFY', 'ML', 'IV', 'C', 'A', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                13: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                14: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'R', 'K'],
                15: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                16: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                17: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                18: ['W', 'FY', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                19: ['W', 'F', 'Y', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                20: ['W', 'F', 'Y', 'M', 'L', 'I', 'V', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                print(raactype)
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type1')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type1')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_2(self):
        try:
            AAGroup = {
                2: ['LVIMCAGSTPFYW', 'EDNQKRH'],
                3: ['LVIMCAGSTP', 'FYW', 'EDNQKRH'],
                4: ['LVIMC', 'AGSTP', 'FYW', 'EDNQKRH'],
                5: ['LVIMC', 'AGSTP', 'FYW', 'EDNQ', 'KRH'],
                6: ['LVIM', 'AGST', 'PHC', 'FYW', 'EDNQ', 'KR'],
                8: ['LVIMC', 'AG', 'ST', 'P', 'FYW', 'EDNQ', 'KR', 'H'],
                15: ['LVIM', 'C', 'A', 'G', 'S', 'T', 'P', 'FY', 'W', 'E', 'D', 'N', 'Q', 'KR', 'H'],
                20: ['L', 'V', 'I', 'M', 'C', 'A', 'G', 'S', 'T', 'P', 'F', 'Y', 'W', 'E', 'D', 'N', 'Q', 'K', 'R', 'H'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type2')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type2')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_3A(self):
        try:
            AAGroup = {
                2: ['AGSPDEQNHTKRMILFYVC', 'W'],
                3: ['AGSPDEQNHTKRMILFYV', 'W', 'C'],
                4: ['AGSPDEQNHTKRMIV', 'W', 'YFL', 'C'],
                5: ['AGSPDEQNHTKR', 'W', 'YF', 'MIVL', 'C'],
                6: ['AGSP', 'DEQNHTKR', 'W', 'YF', 'MIL', 'VC'],
                7: ['AGP', 'DEQNH', 'TKRMIV', 'W', 'YF', 'L', 'CS'],
                8: ['AG', 'DEQN', 'TKRMIV', 'HY', 'W', 'L', 'FP', 'CS'],
                9: ['AG', 'P', 'DEQN', 'TKRMI', 'HY', 'W', 'F', 'L', 'VCS'],
                10: ['AG', 'P', 'DEQN', 'TKRM', 'HY', 'W', 'F', 'I', 'L', 'VCS'],
                11: ['AG', 'P', 'DEQN', 'TK', 'RI', 'H', 'Y', 'W', 'F', 'ML', 'VCS'],
                12: ['FAS', 'P', 'G', 'DEQ', 'NL', 'TK', 'R', 'H', 'W', 'Y', 'IM', 'VC'],
                13: ['FAS', 'P', 'G', 'DEQ', 'NL', 'T', 'K', 'R', 'H', 'W', 'Y', 'IM', 'VC'],
                14: ['FA', 'P', 'G', 'T', 'DE', 'QM', 'NL', 'K', 'R', 'H', 'W', 'Y', 'IV', 'CS'],
                15: ['FAS', 'P', 'G', 'T', 'DE', 'Q', 'NL', 'K', 'R', 'H', 'W', 'Y', 'M', 'I', 'VC'],
                16: ['FA', 'P', 'G', 'ST', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'VC'],
                17: ['FA', 'P', 'G', 'S', 'T', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'VC'],
                18: ['FA', 'P', 'G', 'S', 'T', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
                19: ['FA', 'P', 'G', 'S', 'T', 'D', 'E', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
                20: ['F', 'A', 'P', 'G', 'S', 'T', 'D', 'E', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type3A')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type3A')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_3B(self):
        try:
            AAGroup = {
                2: ['HRKQNEDSTGPACVIM', 'LFYW'],
                3: ['HRKQNEDSTGPACVIM', 'LFY', 'W'],
                4: ['HRKQNEDSTGPA', 'CIV', 'MLFY', 'W'],
                5: ['HRKQNEDSTGPA', 'CV', 'IML', 'FY', 'W'],
                6: ['HRKQNEDSTPA', 'G', 'CV', 'IML', 'FY', 'W'],
                7: ['HRKQNEDSTA', 'G', 'P', 'CV', 'IML', 'FY', 'W'],
                8: ['HRKQSTA', 'NED', 'G', 'P', 'CV', 'IML', 'FY', 'W'],
                9: ['HRKQ', 'NED', 'ASTG', 'P', 'C', 'IV', 'MLF', 'Y', 'W'],
                10: ['RKHSA', 'Q', 'NED', 'G', 'P', 'C', 'TIV', 'MLF', 'Y', 'W'],
                11: ['RKQ', 'NG', 'ED', 'AST', 'P', 'C', 'IV', 'HML', 'F', 'Y', 'W'],
                12: ['RKQ', 'ED', 'NAST', 'G', 'P', 'C', 'IV', 'H', 'ML', 'F', 'Y', 'W'],
                13: ['RK', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'ML', 'F', 'Y', 'W'],
                14: ['R', 'K', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'ML', 'F', 'Y', 'W'],
                15: ['R', 'K', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                16: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                17: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'S', 'T', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                18: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
                19: ['R', 'K', 'Q', 'E', 'D', 'NG', 'H', 'A', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
                20: ['R', 'K', 'Q', 'E', 'D', 'N', 'G', 'H', 'A', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type3B')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type3B')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False   

    def _PseKRAAC_type_4(self):
        try:
            AAGroup = {
                5: ['G', 'IVFYW', 'ALMEQRK', 'P', 'NDHSTC'],
                8: ['G', 'IV', 'FYW', 'ALM', 'EQRK', 'P', 'ND', 'HSTC'],
                9: ['G', 'IV', 'FYW', 'ALM', 'EQRK', 'P', 'ND', 'HS', 'TC'],
                11: ['G', 'IV', 'FYW', 'A', 'LM', 'EQRK', 'P', 'ND', 'HS', 'T', 'C'],
                13: ['G', 'IV', 'FYW', 'A', 'L', 'M', 'E', 'QRK', 'P', 'ND', 'HS', 'T', 'C'],
                20: ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type4')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type4')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_5(self):
        try:
            AAGroup = {
                3: ['FWYCILMVAGSTPHNQ', 'DE', 'KR'],
                4: ['FWY', 'CILMV', 'AGSTP', 'EQNDHKR'],
                8: ['FWY', 'CILMV', 'GA', 'ST', 'P', 'EQND', 'H', 'KR'],
                10: ['G', 'FYW', 'A', 'ILMV', 'RK', 'P', 'EQND', 'H', 'ST', 'C'],
                15: ['G', 'FY', 'W', 'A', 'ILMV', 'E', 'Q', 'RK', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
                20: ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type5')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type5')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True                               
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_6A(self):
        try:
            AAGroup = {
                4: ['AGPST', 'CILMV', 'DEHKNQR', 'FYW'],
                5: ['AHT', 'CFILMVWY', 'DE', 'GP', 'KNQRS'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6A')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6A')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_6B(self):
        try:
            AAGroup = {
                5: ['AEHKQRST', 'CFILMVWY', 'DN', 'G', 'P'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6B')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6B')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_6C(self):
        try:
            AAGroup = {
                5: ['AG', 'C', 'DEKNPQRST', 'FILMVWY', 'H'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6C')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type6C')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_7(self):
        try:
            AAGroup = {
                2: ['C', 'MFILVWYAGTSNQDEHRKP'],
                3: ['C', 'MFILVWYAKR', 'GTSNQDEHP'],
                4: ['C', 'KR', 'MFILVWYA', 'GTSNQDEHP'],
                5: ['C', 'KR', 'MFILVWYA', 'DE', 'GTSNQHP'],
                6: ['C', 'KR', 'WYA', 'MFILV', 'DE', 'GTSNQHP'],
                7: ['C', 'KR', 'WYA', 'MFILV', 'DE', 'QH', 'GTSNP'],
                8: ['C', 'KR', 'WYA', 'MFILV', 'D', 'E', 'QH', 'GTSNP'],
                9: ['C', 'KR', 'WYA', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                10: ['C', 'KR', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                11: ['C', 'K', 'R', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                12: ['C', 'K', 'R', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                13: ['C', 'K', 'R', 'W', 'Y', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                14: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                15: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'GS', 'N'],
                16: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'G', 'S', 'N'],
                17: ['C', 'K', 'R', 'W', 'Y', 'A', 'FI', 'LV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'G', 'S', 'N'],
                18: ['C', 'K', 'R', 'W', 'Y', 'A', 'FI', 'LV', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
                19: ['C', 'K', 'R', 'W', 'Y', 'A', 'F', 'I', 'LV', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
                20: ['C', 'K', 'R', 'W', 'Y', 'A', 'F', 'I', 'L', 'V', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type7')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type7')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_8(self):
        try:
            AAGroup = {
                2: ['ADEGKNPQRST', 'CFHILMVWY'],
                3: ['ADEGNPST', 'CHKQRW', 'FILMVY'],
                4: ['AGNPST', 'CHWY', 'DEKQR', 'FILMV'],
                5: ['AGPST', 'CFWY', 'DEN', 'HKQR', 'ILMV'],
                6: ['APST', 'CW', 'DEGN', 'FHY', 'ILMV', 'KQR'],
                7: ['AGST', 'CW', 'DEN', 'FY', 'HP', 'ILMV', 'KQR'],
                8: ['AST', 'CG', 'DEN', 'FY', 'HP', 'ILV', 'KQR', 'MW'],
                9: ['AST', 'CW', 'DE', 'FY', 'GN', 'HQ', 'ILV', 'KR', 'MP'],
                10: ['AST', 'CW', 'DE', 'FY', 'GN', 'HQ', 'IV', 'KR', 'LM', 'P'],
                11: ['AST', 'C', 'DE', 'FY', 'GN', 'HQ', 'IV', 'KR', 'LM', 'P', 'W'],
                12: ['AST', 'C', 'DE', 'FY', 'G', 'HQ', 'IV', 'KR', 'LM', 'N', 'P', 'W'],
                13: ['AST', 'C', 'DE', 'FY', 'G', 'H', 'IV', 'KR', 'LM', 'N', 'P', 'Q', 'W'],
                14: ['AST', 'C', 'DE', 'FL', 'G', 'H', 'IV', 'KR', 'M', 'N', 'P', 'Q', 'W', 'Y'],
                15: ['AST', 'C', 'DE', 'F', 'G', 'H', 'IV', 'KR', 'L', 'M', 'N', 'P', 'Q', 'W', 'Y'],
                16: ['AT', 'C', 'DE', 'F', 'G', 'H', 'IV', 'KR', 'L', 'M', 'N', 'P', 'Q', 'S', 'W', 'Y'],
                17: ['AT', 'C', 'DE', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'W', 'Y'],
                18: ['A', 'C', 'DE', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
                19: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'V', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type8')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type8')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_9(self):
        try:
            AAGroup = {
                2: ['ACDEFGHILMNPQRSTVWY', 'K'],
                3: ['ACDFGMPQRSTW', 'EHILNVY', 'K'],
                4: ['AGPT', 'CDFMQRSW', 'EHILNVY', 'K'],
                5: ['AGPT', 'CDQ', 'EHILNVY', 'FMRSW', 'K'],
                6: ['AG', 'CDQ', 'EHILNVY', 'FMRSW', 'K', 'PT'],
                7: ['AG', 'CDQ', 'EHNY', 'FMRSW', 'ILV', 'K', 'PT'],
                8: ['AG', 'C', 'DQ', 'EHNY', 'FMRSW', 'ILV', 'K', 'PT'],
                9: ['AG', 'C', 'DQ', 'EHNY', 'FMW', 'ILV', 'K', 'PT', 'RS'],
                10: ['A', 'C', 'DQ', 'EHNY', 'FMW', 'G', 'ILV', 'K', 'PT', 'RS'],
                11: ['A', 'C', 'DQ', 'EHNY', 'FM', 'G', 'ILV', 'K', 'PT', 'RS', 'W'],
                12: ['A', 'C', 'DQ', 'EHNY', 'FM', 'G', 'IL', 'K', 'PT', 'RS', 'V', 'W'],
                13: ['A', 'C', 'DQ', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'RS', 'V', 'W'],
                14: ['A', 'C', 'D', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'Q', 'RS', 'V', 'W'],
                15: ['A', 'C', 'D', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'Q', 'R', 'S', 'V', 'W'],
                16: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'IL', 'K', 'M', 'PT', 'Q', 'R', 'S', 'V', 'W'],
                17: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'IL', 'K', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'],
                18: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'],
                19: ['A', 'C', 'D', 'E', 'F', 'G', 'HN', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'N', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type9')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type9')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_10(self):
        try:
            AAGroup = {
                2: ['CMFILVWY', 'AGTSNQDEHRKP'],
                3: ['CMFILVWY', 'AGTSP', 'NQDEHRK'],
                4: ['CMFWY', 'ILV', 'AGTS', 'NQDEHRKP'],
                5: ['FWYH', 'MILV', 'CATSP', 'G', 'NQDERK'],
                6: ['FWYH', 'MILV', 'CATS', 'P', 'G', 'NQDERK'],
                7: ['FWYH', 'MILV', 'CATS', 'P', 'G', 'NQDE', 'RK'],
                8: ['FWYH', 'MILV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                9: ['FWYH', 'ML', 'IV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                10: ['FWY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'DE', 'QRK'],
                11: ['FWY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                12: ['FWY', 'ML', 'IV', 'C', 'A', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                13: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                14: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'R', 'K'],
                15: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                16: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                17: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                18: ['W', 'FY', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                19: ['W', 'F', 'Y', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                20: ['W', 'F', 'Y', 'M', 'L', 'I', 'V', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'tpye10')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type10')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_11(self):
        try:
            AAGroup = {
                2: ['CFYWMLIV', 'GPATSNHQEDRK'],
                3: ['CFYWMLIV', 'GPATS', 'NHQEDRK'],
                4: ['CFYW', 'MLIV', 'GPATS', 'NHQEDRK'],
                5: ['CFYW', 'MLIV', 'G', 'PATS', 'NHQEDRK'],
                6: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NHQEDRK'],
                7: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NHQED', 'RK'],
                8: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                9: ['CFYW', 'ML', 'IV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                10: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                11: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'TS', 'NH', 'QED', 'RK'],
                12: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'TS', 'NH', 'QE', 'D', 'RK'],
                13: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'NH', 'QE', 'D', 'RK'],
                14: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'RK'],
                15: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'R', 'K'],
                16: ['C', 'FY', 'W', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'R', 'K'],
                17: ['C', 'FY', 'W', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                18: ['C', 'FY', 'W', 'M', 'L', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                19: ['C', 'F', 'Y', 'W', 'M', 'L', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                20: ['C', 'F', 'Y', 'W', 'M', 'L', 'I', 'V', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type11')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type11')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_12(self):
        try:
            AAGroup = {
                2: ['IVMLFWYC', 'ARNDQEGHKPST'],
                3: ['IVLMFWC', 'YA', 'RNDQEGHKPST'],
                4: ['IVLMFW', 'C', 'YA', 'RNDQEGHKPST'],
                5: ['IVLMFW', 'C', 'YA', 'G', 'RNDQEHKPST'],
                6: ['IVLMF', 'WY', 'C', 'AH', 'G', 'RNDQEKPST'],
                7: ['IVLMF', 'WY', 'C', 'AH', 'GP', 'R', 'NDQEKST'],
                8: ['IVLMF', 'WY', 'C', 'A', 'G', 'R', 'Q', 'NDEHKPST'],
                9: ['IVLMF', 'WY', 'C', 'A', 'G', 'P', 'H', 'K', 'RNDQEST'],
                10: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'RN', 'DQEKPST'],
                11: ['IVLMF', 'W', 'Y', 'C', 'A', 'H', 'G', 'R', 'N', 'Q', 'DEKPST'],
                12: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'T', 'RDEKPS'],
                13: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'DEKST'],
                14: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'DEST'],
                15: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'D', 'EST'],
                16: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'DE'],
                17: ['IVL', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'DE'],
                18: ['IVL', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'D', 'E'],
                20: ['I', 'V', 'L', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'D', 'E'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']
            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False
            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i
            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))
            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type12')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type12')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_13(self):
        try:
            AAGroup = {
                4: ['ADKERNTSQ', 'YFLIVMCWH', 'G', 'P'],
                12: ['A', 'D', 'KER', 'N', 'TSQ', 'YF', 'LIVM', 'C', 'W', 'H', 'G', 'P'],
                17: ['A', 'D', 'KE', 'R', 'N', 'T', 'S', 'Q', 'Y', 'F', 'LIV', 'M', 'C', 'W', 'H', 'G', 'P'],
                20: ['A', 'D', 'K', 'E', 'R', 'N', 'T', 'S', 'Q', 'Y', 'F', 'L', 'I', 'V', 'M', 'C', 'W', 'H', 'G', 'P'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type13')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type13')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_14(self):
        try:
            AAGroup = {
                2: ['ARNDCQEGHKPST', 'ILMFWYV'],
                3: ['ARNDQEGHKPST', 'C', 'ILMFWYV'],
                4: ['ARNDQEGHKPST', 'C', 'ILMFYV', 'W'],
                5: ['AGPST', 'RNDQEHK', 'C', 'ILMFYV', 'W'],
                6: ['AGPST', 'RNDQEK', 'C', 'H', 'ILMFYV', 'W'],
                7: ['ANDGST', 'RQEK', 'C', 'H', 'ILMFYV', 'P', 'W'],
                8: ['ANDGST', 'RQEK', 'C', 'H', 'ILMV', 'FY', 'P', 'W'],
                9: ['AGST', 'RQEK', 'ND', 'C', 'H', 'ILMV', 'FY', 'P', 'W'],
                10: ['AGST', 'RK', 'ND', 'C', 'QE', 'H', 'ILMV', 'FY', 'P', 'W'],
                11: ['AST', 'RK', 'ND', 'C', 'QE', 'G', 'H', 'ILMV', 'FY', 'P', 'W'],
                12: ['AST', 'RK', 'ND', 'C', 'QE', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                13: ['AST', 'RK', 'N', 'D', 'C', 'QE', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                14: ['AST', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                15: ['A', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'ST', 'W'],
                16: ['A', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'F', 'P', 'ST', 'W', 'Y'],
                17: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'K', 'F', 'P', 'ST', 'W', 'Y'],
                18: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'K', 'F', 'P', 'S', 'T', 'W', 'Y'],
                19: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y'],
                20: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'V', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type14')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type14')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_15(self):
        try:
            AAGroup = {
                2: ['MFILVAW', 'CYQHPGTSNRKDE'],
                3: ['MFILVAW', 'CYQHPGTSNRK', 'DE'],
                4: ['MFILV', 'ACW', 'YQHPGTSNRK', 'DE'],
                5: ['MFILV', 'ACW', 'YQHPGTSN', 'RK', 'DE'],
                6: ['MFILV', 'A', 'C', 'WYQHPGTSN', 'RK', 'DE'],
                7: ['MFILV', 'A', 'C', 'WYQHP', 'GTSN', 'RK', 'DE'],
                8: ['MFILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'DE'],
                9: ['MF', 'ILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'DE'],
                10: ['MF', 'ILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'D', 'E'],
                11: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'D', 'E'],
                12: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'TS', 'N', 'RK', 'D', 'E'],
                13: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                14: ['MF', 'I', 'L', 'V', 'A', 'C', 'WYQHP', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                15: ['MF', 'IL', 'V', 'A', 'C', 'WYQ', 'H', 'P', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                16: ['MF', 'I', 'L', 'V', 'A', 'C', 'WYQ', 'H', 'P', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                20: ['M', 'F', 'I', 'L', 'V', 'A', 'C', 'W', 'Y', 'Q', 'H', 'P', 'G', 'T', 'S', 'N', 'R', 'K', 'D', 'E'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type15')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type15')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKRAAC_type_16(self):
        try:
            AAGroup = {
                2: ['IMVLFWY', 'GPCASTNHQEDRK'],
                3: ['IMVLFWY', 'GPCAST', 'NHQEDRK'],
                4: ['IMVLFWY', 'G', 'PCAST', 'NHQEDRK'],
                5: ['IMVL', 'FWY', 'G', 'PCAST', 'NHQEDRK'],
                6: ['IMVL', 'FWY', 'G', 'P', 'CAST', 'NHQEDRK'],
                7: ['IMVL', 'FWY', 'G', 'P', 'CAST', 'NHQED', 'RK'],
                8: ['IMV', 'L', 'FWY', 'G', 'P', 'CAST', 'NHQED', 'RK'],
                9: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'AST', 'NHQED', 'RK'],
                10: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'STNH', 'RKQE', 'D'],
                11: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'STNH', 'RKQ', 'E', 'D'],
                12: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'ST', 'N', 'HRKQ', 'E', 'D'],
                13: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'ST', 'N', 'HRKQ', 'E', 'D'],
                14: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'HRKQ', 'E', 'D'],
                15: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'RKQ', 'E', 'D'],
                16: ['IMV', 'L', 'F', 'W', 'Y', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'RKQ', 'E', 'D'],
                20: ['I', 'M', 'V', 'L', 'F', 'W', 'Y', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'R', 'K', 'Q', 'E', 'D'],
            }
            fastas = self.fasta_list
            subtype = self.__default_para['PseKRAAC_model']
            raactype = self.__default_para['RAAC_clust']
            ktuple = self.__default_para['k-tuple']
            glValue = self.__default_para['g-gap'] if subtype == 'g-gap' else self.__default_para['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type16')
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue, 'type16')
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')

class iDNA(Sequence):
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> dna = iFeatureOmegaCLI.iDNA("./data_examples/DNA_sequences.txt")

    # display available feature descriptor methods
    >>> dna.display_feature_types()

    # import parameters for feature descriptors (optimal)
    >>> dna.import_parameters('parameters/DNA_parameters_setting.json')

    # calculate feature descriptors. Take "Kmer" as an example.
    >>> dna.get_descriptor("Kmer")

    # display the feature descriptors
    >>> print(dna.encodings)

    # save feature descriptors
    >>> dna.to_csv("Kmer.csv", "index=False", header=False)
    """

    def __init__(self, file):
        super(iDNA, self).__init__(file=file)
        self.__default_para_dict = {
            'Kmer': {'kmer': 3},
            'RCKmer': {'kmer': 3},
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise'},
            'DPCP type2': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise'},
            'TPCP': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'TPCP type2': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'DAC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'DCC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'DACC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'TAC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TCC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TACC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'PseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'PCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'NMBroto': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Moran': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Geary': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},          
        }
        self.__default_para = {
            'kmer': 3,
            'sliding_window': 5,
            'kspace': 3,            
            'mismatch': 1,
            'delta': 0,
            'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
            'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',            
            'distance': 0,
            'cp': 'cp(20)',
            'nlag': 3,
            'lambdaValue': 3,
            'weight': 0.05,
        }
        self.encodings = None       # pandas dataframe
        self.__cmd_dict ={
            'Kmer': 'self._Kmer()',
            'RCKmer': 'self._RCKmer()',
            'Mismatch': 'self._Mismatch()',
            'Subsequence': 'self._Subsequence()',
            'NAC': 'self._NAC()',
            'ANF': 'self._ANF()',
            'ENAC': 'self._ENAC()',
            'binary': 'self._binary()',
            'PS2': 'self._PS2()',
            'PS3': 'self._PS3()',
            'PS4': 'self._PS4()',
            'CKSNAP': 'self._CKSNAP()',
            'NCP': 'self._NCP()',
            'EIIP': 'self._EIIP()',
            'PseEIIP': 'self._PseEIIP()',
            'ASDC': 'self._ASDC()',
            'DBE': 'self._DBE()',
            'LPDF': 'self._LPDF()',
            'DPCP': 'self._DPCP()',
            'DPCP type2': 'self._DPCP_type2()',
            'TPCP': 'self._TPCP()',
            'TPCP type2': 'self._TPCP_type2()',
            'MMI': 'self._MMI()',
            'Z_curve_9bit': 'self._Z_curve_9bit()',
            'Z_curve_12bit': 'self._Z_curve_12bit()',
            'Z_curve_36bit': 'self._Z_curve_36bit()',
            'Z_curve_48bit': 'self._Z_curve_48bit()',
            'Z_curve_144bit': 'self._Z_curve_144bit()',
            'NMBroto': 'self._NMBroto()',
            'Moran': 'self._Moran()',
            'Geary': "self._Geary()",
            'DAC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'DCC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'DACC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'TAC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'TCC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'TACC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'PseDNC': 'self._PseDNC(my_property_name, my_property_value)',
            'PseKNC': 'self._PseKNC(my_property_name, my_property_value)',
            'PCPseDNC': 'self._PCPseDNC(my_property_name, my_property_value)',
            'PCPseTNC': 'self._PCPseTNC(my_property_name, my_property_value)',
            'SCPseDNC': 'self._SCPseDNC(my_property_name, my_property_value)',
            'SCPseTNC': 'self._SCPseTNC(my_property_name, my_property_value)',
        }

        # variable for DAC, DCC, DACC, TAC, TCC, TACC
        self.didna_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content', 'A-philicity',
                    'Propeller twist', 'Duplex stability:(freeenergy)',
                    'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                    'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                    'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                    'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                    'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                    'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
                    'Shift', 'Slide', 'Rise',
                    'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
                    'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
                    'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
                    'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
                    'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3', 'Propeller Twist',
                    'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
                    'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift', 'Slide stiffness',
                    'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
                    'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
                    'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size', 'Twist_rise',
                    'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
                    'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width', 'Major Groove Depth',
                    'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
                    'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
                    'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content', 'Slide_rise',
                    'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width', 'Inclination',
                    'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2', 'Twist5',
                    'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide', 'Minor Groove Depth',
                    'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
                    'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
                    'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
                    'Mobility to bend towards minor groove']
        self.tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                    'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                    'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
        self.dirna_list = ['Slide (RNA)', 'Adenine content', 'Hydrophilicity (RNA)', 'Tilt (RNA)', 'Stacking energy (RNA)',
                    'Twist (RNA)', 'Entropy (RNA)', 'Roll (RNA)', 'Purine (AG) content', 'Hydrophilicity (RNA)1',
                    'Enthalpy (RNA)1', 'GC content', 'Entropy (RNA)1', 'Rise (RNA)', 'Free energy (RNA)',
                    'Keto (GT) content', 'Free energy (RNA)1', 'Enthalpy (RNA)', 'Guanine content', 'Shift (RNA)',
                    'Cytosine content', 'Thymine content']
        self.myDict = {
            'DAC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'DCC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'DACC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'TAC': {'DNA': self.tridna_list, 'RNA': []},
            'TCC': {'DNA': self.tridna_list, 'RNA': []},
            'TACC': {'DNA': self.tridna_list, 'RNA': []},
            'PseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PseKNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PCPseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PCPseTNC': {'DNA': self.tridna_list, 'RNA': []},
            'SCPseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'SCPseTNC': {'DNA': self.tridna_list, 'RNA': []},
        }
        self.myDictDefault = {
            'DAC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'DCC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'DACC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'TAC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'TCC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'TACC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'PseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PCPseDNC': {
                'DNA': ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 'Propeller twist',
                        'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                        'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Electron_interaction',
                        'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Lisser_BZ_transition', 'Polar_interaction',
                        'SantaLucia_dG', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH',
                        'Sugimoto_dS', 'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                        'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                        'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'],
                'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                        'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
        }
        self.myKmer = {
            'DAC': 2, 'DCC': 2, 'DACC': 2,
            'TAC': 3, 'TCC': 3, 'TACC': 3
        }
        self.myDataFile = {
            'DAC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'DCC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'DACC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'TAC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'TCC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'TACC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'PseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
        }
        self.myDiIndex = {
            'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
            'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
            'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
            'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
        }
        self.myTriIndex = {
            'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
            'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
            'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
            'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
            'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
            'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
            'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
            'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
            'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
            'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
            'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
            'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
            'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
            'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
            'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
            'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
        }

    def __check_acc_arguments(self, method, type, parameters):
        """
        Check auto-correlation parameters.
        :param method: i.e. DAC, DCC, DACC, TAC, TCC, TACC
        :param type: i.e. DNA, RNA
        :return:
        """
        kmer = self.myKmer[method]
        myIndex = []
        myProperty = {}
        dataFile = ''
        if type == 'DNA':
            if method in ('DAC', 'DCC', 'DACC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            else:
                myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
        if type == 'RNA':
            if method in ('DAC', 'DCC', 'DACC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

        dataFile = self.myDataFile[method][type]
        file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', dataFile)        
        with open(file_path, 'rb') as f:
            myProperty = pickle.load(f)
        if len(myIndex) == 0 or len(myProperty) == 0:
            return myIndex, myProperty, kmer, False
        return myIndex, myProperty, kmer, True

    def __check_Pse_arguments(self, method, type, parameters):
        """
        Check auto-correlation parameters.
        :param method: i.e. PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
        :param type: i.e. DNA, RNA
        :return:
        """
        myIndex = []
        myProperty = {}
        dataFile = ''

        if type == 'DNA':
            if method in ('PseDNC', 'PseKNC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            if method in ('PCPseDNC', 'SCPseDNC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            if method in ('PCPseTNC', 'SCPseTNC'):
                myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
        if type == 'RNA':
            if method in ('PseDNC', 'PseKNC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')
            if method in ('PCPseDNC', 'SCPseDNC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

        dataFile = self.myDataFile[method][type]
        file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', dataFile)        
        with open(file_path, 'rb') as f:
            myProperty = pickle.load(f)

        if len(myIndex) == 0 or len(myProperty) == 0:
            return myIndex, myProperty, False
        return myIndex, myProperty, True    

    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        Kmer                      The cccurrence frequencies of k neighboring nucleic acids
        RCKmer                    Reverse compliment Kmer
        Mismatch                  Mismatch profile
        Subsequence               Subsequence profile
        NAC                       Nucleic acid composition
        ANF                       Accumulated nucleotide frequency
        ENAC                      Enhanced nucleic acid composition
        binary                    DNA binary
        PS2                       Position-specific of two nucleotides
        PS3                       Position-specific of three nucleotides
        PS4                       Postiion-specific of four nucleotides
        CKSNAP                    Composition of k-spaced Nucleic acid pairs
        NCP                       Nucleotide chemical property
        EIIP                      Electron-ion interaction pseudopotentials value
        PseEIIP                   Electron-ion interaction pseudopotentials of trinucleotide
        ASDC                      Adaptive skip dinucleotide composition
        DBE                       Dinucleotide binary encoding
        LPDF                      Local position-specific dinucleotide frequency
        DPCP                      Dinucleotide physicochemical properties
        DPCP type2                Dinucleotide physicochemical properties type 2
        TPCP                      Trinucleotide physicochemical properties
        TPCP type2                Trinucleotide physicochemical properties type 2
        MMI                       Multivariate mutual information
        Z_curve_9bit              The Z curve parameters for frequencies of phase-specific mononucleotides
        Z_curve_12bit             The Z curve parameters for frequencies of phase-independent dinucleotides
        Z_curve_36bit             The Z curve parameters for frequencies of phase-specific dinucleotides
        Z_curve_48bit             The Z curve parameters for frequencies of phase-independent trinucleotides
        Z_curve_144bit            The Z curve parameters for frequencies of phase-specific trinucleotides
        NMBroto                   Normalized Moreau-Broto
        Moran                     Moran
        Geary                     Geary
        DAC                       Dinucleotide-based auto covariance
        DCC                       Dinucleotide-based cross covariance
        DACC                      Dinucleotide-based auto-cross covariance 
        TAC                       Trinucleotide-based auto covariance 
        TCC                       Trinucleotide-based cross covariance  
        TACC                      Trinucleotide-based auto-cross covariance  
        PseDNC                    Pseudo dinucleotide composition  
        PseKNC                    Pseudo k-tupler composition    
        PCPseDNC                  Parallel correlation pseudo dinucleotide composition  
        PCPseTNC                  Parallel correlation pseudo trinucleotide composition     
        SCPseDNC                  Series correlation pseudo dinucleotide composition   
        SCPseTNC                  Series correlation pseudo trinucleotide composition      

        Note: the first column is the names of availables while the second column
              is description.  
        
        '''
        
        print(info)

    def import_parameters(self, file):
        if os.path.exists(file):
            with open(file) as f:
                records = f.read().strip()
            try:
                self.__default_para_dict = json.loads(records)
                print('File imported successfully.')
            except Exception as e:
                print('Parameter file parser error.')
    
    def get_descriptor(self, descriptor='Kmer'):
        # copy parameters
        if descriptor in self.__default_para_dict:
            for key in self.__default_para_dict[descriptor]:                
                self.__default_para[key] = self.__default_para_dict[descriptor][key]
            
        if descriptor in self.__cmd_dict:
            if descriptor in ['DAC', 'TAC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['DCC', 'TCC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['DACC', 'TACC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                my_property_name, my_property_value, ok = self.__check_Pse_arguments(descriptor, self.sequence_type, self.__default_para)            
            cmd = self.__cmd_dict[descriptor]
            status = eval(cmd)
        else:
            print('The descriptor type does not exist.')

    def kmerArray(self, sequence, k):
        kmer = []
        for i in range(len(sequence) - k + 1):
            kmer.append(sequence[i:i + k])
        return kmer

    def _Kmer(self):
        try:
            fastas = self.fasta_list
            k = self.__default_para['kmer']
            upto = False
            normalize = True
            type = self.sequence_type

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'
            if type in ("DNA", 'RNA'):
                NA = 'ACGT'
            else:
                NA = 'ACDEFGHIKLMNPQRSTVWY'

            if upto == True:
                tmp_header = ['Sample', 'label']
                for tmpK in range(1, k + 1):
                    for kmer in itertools.product(NA, repeat=tmpK):
                        header.append('Kmer_' + ''.join(kmer))
                        tmp_header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(tmp_header)):
                        if tmp_header[j] in count:
                            code.append(count[tmp_header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmp_header = ['Sample', 'label']
                for kmer in itertools.product(NA, repeat=k):
                    header.append('Kmer_' + ''.join(kmer))
                    tmp_header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(tmp_header)):
                        if tmp_header[j] in count:
                            code.append(count[tmp_header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)            
            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)            
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def imismatch_count(self, seq1, seq2):
        imismatch = 0
        min_number = np.min(np.array([len(seq1), len(seq2)]))
        for i in range(min_number):
            if seq1[i] != seq2[i]:
                imismatch += 1        
        return imismatch

    def _Mismatch(self):
        try:
            k = self.__default_para['kmer']
            m = self.__default_para['mismatch']
            NN = 'ACGT'

            encoding = []
            header = ['SampleName', 'label']
            template_dict = {}
            for kmer in itertools.product(NN, repeat=k):
                header.append('Mismatch_' + ''.join(kmer))
                template_dict[''.join(kmer)] = 0
            encoding.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                kmers = self.kmerArray(sequence, k)
                tmp_dict = template_dict.copy()
                for kmer in kmers:
                    for key in tmp_dict:
                        if self.imismatch_count(kmer, key) <= m:
                            tmp_dict[key] += 1
                        
                code = [name, label] + [tmp_dict[key] for key in sorted(tmp_dict.keys())]
                encoding.append(code)
            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)            
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ Subsequence """    
    def GetKmerDict(self, piRNAletter, k):
        kmerlst = []
        partkmers = list(itertools.combinations_with_replacement(piRNAletter, k))
        for element in partkmers:
            elelst = set(itertools.permutations(element, k))
            strlst = [''.join(ele) for ele in elelst]
            kmerlst += strlst
        kmerlst = np.sort(kmerlst)
        kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
        return kmerdict

    def GetSubsequenceProfile(self, instances, piRNAletter, k, delta):
        kmerdict = self.GetKmerDict(piRNAletter, k)
        X = []
        for sequence in instances:
            vector = self.GetSubsequenceProfileVector(sequence, kmerdict, k, delta)
            X.append(vector)
        X = np.array(X)
        return X

    def GetSubsequenceProfileVector(self, sequence, kmerdict, k, delta):
        vector = np.zeros((1, len(kmerdict)))
        sequence = np.array(list(sequence))
        n = len(sequence)
        index_lst = list(itertools.combinations(range(n), k))
        for subseq_index in index_lst:
            subseq_index = list(subseq_index)
            subsequence = sequence[subseq_index]
            position = kmerdict.get(''.join(subsequence))
            subseq_length = subseq_index[-1] - subseq_index[0] + 1
            subseq_score = 1 if subseq_length == k else delta ** subseq_length
            vector[0, position] += subseq_score
        return list(vector[0])

    def _Subsequence(self):
        try:
            k = self.__default_para['kmer']
            delta = self.__default_para['delta']
            NN_list = ['A', 'C', 'G', 'T']

            encoding = []
            header = ['SampleName', 'label']
            for kmer in itertools.product(NN_list, repeat=k):
                header.append('Subsequence_' + ''.join(kmer))
            # encoding.append(header)

            instances = [elem[1] for elem in self.fasta_list]

            code = self.GetSubsequenceProfile(instances, NN_list, k, delta)
            info_list = np.array([[item[0], item[2]] for item in self.fasta_list])
            code = np.hstack((info_list, code))
            encoding = np.vstack((header, code))

            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ end of Subsequence """

    def RC(self, kmer):
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        return ''.join([myDict[nc] for nc in kmer[::-1]])

    def generateRCKmer(self, kmerList):
        rckmerList = set()
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        for kmer in kmerList:
            rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
        return sorted(rckmerList)

    def _RCKmer(self):
        try:
            fastas = self.fasta_list
            k = self.__default_para['kmer']
            upto = False
            normalize = True

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'

            if upto == True:
                for tmpK in range(1, k + 1):
                    tmpHeader = []
                    for kmer in itertools.product(NA, repeat=tmpK):
                        tmpHeader.append(''.join(kmer))
                    header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        for j in range(len(kmers)):
                            if kmers[j] in myDict:
                                kmers[j] = myDict[kmers[j]]
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmpHeader = []
                for kmer in itertools.product(NA, repeat=k):
                    tmpHeader.append(''.join(kmer))
                header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer

                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    for j in range(len(kmers)):
                        if kmers[j] in myDict:
                            kmers[j] = myDict[kmers[j]]
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
    
    def _NAC(self):
        try:
            NA = 'ACGT'
            encoding = []
            header = ['SampleName']
            for i in NA:
                header.append('NAC_{0}'.format(i))
            encoding.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name]
                for na in NA:
                    code.append(count[na])
                encoding.append(code)
            encoding = np.array(encoding)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ANF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ANF descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) + 1):
                header.append('ANF_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence)):
                    code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _NCP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'NCP descriptor need fasta sequence with equal length.'
                return False
            chemical_property = {
                'A': [1, 1, 1],
                'C': [0, 1, 0],
                'G': [1, 0, 0],
                'T': [0, 0, 1],
                'U': [0, 0, 1],
                '-': [0, 0, 0],
            }
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 3 + 1):
                header.append('NCP_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code = code + chemical_property.get(aa, [0, 0, 0])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ENAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ENAC descriptor need fasta sequence with equal length.'
                return False
            window = self.__default_para['sliding_window']
            if self.minimum_length < window:
                self.error_msg = 'ENAC descriptor, all the sequence length should be larger than the sliding window'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for aa in AA:
                    header.append('ENAC_sw.' + str(w) + '.' + aa)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence)):
                    if j < len(sequence) and j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        for key in count:
                            count[key] = count[key] / len(sequence[j:j + window])
                        for aa in AA:
                            code.append(count[aa])
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 4 + 1):
                header.append('Binary_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _CKSNAP(self):
        try:
            gap = self.__default_para['kspace']
            if self.minimum_length_without_minus < gap + 2:
                self.error_msg = 'CKSNAP - all the sequence length should be larger than the (gap value) + 2 = %s.' % (
                        gap + 2)
                return False

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append('CKSNAP_' + aa + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _EIIP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EIIP descriptor need fasta sequence with equal length.'
                return False
            fastas = self.fasta_list
            AA = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
                '-': 0,
            }
            encodings = []
            header = ['SampleName']
            for i in range(1, len(fastas[0][1]) + 1):
                header.append('EIIP_' + str(i))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code.append(EIIP_dict.get(aa, 0))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TriNcleotideComposition(self, sequence, base):
        trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
        tnc_dict = {}
        for triN in trincleotides:
            tnc_dict[triN] = 0
        for i in range(len(sequence) - 2):
            tnc_dict[sequence[i:i + 3]] += 1
        for key in tnc_dict:
            tnc_dict[key] /= (len(sequence) - 2)
        return tnc_dict

    def _PseEIIP(self):
        try:
            fastas = self.fasta_list
            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.'
                    return False
            base = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
            }
            trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
            EIIPxyz = {}
            for triN in trincleotides:
                EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

            encodings = []
            header = ['SampleName'] + ['PseEIIP_{0}'.format(i) for i in trincleotides]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                trincleotide_frequency = self.TriNcleotideComposition(sequence, base)
                code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ASDC(self):
        try:
            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName']
            header += ['ASDC_' + aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DBE(self):
        try:
            if not self.is_equal:
                self.error_msg = 'DBE descriptor need fasta sequence with equal length.'
                return False

            AA_dict = {
                'AA': [0, 0, 0, 0],
                'AC': [0, 0, 0, 1],
                'AG': [0, 0, 1, 0],
                'AT': [0, 0, 1, 1],
                'CA': [0, 1, 0, 0],
                'CC': [0, 1, 0, 1],
                'CG': [0, 1, 1, 0],
                'CT': [0, 1, 1, 1],
                'GA': [1, 0, 0, 0],
                'GC': [1, 0, 0, 1],
                'GG': [1, 0, 1, 0],
                'GT': [1, 0, 1, 1],
                'TA': [1, 1, 0, 0],
                'TC': [1, 1, 0, 1],
                'TG': [1, 1, 1, 0],
                'TT': [1, 1, 1, 1],
            }

            encodings = []
            header = ['SampleName']
            for i in range((len(self.fasta_list[0][1]) - 1) * 4):
                header.append('DBE_%s' %(i + 1))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 1):
                    if sequence[j] + sequence[j + 1] in AA_dict:
                        code += AA_dict[sequence[j] + sequence[j + 1]]
                    else:
                        code += [0.5, 0.5, 0.5, 0.5]
                encodings.append(code)

            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _LPDF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'LPDF descriptor need fasta sequence with equal length.'
                return False

            encodings = []
            header = ['SampleName']
            header += ['LPDF_%s' % i for i in range(1, len(self.fasta_list[0][1]))]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                dinucleotide_dict = {
                    'AA': 0,
                    'AC': 0,
                    'AG': 0,
                    'AT': 0,
                    'A-': 0,
                    'CA': 0,
                    'CC': 0,
                    'CG': 0,
                    'CT': 0,
                    'C-': 0,
                    'GA': 0,
                    'GC': 0,
                    'GG': 0,
                    'GT': 0,
                    'G-': 0,
                    'TA': 0,
                    'TC': 0,
                    'TG': 0,
                    'TT': 0,
                    'T-': 0,
                    '-A': 0,
                    '-C': 0,
                    '-G': 0,
                    '-T': 0,
                    '--': 0,
                }
                for j in range(1, len(sequence)):
                    dinucleotide_dict[sequence[j] + sequence[j - 1]] += 1
                    code.append(dinucleotide_dict[sequence[j] + sequence[j - 1]] / j)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DPCP(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                print(self.__default_para)
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')

            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 + '_' + p_name for p_name in property_name for n1 in base for n2 in base]
            header = ['SampleName'] + ['DPCP_{0}'.format(i) for i in dinucleotides]
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DPCP_type2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'DPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')

            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(len(self.fasta_list[0][1]) - 1):
                for p_name in property_name:
                    header.append('DPCP2_' + p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for p_name in property_name:
                    for j in range(len(sequence) - 1):
                        if sequence[j: j + 2] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 2]]])
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _TPCP(self):
        try:
            file_name = 'tridnaPhyche.data'
            property_name = self.__default_para['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
                property_name = property_dict.keys()
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 + '_' + p_name for p_name in property_name for aa1 in AA for aa2 in AA for aa3 in
                        AA]
            header = ['SampleName'] + ['TPCP_{0}'.format(i) for i in triPeptides]
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _TPCP_type2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'TPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'tridnaPhyche.data'
                property_name = self.__default_para['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(len(self.fasta_list[0][1]) - 2):
                for p_name in property_name:
                    header.append('TPCP2_' + p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for p_name in property_name:
                    for j in range(len(sequence) - 2):
                        if sequence[j: j + 3] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 3]]])
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _MMI(self):
        try:
            NA = 'ACGT'
            dinucleotide_list = [a1 + a2 for a1 in NA for a2 in NA]
            trinucleotide_list = [a1 + a2 + a3 for a1 in NA for a2 in NA for a3 in NA]
            dinucleotide_dict = {}
            trinucleotide_dict = {}
            for elem in dinucleotide_list:
                dinucleotide_dict[''.join(sorted(elem))] = 0
            for elem in trinucleotide_list:
                trinucleotide_dict[''.join(sorted(elem))] = 0

            encodings = []
            header = ['SampleName']
            header += ['MMI_%s' % elem for elem in sorted(dinucleotide_dict.keys())]
            header += ['MMI_%s' % elem for elem in sorted(trinucleotide_dict.keys())]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                f1_dict = {
                    'A': 0,
                    'C': 0,
                    'G': 0,
                    'T': 0,
                }
                f2_dict = dinucleotide_dict.copy()
                f3_dict = trinucleotide_dict.copy()

                for elem in sequence:
                    if elem in f1_dict:
                        f1_dict[elem] += 1
                for key in f1_dict:
                    f1_dict[key] /= len(sequence)

                for i in range(len(sequence) - 1):
                    if ''.join(sorted(sequence[i: i + 2])) in f2_dict:
                        f2_dict[''.join(sorted(sequence[i: i + 2]))] += 1
                for key in f2_dict:
                    f2_dict[key] /= (len(sequence) - 1)

                for i in range(len(sequence) - 2):
                    if ''.join(sorted(sequence[i: i + 3])) in f3_dict:
                        f3_dict[''.join(sorted(sequence[i: i + 3]))] += 1
                for key in f3_dict:
                    f3_dict[key] /= (len(sequence) - 2)

                for key in sorted(f2_dict.keys()):
                    if f2_dict[key] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        code.append(f2_dict[key] * math.log(f2_dict[key] / (f1_dict[key[0]] * f1_dict[key[1]])))
                    else:
                        code.append(0)
                for key in sorted(f3_dict.keys()):
                    element_1 = 0
                    element_2 = 0
                    element_3 = 0
                    if f2_dict[key[0:2]] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        element_1 = f2_dict[key[0:2]] * math.log(f2_dict[key[0:2]] / (f1_dict[key[0]] * f1_dict[key[1]]))
                    if f2_dict[key[0] + key[2]] != 0 and f1_dict[key[2]] != 0:
                        element_2 = (f2_dict[key[0] + key[2]] / f1_dict[key[2]]) * math.log(
                            f2_dict[key[0] + key[2]] / f1_dict[key[2]])
                    if f2_dict[key[1:3]] != 0 and f3_dict[key] / f2_dict[key[1:3]] != 0:
                        element_3 = (f3_dict[key] / f2_dict[key[1:3]]) * math.log(f3_dict[key] / f2_dict[key[1:3]])
                    code.append(element_1 + element_2 - element_3)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS2 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 for a1 in AA for a2 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 16
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 1) * 16 + 1):
                header.append('PS2_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 1):
                    code += AA_dict.get(sequence[j: j+2], [0]*16)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS3 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 for a1 in AA for a2 in AA for a3 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 64
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 2) * 64 + 1):
                header.append('PS3_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 2):
                    code += AA_dict.get(sequence[j: j+3], [0]*64)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS4(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS4 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 + a4 for a1 in AA for a2 in AA for a3 in AA for a4 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 256
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 3) * 256 + 1):
                header.append('PS4_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 3):
                    code += AA_dict.get(sequence[j: j+4], [0]*256)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_9bit(self):
        try:
            encodings = []
            header = ['SampleName']
            for pos in range(1, 4):
                for elem in ['x', 'y', 'z']:
                    header.append('Zcurve9_%s.%s' %(pos, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence)):
                    if (i+1) % 3 == 1:
                        if sequence[i] in pos1_dict:
                            pos1_dict[sequence[i]] += 1
                        else:
                            pos1_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i] in pos2_dict:
                            pos2_dict[sequence[i]] += 1
                        else:
                            pos2_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i] in pos3_dict:
                            pos3_dict[sequence[i]] += 1
                        else:
                            pos3_dict[sequence[i]] = 1

                code += [
                    (pos1_dict.get('A', 0) + pos1_dict.get('G', 0) - pos1_dict.get('C', 0) - pos1_dict.get('T', 0)) / len(sequence), # x
                    (pos1_dict.get('A', 0) + pos1_dict.get('C', 0) - pos1_dict.get('G', 0) - pos1_dict.get('T', 0)) / len(sequence), # y
                    (pos1_dict.get('A', 0) + pos1_dict.get('T', 0) - pos1_dict.get('G', 0) - pos1_dict.get('C', 0)) / len(sequence)  # z
                    ]
                code += [
                    (pos2_dict.get('A', 0) + pos2_dict.get('G', 0) - pos2_dict.get('C', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('C', 0) - pos2_dict.get('G', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('T', 0) - pos2_dict.get('G', 0) - pos2_dict.get('C', 0)) / len(sequence)
                    ]
                code += [
                    (pos3_dict.get('A', 0) + pos3_dict.get('G', 0) - pos3_dict.get('C', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('C', 0) - pos3_dict.get('G', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('T', 0) - pos3_dict.get('G', 0) - pos3_dict.get('C', 0)) / len(sequence)
                    ]
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_12bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']
            for base in NN:
                for elem in ['x', 'y', 'z']:
                    header.append('Zcurve12_%s.%s' %(base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos_dict = {}
                for i in range(len(sequence) - 1):
                    if sequence[i: i+2] in pos_dict:
                        pos_dict[sequence[i: i+2]] += 1
                    else:
                        pos_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sC' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sT' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]

                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_36bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']

            for base in NN:
                for pos in range(1, 4):
                    for elem in ['x', 'y', 'z']:
                        header.append('Zcurve36_%s_%s.%s' %(pos, base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 1):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+2] in pos1_dict:
                            pos1_dict[sequence[i: i+2]] += 1
                        else:
                            pos1_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+2] in pos2_dict:
                            pos2_dict[sequence[i: i+2]] += 1
                        else:
                            pos2_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+2] in pos3_dict:
                            pos3_dict[sequence[i: i+2]] += 1
                        else:
                            pos3_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sT' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]
                    code += [
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sT' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                    code += [
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sT' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                encodings.append(code)

            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_48bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']
            for base in NN:
                for base1 in NN:
                    for elem in ['x', 'y', 'z']:
                        header.append('Zcurve48_%s%s.%s' %(base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos_dict = {}
                for i in range(len(sequence) - 2):
                    if sequence[i: i+3] in pos_dict:
                        pos_dict[sequence[i: i+3]] += 1
                    else:
                        pos_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sT' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_144bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']

            for base in NN:
                for base1 in NN:
                    for pos in range(1, 4):
                        for elem in ['x', 'y', 'z']:
                            header.append('Zcurve144_%s_%s%s.%s' %(pos, base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 2):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+3] in pos1_dict:
                            pos1_dict[sequence[i: i+3]] += 1
                        else:
                            pos1_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+3] in pos2_dict:
                            pos2_dict[sequence[i: i+3]] += 1
                        else:
                            pos2_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+3] in pos3_dict:
                            pos3_dict[sequence[i: i+3]] += 1
                        else:
                            pos3_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sT' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sT' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sT' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _NMBroto(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('NMBroto_'+p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    for d in range(1, nlag + 1):
                        try:
                            if N > nlag:
                                atsd = sum([float(property_dict[p_name][AADict[sequence[j: j+2]]]) * float(property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) for j in range(N-d)]) / (N - d)
                            else:
                                atsd = 0
                        except Exception as e:
                            atsd = 0
                        code.append(atsd)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Moran(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[
                #                 0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                # os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('Moran_' + p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i
            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i+2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Idup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) * (property_dict[p_name][AADict[sequence[j+d: j+d+2]]] - pmean) for j in range(N-d)]) / (N - d)
                            Iddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N-d)]) / N
                            code.append(Idup/Iddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Geary(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[
                #                 0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                #     os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('Geary_' + p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i + 2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Cdup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) ** 2 for j in range(N - d)]) / (2 * (N - d))
                            Cddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N - d)]) / (N - 1)
                            code.append(Cdup / Cddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generatePropertyPairs(self, myPropertyName):
        pairs = []
        for i in range(len(myPropertyName)):
            for j in range(i + 1, len(myPropertyName)):
                pairs.append([myPropertyName[i], myPropertyName[j]])
                pairs.append([myPropertyName[j], myPropertyName[i]])
        return pairs

    def _make_ac_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            header = ['SampleName']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s_%s.lag%d' % (desc, p, l))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]

                for p in myPropertyName:
                    meanValue = 0
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])                   
                    meanValue = meanValue / (len(sequence) - kmer + 1)
                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):                            
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)                        
                        code.append(acValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _make_cc_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = ['SampleName'] + [desc + '_' + n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]

                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _make_acc_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False

            header = ['SampleName']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = header + [desc + '_' + n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                ## Auto covariance
                for p in myPropertyName:
                    meanValue = 0                    
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])                    
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            # acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)                        
                        code.append(acValue)

                ## Cross covariance
                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_kmer_frequency(self, sequence, kmer):
        baseSymbol = 'ACGT'
        myFrequency = {}
        for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
            myFrequency[pep] = 0
        for i in range(len(sequence) - kmer + 1):
            myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
        for key in myFrequency:
            myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
        return myFrequency

    def correlationFunction(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
        return CC / len(myPropertyName)

    def correlationFunction_type2(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
        return CC

    def get_theta_array(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + self.correlationFunction(sequence[i:i + kmer],
                                                         sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                         myPropertyName, myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def get_theta_array_type2(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            for p in myPropertyName:
                theta = 0
                for i in range(len(sequence) - tmpLamada - kmer):
                    theta = theta + self.correlationFunction_type2(sequence[i:i + kmer],
                                                                   sequence[
                                                                   i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                                   myIndex,
                                                                   [p], myPropertyValue)
                thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def _PseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('PseDNC_' + pair)
            for k in range(1, lamadaValue + 1):
                header.append('PseDNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('PCPseDNC_' + pair)
            for k in range(1, lamadaValue + 1):
                header.append('PCPseDNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myTriIndex

            header = ['SampleName']
            for tripeptide in sorted(myIndex):
                header.append('PCPseTNC_' + tripeptide)
            for k in range(1, lamadaValue + 1):
                header.append('PCPseTNC_lamada_' + str(k))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _SCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('SCPseDNC_'+pair)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('SCPseDNC_lamada_' + str(k))

            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _SCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myTriIndex
            header = ['SampleName']
            for pep in sorted(myIndex):
                header.append('SCPseTNC_' + pep)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('SCPseTNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKNC(self, myPropertyName, myPropertyValue):
        try:
            baseSymbol = 'ACGT'
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            kmer = self.__default_para['kmer']           
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            header = header + sorted(['PseKNC_'+''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])
            for k in range(1, lamadaValue + 1):
                header.append('PseKNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                kmerFreauency = self.get_kmer_frequency(sequence, kmer)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
                    code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
                    code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')

class iRNA(Sequence):
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> rna = iFeatureOmegaCLI.iRNA("./data_examples/RNA_sequences.txt")

    # display available feature descriptor methods
    >>> rna.display_feature_types()

    # import parameters for feature descriptors (optimal)
    >>> rna.import_parameters('parameters/RNA_parameters_setting.json')

    # calculate feature descriptors. Take "Kmer" as an example.
    >>> rna.get_descriptor("Kmer")

    # display the feature descriptors
    >>> print(rna.encodings)

    # save feature descriptors
    >>> rna.to_csv("Kmer.csv", "index=False", header=False)
    """

    def __init__(self, file):
        super(iRNA, self).__init__(file=file)
        self.__default_para_dict = {
            'Kmer': {'kmer': 3},            
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'DPCP type2': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},            
            'DAC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DCC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DACC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},            
            'PseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},            
            'SCPseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},            
            'NMBroto': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'Moran': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'Geary': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},         
        }
        self.__default_para = {
            'sliding_window': 5,
            'kspace': 3,
            'kmer': 3,
            'mismatch': 1,
            'delta': 0,            
            'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
            'distance': 0,
            'cp': 'cp(20)',
            'nlag': 3,
            'lambdaValue': 3,
            'weight': 0.05,
        }
        self.encodings = None       # pandas dataframe
        self.__cmd_dict ={
            'Kmer': 'self._Kmer()',
            'Mismatch': 'self._Mismatch()',
            'Subsequence': 'self._Subsequence()',
            'NAC': 'self._NAC()',
            'ENAC': 'self._ENAC()',
            'ANF': 'self._ANF()',
            'NCP': 'self._NCP()',
            'binary': 'self._binary()',
            'PS2': 'self._PS2()',
            'PS3': 'self._PS3()',
            'PS4': 'self._PS4()',
            'CKSNAP': 'self._CKSNAP()',
            'ASDC': 'self._ASDC()',
            'DBE': 'self._DBE()',
            'LPDF': 'self._LPDF()',
            'DPCP': 'self._DPCP()',
            'DPCP type2': 'self._DPCP_type2()',
            'MMI': 'self._MMI()',
            'Z_curve_9bit': 'self._Z_curve_9bit()',
            'Z_curve_12bit': 'self._Z_curve_12bit()',
            'Z_curve_36bit': 'self._Z_curve_36bit()',
            'Z_curve_48bit': 'self._Z_curve_48bit()',
            'Z_curve_144bit': 'self._Z_curve_144bit()',
            'NMBroto': 'self._NMBroto()',
            'Moran': 'self._Moran()',
            'Geary': "self._Geary()",
            'DAC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'DCC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'DACC': 'self._make_ac_vector(my_property_name, my_property_value, my_kmer, descriptor)',
            'PseDNC': 'self._PseDNC(my_property_name, my_property_value)',
            'PseKNC': 'self._PseKNC(my_property_name, my_property_value)',
            'PCPseDNC': 'self._PCPseDNC(my_property_name, my_property_value)',
            'SCPseDNC': 'self._SCPseDNC(my_property_name, my_property_value)',            
        }

        # variable for DAC, DCC, DACC, TAC, TCC, TACC
        self.didna_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content', 'A-philicity',
                    'Propeller twist', 'Duplex stability:(freeenergy)',
                    'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                    'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                    'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                    'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                    'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                    'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
                    'Shift', 'Slide', 'Rise',
                    'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
                    'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
                    'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
                    'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
                    'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3', 'Propeller Twist',
                    'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
                    'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift', 'Slide stiffness',
                    'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
                    'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
                    'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size', 'Twist_rise',
                    'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
                    'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width', 'Major Groove Depth',
                    'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
                    'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
                    'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content', 'Slide_rise',
                    'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width', 'Inclination',
                    'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2', 'Twist5',
                    'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide', 'Minor Groove Depth',
                    'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
                    'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
                    'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
                    'Mobility to bend towards minor groove']
        self.tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                    'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                    'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
        self.dirna_list = ['Slide (RNA)', 'Adenine content', 'Hydrophilicity (RNA)', 'Tilt (RNA)', 'Stacking energy (RNA)',
                    'Twist (RNA)', 'Entropy (RNA)', 'Roll (RNA)', 'Purine (AG) content', 'Hydrophilicity (RNA)1',
                    'Enthalpy (RNA)1', 'GC content', 'Entropy (RNA)1', 'Rise (RNA)', 'Free energy (RNA)',
                    'Keto (GT) content', 'Free energy (RNA)1', 'Enthalpy (RNA)', 'Guanine content', 'Shift (RNA)',
                    'Cytosine content', 'Thymine content']
        self.myDict = {
            'DAC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'DCC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'DACC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'TAC': {'DNA': self.tridna_list, 'RNA': []},
            'TCC': {'DNA': self.tridna_list, 'RNA': []},
            'TACC': {'DNA': self.tridna_list, 'RNA': []},
            'PseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PseKNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PCPseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'PCPseTNC': {'DNA': self.tridna_list, 'RNA': []},
            'SCPseDNC': {'DNA': self.didna_list, 'RNA': self.dirna_list},
            'SCPseTNC': {'DNA': self.tridna_list, 'RNA': []},
        }
        self.myDictDefault = {
            'DAC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'DCC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'DACC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'TAC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'TCC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'TACC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'PseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                    'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PCPseDNC': {
                'DNA': ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 'Propeller twist',
                        'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                        'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Electron_interaction',
                        'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Lisser_BZ_transition', 'Polar_interaction',
                        'SantaLucia_dG', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH',
                        'Sugimoto_dS', 'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                        'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                        'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'],
                'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'PCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
            'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                        'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
            'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
        }
        self.myKmer = {
            'DAC': 2, 'DCC': 2, 'DACC': 2,
            'TAC': 3, 'TCC': 3, 'TACC': 3
        }
        self.myDataFile = {
            'DAC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'DCC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'DACC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'TAC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'TCC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'TACC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'PseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'PCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
            'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
            'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
        }
        self.myDiIndex = {
            'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
            'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
            'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
            'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
        }
        self.myTriIndex = {
            'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
            'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
            'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
            'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
            'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
            'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
            'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
            'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
            'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
            'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
            'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
            'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
            'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
            'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
            'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
            'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
        }

    def __check_acc_arguments(self, method, type, parameters):
        """
        Check auto-correlation parameters.
        :param method: i.e. DAC, DCC, DACC, TAC, TCC, TACC
        :param type: i.e. DNA, RNA
        :return:
        """
        kmer = self.myKmer[method]
        myIndex = []
        myProperty = {}
        dataFile = ''
        if type == 'DNA':
            if method in ('DAC', 'DCC', 'DACC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            else:
                myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
        if type == 'RNA':
            if method in ('DAC', 'DCC', 'DACC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

        dataFile = self.myDataFile[method][type]
        file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', dataFile)        
        with open(file_path, 'rb') as f:
            myProperty = pickle.load(f)
        if len(myIndex) == 0 or len(myProperty) == 0:
            return myIndex, myProperty, kmer, False
        return myIndex, myProperty, kmer, True

    def __check_Pse_arguments(self, method, type, parameters):
        """
        Check auto-correlation parameters.
        :param method: i.e. PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
        :param type: i.e. DNA, RNA
        :return:
        """
        myIndex = []
        myProperty = {}
        dataFile = ''

        if type == 'DNA':
            if method in ('PseDNC', 'PseKNC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            if method in ('PCPseDNC', 'SCPseDNC'):
                myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
            if method in ('PCPseTNC', 'SCPseTNC'):
                myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
        if type == 'RNA':
            if method in ('PseDNC', 'PseKNC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')
            if method in ('PCPseDNC', 'SCPseDNC'):
                myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

        dataFile = self.myDataFile[method][type]
        file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', dataFile)        
        with open(file_path, 'rb') as f:
            myProperty = pickle.load(f)

        if len(myIndex) == 0 or len(myProperty) == 0:
            return myIndex, myProperty, False
        return myIndex, myProperty, True    

    def import_parameters(self, file):
        if os.path.exists(file):
            with open(file) as f:
                records = f.read().strip()
            try:
                self.__default_para_dict = json.loads(records)
                print('File imported successfully.')
            except Exception as e:
                print('Parameter file parser error.')
    
    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        Kmer                      The cccurrence frequencies of k neighboring nucleic acids        
        Mismatch                  Mismatch profile
        Subsequence               Subsequence profile
        NAC                       Nucleic acid composition
        ANF                       Accumulated nucleotide frequency
        ENAC                      Enhanced nucleic acid composition
        binary                    DNA binary
        PS2                       Position-specific of two nucleotides
        PS3                       Position-specific of three nucleotides
        PS4                       Postiion-specific of four nucleotides
        CKSNAP                    Composition of k-spaced Nucleic acid pairs
        NCP                       Nucleotide chemical property        
        ASDC                      Adaptive skip dinucleotide composition
        DBE                       Dinucleotide binary encoding
        LPDF                      Local position-specific dinucleotide frequency
        DPCP                      Dinucleotide physicochemical properties
        DPCP type2                Dinucleotide physicochemical properties type 2
        MMI                       Multivariate mutual information
        Z_curve_9bit              The Z curve parameters for frequencies of phase-specific mononucleotides
        Z_curve_12bit             The Z curve parameters for frequencies of phase-independent dinucleotides
        Z_curve_36bit             The Z curve parameters for frequencies of phase-specific dinucleotides
        Z_curve_48bit             The Z curve parameters for frequencies of phase-independent trinucleotides
        Z_curve_144bit            The Z curve parameters for frequencies of phase-specific trinucleotides
        NMBroto                   Normalized Moreau-Broto
        Moran                     Moran
        Geary                     Geary
        DAC                       Dinucleotide-based auto covariance
        DCC                       Dinucleotide-based cross covariance
        DACC                      Dinucleotide-based auto-cross covariance
        PseDNC                    Pseudo dinucleotide composition  
        PseKNC                    Pseudo k-tupler composition    
        PCPseDNC                  Parallel correlation pseudo dinucleotide composition        
        SCPseDNC                  Series correlation pseudo dinucleotide composition         

        Note: the first column is the names of availables while the second column
              is description.  
        
        '''
        
        print(info)
    
    def get_descriptor(self, descriptor='Kmer'):
        # copy parameters
        if descriptor in self.__default_para_dict:
            for key in self.__default_para_dict[descriptor]:
                self.__default_para[key] = self.__default_para_dict[descriptor][key]
            
        if descriptor in self.__cmd_dict:
            if descriptor in ['DAC', 'TAC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['DCC', 'TCC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['DACC', 'TACC']:
                my_property_name, my_property_value, my_kmer, ok = self.__check_acc_arguments(descriptor, self.sequence_type, self.__default_para)
            if descriptor in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                my_property_name, my_property_value, ok = self.__check_Pse_arguments(descriptor, self.sequence_type, self.__default_para)            
            cmd = self.__cmd_dict[descriptor]
            status = eval(cmd)
            # print(status)
        else:
            print('The descriptor type does not exist.')

    def kmerArray(self, sequence, k):
        kmer = []
        for i in range(len(sequence) - k + 1):
            kmer.append(sequence[i:i + k])
        return kmer

    def _Kmer(self):
        try:
            fastas = self.fasta_list
            k = self.__default_para['kmer']
            upto = False
            normalize = True
            type = self.sequence_type

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'
            if type in ("DNA", 'RNA'):
                NA = 'ACGT'
            else:
                NA = 'ACDEFGHIKLMNPQRSTVWY'

            if upto == True:
                tmp_header = ['Sample', 'label']
                for tmpK in range(1, k + 1):
                    for kmer in itertools.product(NA, repeat=tmpK):
                        header.append('Kmer_' + ''.join(kmer))
                        tmp_header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(tmp_header)):
                        if tmp_header[j] in count:
                            code.append(count[tmp_header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmp_header = ['Sample', 'label']
                for kmer in itertools.product(NA, repeat=k):
                    header.append('Kmer_' + ''.join(kmer))
                    tmp_header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(tmp_header)):
                        if tmp_header[j] in count:
                            code.append(count[tmp_header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)            
            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)            
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def mismatch_count(self, seq1, seq2):
        mismatch = 0
        min_number = np.min(np.array([len(seq1), len(seq2)]))     
        for i in range(min_number):
            if seq1[i] != seq2[i]:
                mismatch += 1
        return mismatch

    def _Mismatch(self):
        try:
            k = self.__default_para['kmer']
            m = self.__default_para['mismatch']
            NN = 'ACGT'
            encoding = []
            header = ['SampleName']
            template_dict = {}
            for kmer in itertools.product(NN, repeat=k):
                header.append('Mismatch_' + ''.join(kmer))
                template_dict[''.join(kmer)] = 0
            encoding.append(header)
            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                kmers = self.kmerArray(sequence, k)
                tmp_dict = template_dict.copy()
                for kmer in kmers:
                    for key in tmp_dict:
                        if self.mismatch_count(kmer, key) <= m:
                            tmp_dict[key] += 1
                code = [name] + [tmp_dict[key] for key in sorted(tmp_dict.keys())]
                encoding.append(code)
            encoding = np.array(encoding)            
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True            
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ Subsequence """    
    def GetKmerDict(self, piRNAletter, k):
        kmerlst = []
        partkmers = list(itertools.combinations_with_replacement(piRNAletter, k))
        for element in partkmers:
            elelst = set(itertools.permutations(element, k))
            strlst = [''.join(ele) for ele in elelst]
            kmerlst += strlst
        kmerlst = np.sort(kmerlst)
        kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
        return kmerdict

    def GetSubsequenceProfile(self, instances, piRNAletter, k, delta):
        kmerdict = self.GetKmerDict(piRNAletter, k)
        X = []
        for sequence in instances:
            vector = self.GetSubsequenceProfileVector(sequence, kmerdict, k, delta)
            X.append(vector)
        X = np.array(X)
        return X

    def GetSubsequenceProfileVector(self, sequence, kmerdict, k, delta):
        vector = np.zeros((1, len(kmerdict)))
        sequence = np.array(list(sequence))
        n = len(sequence)
        index_lst = list(itertools.combinations(range(n), k))
        for subseq_index in index_lst:
            subseq_index = list(subseq_index)
            subsequence = sequence[subseq_index]
            position = kmerdict.get(''.join(subsequence))
            subseq_length = subseq_index[-1] - subseq_index[0] + 1
            subseq_score = 1 if subseq_length == k else delta ** subseq_length
            vector[0, position] += subseq_score
        return list(vector[0])

    def _Subsequence(self):
        try:
            k = self.__default_para['kmer']
            delta = self.__default_para['delta']
            NN_list = ['A', 'C', 'G', 'T']

            encoding = []
            header = ['SampleName', 'label']
            for kmer in itertools.product(NN_list, repeat=k):
                header.append('Subsequence_' + ''.join(kmer))
            # encoding.append(header)

            instances = [elem[1] for elem in self.fasta_list]

            code = self.GetSubsequenceProfile(instances, NN_list, k, delta)
            info_list = np.array([[item[0], item[2]] for item in self.fasta_list])
            code = np.hstack((info_list, code))
            encoding = np.vstack((header, code))

            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ end of Subsequence """

    def RC(self, kmer):
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        return ''.join([myDict[nc] for nc in kmer[::-1]])

    def generateRCKmer(self, kmerList):
        rckmerList = set()
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        for kmer in kmerList:
            rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
        return sorted(rckmerList)

    def _RCKmer(self):
        try:
            fastas = self.fasta_list
            k = self.__default_para['kmer']
            upto = False
            normalize = True

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'

            if upto == True:
                for tmpK in range(1, k + 1):
                    tmpHeader = []
                    for kmer in itertools.product(NA, repeat=tmpK):
                        tmpHeader.append(''.join(kmer))
                    header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        for j in range(len(kmers)):
                            if kmers[j] in myDict:
                                kmers[j] = myDict[kmers[j]]
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmpHeader = []
                for kmer in itertools.product(NA, repeat=k):
                    tmpHeader.append(''.join(kmer))
                header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer

                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    for j in range(len(kmers)):
                        if kmers[j] in myDict:
                            kmers[j] = myDict[kmers[j]]
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            encoding = np.array(encoding)
            encoding = np.delete(encoding, 1, axis=1)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
        
    def _NAC(self):
        try:
            NA = 'ACGT'
            encoding = []
            header = ['SampleName']
            for i in NA:
                header.append('NAC_{0}'.format(i))
            encoding.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name]
                for na in NA:
                    code.append(count[na])
                encoding.append(code)
            encoding = np.array(encoding)
            self.encodings = pd.DataFrame(encoding[1:, 1:].astype(float), columns=encoding[0, 1:], index=encoding[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ANF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ANF descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) + 1):
                header.append('ANF_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence)):
                    code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _NCP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'NCP descriptor need fasta sequence with equal length.'
                return False
            chemical_property = {
                'A': [1, 1, 1],
                'C': [0, 1, 0],
                'G': [1, 0, 0],
                'T': [0, 0, 1],
                'U': [0, 0, 1],
                '-': [0, 0, 0],
            }
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 3 + 1):
                header.append('NCP_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code = code + chemical_property.get(aa, [0, 0, 0])
                encodings.append(code)
            encodings = np.array(encodings)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ENAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ENAC descriptor need fasta sequence with equal length.'
                return False
            window = self.__default_para['sliding_window']
            if self.minimum_length < window:
                self.error_msg = 'ENAC descriptor, all the sequence length should be larger than the sliding window'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for aa in AA:
                    header.append('ENAC_sw.' + str(w) + '.' + aa)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence)):
                    if j < len(sequence) and j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        for key in count:
                            count[key] = count[key] / len(sequence[j:j + window])
                        for aa in AA:
                            code.append(count[aa])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _binary(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(1, len(self.fasta_list[0][1]) * 4 + 1):
                header.append('Binary_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _CKSNAP(self):
        try:
            gap = self.__default_para['kspace']
            if self.minimum_length_without_minus < gap + 2:
                self.error_msg = 'CKSNAP - all the sequence length should be larger than the (gap value) + 2 = %s.' % (
                        gap + 2)
                return False

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append('CKSNAP_' + aa + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _EIIP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EIIP descriptor need fasta sequence with equal length.'
                return False
            fastas = self.fasta_list
            AA = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
                '-': 0,
            }
            encodings = []
            header = ['SampleName']
            for i in range(1, len(fastas[0][1]) + 1):
                header.append('EIIP_' + str(i))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for aa in sequence:
                    code.append(EIIP_dict.get(aa, 0))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TriNcleotideComposition(self, sequence, base):
        trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
        tnc_dict = {}
        for triN in trincleotides:
            tnc_dict[triN] = 0
        for i in range(len(sequence) - 2):
            tnc_dict[sequence[i:i + 3]] += 1
        for key in tnc_dict:
            tnc_dict[key] /= (len(sequence) - 2)
        return tnc_dict

    def _PseEIIP(self):
        try:
            fastas = self.fasta_list
            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.'
                    return False
            base = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
            }
            trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
            EIIPxyz = {}
            for triN in trincleotides:
                EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

            encodings = []
            header = ['SampleName'] + ['PseEIIP_{0}'.format(i) for i in trincleotides]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                trincleotide_frequency = self.TriNcleotideComposition(sequence, base)
                code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _ASDC(self):
        try:
            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName']
            header += ['ASDC_' + aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DBE(self):
        try:
            if not self.is_equal:
                self.error_msg = 'DBE descriptor need fasta sequence with equal length.'
                return False

            AA_dict = {
                'AA': [0, 0, 0, 0],
                'AC': [0, 0, 0, 1],
                'AG': [0, 0, 1, 0],
                'AT': [0, 0, 1, 1],
                'CA': [0, 1, 0, 0],
                'CC': [0, 1, 0, 1],
                'CG': [0, 1, 1, 0],
                'CT': [0, 1, 1, 1],
                'GA': [1, 0, 0, 0],
                'GC': [1, 0, 0, 1],
                'GG': [1, 0, 1, 0],
                'GT': [1, 0, 1, 1],
                'TA': [1, 1, 0, 0],
                'TC': [1, 1, 0, 1],
                'TG': [1, 1, 1, 0],
                'TT': [1, 1, 1, 1],
            }

            encodings = []
            header = ['SampleName']
            for i in range((len(self.fasta_list[0][1]) - 1) * 4):
                header.append('DBE_%s' %(i + 1))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 1):
                    if sequence[j] + sequence[j + 1] in AA_dict:
                        code += AA_dict[sequence[j] + sequence[j + 1]]
                    else:
                        code += [0.5, 0.5, 0.5, 0.5]
                encodings.append(code)

            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _LPDF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'LPDF descriptor need fasta sequence with equal length.'
                return False

            encodings = []
            header = ['SampleName']
            header += ['LPDF_%s' % i for i in range(1, len(self.fasta_list[0][1]))]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                dinucleotide_dict = {
                    'AA': 0,
                    'AC': 0,
                    'AG': 0,
                    'AT': 0,
                    'A-': 0,
                    'CA': 0,
                    'CC': 0,
                    'CG': 0,
                    'CT': 0,
                    'C-': 0,
                    'GA': 0,
                    'GC': 0,
                    'GG': 0,
                    'GT': 0,
                    'G-': 0,
                    'TA': 0,
                    'TC': 0,
                    'TG': 0,
                    'TT': 0,
                    'T-': 0,
                    '-A': 0,
                    '-C': 0,
                    '-G': 0,
                    '-T': 0,
                    '--': 0,
                }
                for j in range(1, len(sequence)):
                    dinucleotide_dict[sequence[j] + sequence[j - 1]] += 1
                    code.append(dinucleotide_dict[sequence[j] + sequence[j - 1]] / j)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DPCP(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                print(self.__default_para)
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')

            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 + '_' + p_name for p_name in property_name for n1 in base for n2 in base]
            header = ['SampleName'] + ['DPCP_{0}'.format(i) for i in dinucleotides]
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _DPCP_type2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'DPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')

            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(len(self.fasta_list[0][1]) - 1):
                for p_name in property_name:
                    header.append('DPCP2_' + p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for p_name in property_name:
                    for j in range(len(sequence) - 1):
                        if sequence[j: j + 2] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 2]]])
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _TPCP(self):
        try:
            file_name = 'tridnaPhyche.data'
            property_name = self.__default_para['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
                property_name = property_dict.keys()
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 + '_' + p_name for p_name in property_name for aa1 in AA for aa2 in AA for aa3 in
                        AA]
            header = ['SampleName'] + ['TPCP_{0}'.format(i) for i in triPeptides]
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _TPCP_type2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'TPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'tridnaPhyche.data'
                property_name = self.__default_para['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName']
            for i in range(len(self.fasta_list[0][1]) - 2):
                for p_name in property_name:
                    header.append('TPCP2_' + p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for p_name in property_name:
                    for j in range(len(sequence) - 2):
                        if sequence[j: j + 3] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 3]]])
                        else:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _MMI(self):
        try:
            NA = 'ACGT'
            dinucleotide_list = [a1 + a2 for a1 in NA for a2 in NA]
            trinucleotide_list = [a1 + a2 + a3 for a1 in NA for a2 in NA for a3 in NA]
            dinucleotide_dict = {}
            trinucleotide_dict = {}
            for elem in dinucleotide_list:
                dinucleotide_dict[''.join(sorted(elem))] = 0
            for elem in trinucleotide_list:
                trinucleotide_dict[''.join(sorted(elem))] = 0

            encodings = []
            header = ['SampleName']
            header += ['MMI_%s' % elem for elem in sorted(dinucleotide_dict.keys())]
            header += ['MMI_%s' % elem for elem in sorted(trinucleotide_dict.keys())]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                f1_dict = {
                    'A': 0,
                    'C': 0,
                    'G': 0,
                    'T': 0,
                }
                f2_dict = dinucleotide_dict.copy()
                f3_dict = trinucleotide_dict.copy()

                for elem in sequence:
                    if elem in f1_dict:
                        f1_dict[elem] += 1
                for key in f1_dict:
                    f1_dict[key] /= len(sequence)

                for i in range(len(sequence) - 1):
                    if ''.join(sorted(sequence[i: i + 2])) in f2_dict:
                        f2_dict[''.join(sorted(sequence[i: i + 2]))] += 1
                for key in f2_dict:
                    f2_dict[key] /= (len(sequence) - 1)

                for i in range(len(sequence) - 2):
                    if ''.join(sorted(sequence[i: i + 3])) in f3_dict:
                        f3_dict[''.join(sorted(sequence[i: i + 3]))] += 1
                for key in f3_dict:
                    f3_dict[key] /= (len(sequence) - 2)

                for key in sorted(f2_dict.keys()):
                    if f2_dict[key] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        code.append(f2_dict[key] * math.log(f2_dict[key] / (f1_dict[key[0]] * f1_dict[key[1]])))
                    else:
                        code.append(0)
                for key in sorted(f3_dict.keys()):
                    element_1 = 0
                    element_2 = 0
                    element_3 = 0
                    if f2_dict[key[0:2]] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        element_1 = f2_dict[key[0:2]] * math.log(f2_dict[key[0:2]] / (f1_dict[key[0]] * f1_dict[key[1]]))
                    if f2_dict[key[0] + key[2]] != 0 and f1_dict[key[2]] != 0:
                        element_2 = (f2_dict[key[0] + key[2]] / f1_dict[key[2]]) * math.log(
                            f2_dict[key[0] + key[2]] / f1_dict[key[2]])
                    if f2_dict[key[1:3]] != 0 and f3_dict[key] / f2_dict[key[1:3]] != 0:
                        element_3 = (f3_dict[key] / f2_dict[key[1:3]]) * math.log(f3_dict[key] / f2_dict[key[1:3]])
                    code.append(element_1 + element_2 - element_3)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS2 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 for a1 in AA for a2 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 16
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 1) * 16 + 1):
                header.append('PS2_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 1):
                    code += AA_dict.get(sequence[j: j+2], [0]*16)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS3 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 for a1 in AA for a2 in AA for a3 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 64
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 2) * 64 + 1):
                header.append('PS3_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 2):
                    code += AA_dict.get(sequence[j: j+3], [0]*64)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PS4(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS4 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 + a4 for a1 in AA for a2 in AA for a3 in AA for a4 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 256
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName']
            for i in range(1, (len(self.fasta_list[0][1]) - 3) * 256 + 1):
                header.append('PS4_' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name]
                for j in range(len(sequence) - 3):
                    code += AA_dict.get(sequence[j: j+4], [0]*256)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_9bit(self):
        try:
            encodings = []
            header = ['SampleName']
            for pos in range(1, 4):
                for elem in ['x', 'y', 'z']:
                    header.append('Zcurve9_%s.%s' %(pos, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence)):
                    if (i+1) % 3 == 1:
                        if sequence[i] in pos1_dict:
                            pos1_dict[sequence[i]] += 1
                        else:
                            pos1_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i] in pos2_dict:
                            pos2_dict[sequence[i]] += 1
                        else:
                            pos2_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i] in pos3_dict:
                            pos3_dict[sequence[i]] += 1
                        else:
                            pos3_dict[sequence[i]] = 1

                code += [
                    (pos1_dict.get('A', 0) + pos1_dict.get('G', 0) - pos1_dict.get('C', 0) - pos1_dict.get('T', 0)) / len(sequence), # x
                    (pos1_dict.get('A', 0) + pos1_dict.get('C', 0) - pos1_dict.get('G', 0) - pos1_dict.get('T', 0)) / len(sequence), # y
                    (pos1_dict.get('A', 0) + pos1_dict.get('T', 0) - pos1_dict.get('G', 0) - pos1_dict.get('C', 0)) / len(sequence)  # z
                    ]
                code += [
                    (pos2_dict.get('A', 0) + pos2_dict.get('G', 0) - pos2_dict.get('C', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('C', 0) - pos2_dict.get('G', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('T', 0) - pos2_dict.get('G', 0) - pos2_dict.get('C', 0)) / len(sequence)
                    ]
                code += [
                    (pos3_dict.get('A', 0) + pos3_dict.get('G', 0) - pos3_dict.get('C', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('C', 0) - pos3_dict.get('G', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('T', 0) - pos3_dict.get('G', 0) - pos3_dict.get('C', 0)) / len(sequence)
                    ]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_12bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']
            for base in NN:
                for elem in ['x', 'y', 'z']:
                    header.append('Zcurve12_%s.%s' %(base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos_dict = {}
                for i in range(len(sequence) - 1):
                    if sequence[i: i+2] in pos_dict:
                        pos_dict[sequence[i: i+2]] += 1
                    else:
                        pos_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sC' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sT' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]

                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_36bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']

            for base in NN:
                for pos in range(1, 4):
                    for elem in ['x', 'y', 'z']:
                        header.append('Zcurve36_%s_%s.%s' %(pos, base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 1):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+2] in pos1_dict:
                            pos1_dict[sequence[i: i+2]] += 1
                        else:
                            pos1_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+2] in pos2_dict:
                            pos2_dict[sequence[i: i+2]] += 1
                        else:
                            pos2_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+2] in pos3_dict:
                            pos3_dict[sequence[i: i+2]] += 1
                        else:
                            pos3_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sT' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]
                    code += [
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sT' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                    code += [
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sT' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                encodings.append(code)

            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_48bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']
            for base in NN:
                for base1 in NN:
                    for elem in ['x', 'y', 'z']:
                        header.append('Zcurve48_%s%s.%s' %(base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos_dict = {}
                for i in range(len(sequence) - 2):
                    if sequence[i: i+3] in pos_dict:
                        pos_dict[sequence[i: i+3]] += 1
                    else:
                        pos_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sT' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Z_curve_144bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName']

            for base in NN:
                for base1 in NN:
                    for pos in range(1, 4):
                        for elem in ['x', 'y', 'z']:
                            header.append('Zcurve144_%s_%s%s.%s' %(pos, base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 2):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+3] in pos1_dict:
                            pos1_dict[sequence[i: i+3]] += 1
                        else:
                            pos1_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+3] in pos2_dict:
                            pos2_dict[sequence[i: i+3]] += 1
                        else:
                            pos2_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+3] in pos3_dict:
                            pos3_dict[sequence[i: i+3]] += 1
                        else:
                            pos3_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sT' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sT' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sT' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _NMBroto(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('NMBroto_'+p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    for d in range(1, nlag + 1):
                        try:
                            if N > nlag:
                                atsd = sum([float(property_dict[p_name][AADict[sequence[j: j+2]]]) * float(property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) for j in range(N-d)]) / (N - d)
                            else:
                                atsd = 0
                        except Exception as e:
                            atsd = 0
                        code.append(atsd)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Moran(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[
                #                 0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                # os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('Moran_' + p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i
            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i+2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Idup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) * (property_dict[p_name][AADict[sequence[j+d: j+d+2]]] - pmean) for j in range(N-d)]) / (N - d)
                            Iddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N-d)]) / N
                            code.append(Idup/Iddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _Geary(self):
        try:
            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.__default_para['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.__default_para['Di-RNA-Phychem'].split(';')
            try:
                # data_file = os.path.split(os.path.realpath(__file__))[
                #                 0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                #     os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                data_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data', file_name)
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.__default_para['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append('Geary_' + p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i + 2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Cdup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) ** 2 for j in range(N - d)]) / (2 * (N - d))
                            Cddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N - d)]) / (N - 1)
                            code.append(Cdup / Cddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generatePropertyPairs(self, myPropertyName):
        pairs = []
        for i in range(len(myPropertyName)):
            for j in range(i + 1, len(myPropertyName)):
                pairs.append([myPropertyName[i], myPropertyName[j]])
                pairs.append([myPropertyName[j], myPropertyName[i]])
        return pairs

    def _make_ac_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            header = ['SampleName']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s_%s.lag%d' % (desc, p, l))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]

                for p in myPropertyName:
                    meanValue = 0
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])                   
                    meanValue = meanValue / (len(sequence) - kmer + 1)
                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):                            
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)                        
                        code.append(acValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _make_cc_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = ['SampleName'] + [desc + '_' + n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]

                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _make_acc_vector(self, myPropertyName, myPropertyValue, kmer, desc):
        try:
            fastas = self.fasta_list
            lag = self.__default_para['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False

            header = ['SampleName']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = header + [desc + '_' + n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                ## Auto covariance
                for p in myPropertyName:
                    meanValue = 0                    
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])                    
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            # acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)                        
                        code.append(acValue)

                ## Cross covariance
                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_kmer_frequency(self, sequence, kmer):
        baseSymbol = 'ACGT'
        myFrequency = {}
        for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
            myFrequency[pep] = 0
        for i in range(len(sequence) - kmer + 1):
            myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
        for key in myFrequency:
            myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
        return myFrequency

    def correlationFunction(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
        return CC / len(myPropertyName)

    def correlationFunction_type2(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
        return CC

    def get_theta_array(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + self.correlationFunction(sequence[i:i + kmer],
                                                         sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                         myPropertyName, myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def get_theta_array_type2(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            for p in myPropertyName:
                theta = 0
                for i in range(len(sequence) - tmpLamada - kmer):
                    theta = theta + self.correlationFunction_type2(sequence[i:i + kmer],
                                                                   sequence[
                                                                   i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                                   myIndex,
                                                                   [p], myPropertyValue)
                thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def _PseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('PseDNC_' + pair)
            for k in range(1, lamadaValue + 1):
                header.append('PseDNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('PCPseDNC_' + pair)
            for k in range(1, lamadaValue + 1):
                header.append('PCPseDNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myTriIndex

            header = ['SampleName']
            for tripeptide in sorted(myIndex):
                header.append('PCPseTNC_' + tripeptide)
            for k in range(1, lamadaValue + 1):
                header.append('PCPseTNC_lamada_' + str(k))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _SCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            for pair in sorted(myIndex):
                header.append('SCPseDNC_'+pair)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('SCPseDNC_lamada_' + str(k))

            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _SCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            encodings = []
            myIndex = self.myTriIndex
            header = ['SampleName']
            for pep in sorted(myIndex):
                header.append('SCPseTNC_' + pep)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('SCPseTNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)            
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def _PseKNC(self, myPropertyName, myPropertyValue):
        try:
            baseSymbol = 'ACGT'
            fastas = self.fasta_list
            lamadaValue = self.__default_para['lambdaValue']
            weight = self.__default_para['weight']
            kmer = self.__default_para['kmer']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName']
            header = header + sorted(['PseKNC_'+''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])
            for k in range(1, lamadaValue + 1):
                header.append('PseKNC_lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name]
                kmerFreauency = self.get_kmer_frequency(sequence, kmer)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
                    code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
                    code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            encodings = np.array(encodings)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')

class iStructure(object):
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> structure = iFeatureOmegaCLI.iStructure("./data_examples/1iir.pdb")

    # display available feature descriptor methods
    >>> structure.display_feature_types()

    # import parameters for feature descriptors (optimal)
    >>> structure.import_parameters('parameters/Structure_parameters_setting.json')

    # calculate feature descriptors. Take "Kmer" as an example.
    >>> structure.get_descriptor("AAC_type1")

    # display the feature descriptors
    >>> print(structure.encodings)

    # save feature descriptors
    >>> structure.to_csv("structure_AAC.csv", "index=False", header=False)
    """

    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.protein = None
        self.structure = None
        self.model = None
        self.encodings = None
        self.error_msg = None
        self.pdb_id = None
        self.AA_3to1 = {
            'GLY': 'G',
            'ALA': 'A',
            'LEU': 'L',
            'ILE': 'I',
            'VAL': 'V',
            'PRO': 'P',
            'PHE': 'F',
            'MET': 'M',
            'TRP': 'W',
            'SER': 'S',
            'GLN': 'Q',
            'THR': 'T',
            'CYS': 'C',
            'ASN': 'N',
            'TYR': 'Y',
            'ASP': 'D',
            'GLU': 'E',
            'LYS': 'K',
            'ARG': 'R',
            'HIS': 'H',
        }
        self.AA_1to3 = dict(zip(self.AA_3to1.values(), self.AA_3to1.keys()))
        self.AA_group = {
            'G': 'aliphatic',
            'A': 'aliphatic',
            'V': 'aliphatic',
            'L': 'aliphatic',
            'M': 'aliphatic',
            'I': 'aliphatic',
            'F': 'aromatic',
            'Y': 'aromatic',
            'W': 'aromatic',
            'K': 'positive charged',
            'R': 'positive charged',
            'H': 'positive charged',
            'D': 'negative charged',
            'E': 'negative charged',
            'S': 'uncharged',
            'T': 'uncharged',
            'C': 'uncharged',
            'P': 'uncharged',
            'N': 'uncharged',
            'Q': 'uncharged',
        }
        self.AA_HEC = {
            'H': 'H',
            'B': 'E',
            'E': 'E',
            'G': 'H',
            'I': 'H',
            'T': 'C',
            'S': 'C',
            '-': 'C'
        }
        self.__default_para_dict = {
            'AAC_type1': {'shell': (3, 20, 2)},
            'AAC_type2': {'shell': (3, 20, 2)},
            'GAAC_type1': {'shell': (3, 20, 2)},
            'GAAC_type2': {'shell': (3, 20, 2)},
            'SS3_type1': {'shell': (3, 20, 2)},
            'SS3_type2': {'shell': (3, 20, 2)},
            'SS8_type1': {'shell': (3, 20, 2)},
            'SS8_type2': {'shell': (3, 20, 2)},
            'AC_type1': {'shell': (1, 10, 1)},
            'AC_type2': {'shell': (1, 10, 1)},
            'Network-based index': {'distance': 11},
        }        
        self.__cmd_dict = {
            'AAC_type1': 'self.get_residue_descriptor("AAC_type1")',
            'AAC_type2': 'self.get_residue_descriptor("AAC_type2")',
            'GAAC_type1': 'self.get_residue_descriptor("GAAC_type1")',
            'GAAC_type2': 'self.get_residue_descriptor("GAAC_type2")',
            'SS3_type1': 'self.get_residue_descriptor("SS3_type1")',
            'SS3_type2': 'self.get_residue_descriptor("SS3_type2")',
            'SS8_type1': 'self.get_residue_descriptor("SS8_type1")',
            'SS8_type2': 'self.get_residue_descriptor("SS8_type2")',
            'HSE_CA': 'self.get_HSE_CA()',
            'HSE_CB': 'self.get_HSE_CB()',
            'Residue depth': 'self.get_residue_depth()',
            'AC_type1': 'self.get_atom_descriptor("AC_type1")',
            'AC_type2': 'self.get_atom_descriptor("AC_type2")',
            'Network-based index': 'self.get_network_descriptor()',            
        }
        self.read_pdb()

    def read_pdb(self):
        if self.pdb_file.endswith('.pdb') or self.pdb_file.endswith('.PDB'):
            self.protein = PDBParser(PERMISSIVE=1)
            self.structure = self.protein.get_structure(self.pdb_file[0:4], self.pdb_file)
        elif self.pdb_file.endswith('.cif') or self.pdb_file.endswith('.CIF'):
            parser = MMCIFParser()
            self.structure = parser.get_structure(self.pdb_file[0:4], self.pdb_file)
        try:
            self.model = self.structure[0]           
            return True
        except KeyError as e:
            self.error_msg = str(e)
            return False

    def get_pdb_id(self):
        try:
            if self.pdb_file.lower().endswith('.pdb'):
                with open(self.pdb_file) as f:
                    record = f.read().strip().split('\n')[0]
                tmp = re.split('\s+', record)               
                self.pdb_id = tmp[3].lower()
            if self.pdb_file.lower().endswith('.cif'):
                with open(self.pdb_file) as f:
                    self.pdb_id = f.read().strip().split('\n')[0].split('_')[1].lower()
            return True, self.pdb_id
        except Exception as e:
            self.error_msg = str(e)
            return False, None

    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        AAC_type1                      Amino acids content type 1
        AAC_type2                      Amino acids content type 2
        GAAC_type1                     Grouped amino acids content type 1
        GAAC_type2                     Grouped amino acids content type 2
        SS3_type1                      Secondary structure elements (3) type 1
        SS3_type2                      Secondary structure elements (3) type 2
        SS8_type1                      Secondary structure elements (8) type 1
        SS8_type2                      Secondary structure elements (8) type 2
        HSE_CA                         Half sphere exposure α
        HSE_CB                         Half sphere exposure β
        Residue depth                  Residue depth
        AC_type1                       Atom content type 1
        AC_type2                       Atom content type 2
        Network-based index            Network-based index

        Note: the first column is the names of available feature types while the second column description.  
        
        '''
        
        print(info)
    
    def get_residue_descriptor(self, method='AAC_type1'):   # target_list: 2d list, [[chain, resseq, resname], [chain, resseq, resname]]
        try:
            tmp_residues = list(self.model.get_residues())
            dssp_status = True
            try:
                dssp = DSSP(self.model, self.pdb_file)  # calculate secondary structure
            except Exception as e:
                self.error_msg = 'Secondary structure calculate failed. Please check whether DSSP was installed?'
                dssp_status = 0
                dssp = []
            
            residues = []  # remove hetfield, only residues are saved.
            for residue in tmp_residues:
                if residue.has_id('CA') or residue.has_id('CB'):
                    residue_id = residue.get_id()
                    if residue_id[0] == ' ':
                        residues.append(residue)
            
            target_list = []
            for residue in residues:
                target_list.append([residue.parent.id, residue.id[1], residue.resname])
            
            header = []
            encodings = []
            if len(residues) == len(dssp):
                for item in target_list:
                    target_chain, target_resseq, target_resname = item[0], item[1], item[2]
                    df_residue = None
                    if self.model.has_id(target_chain) and self.model[target_chain].has_id(target_resseq) and self.model[target_chain][target_resseq].get_resname() == target_resname:  # check if the item in target_list exist
                        target_residue = self.model[target_chain][target_resseq]
                        target_residue_name = target_residue.get_resname()
                        target_atom = target_residue['CB'] if target_residue.has_id('CB') else target_residue['CA']

                        record_list = []  # [chain, resseq, resname, distance, property, hec8, hec3]
                        for residue, hec in zip(residues, dssp):
                            source_atom = residue['CB'] if residue.has_id('CB') else residue['CA']
                            record_list.append([residue.parent.get_id(), residue.get_id()[1], self.AA_3to1[residue.get_resname()], target_atom - source_atom, self.AA_group[self.AA_3to1[residue.get_resname()]], hec[2], self.AA_HEC[hec[2]]])

                        df_residue = pd.DataFrame(np.array(record_list), columns=['chain', 'resseq', 'resname', 'distance', 'property', 'hec8', 'hec3'])
                        df_residue['distance'] = df_residue['distance'].astype('float64')                    
                    
                        if method == 'AAC_type1':
                            _, tmp_code = self.AAC_type1(df_residue)
                        elif method == 'AAC_type2':
                            _, tmp_code = self.AAC_type2(df_residue)
                        elif method == 'GAAC_type1':
                            _, tmp_code = self.GAAC_type1(df_residue)
                        elif method == 'GAAC_type2':
                            _, tmp_code = self.GAAC_type2(df_residue)
                        elif method == 'SS8_type1':
                            _, tmp_code = self.SS8_type1(df_residue)
                        elif method == 'SS8_type2':
                            _, tmp_code = self.SS8_type2(df_residue)
                        elif method == 'SS3_type1':
                            _, tmp_code = self.SS3_type1(df_residue)
                        elif method == 'SS3_type2':
                            _, tmp_code = self.SS3_type2(df_residue)
                        else:
                            return False, None
                        encodings.append([target_chain + '_' + target_resname + '_' + str(target_resseq)] + tmp_code)
                        header = ['Sample'] + _
            else:
                self.error_msg = 'Secondary structure calculate failed.'
                for item in target_list:
                    target_chain, target_resseq, target_resname = item[0], item[1], item[2]
                    if self.model.has_id(target_chain) and self.model[target_chain].has_id(target_resseq) and self.model[target_chain][target_resseq].get_resname() == target_resname:  # check if the item in target_list exist
                        target_residue = self.model[target_chain][target_resseq]
                        target_residue_name = target_residue.get_resname()
                        target_atom = target_residue['CB'] if target_residue.has_id('CB') else target_residue['CA']

                        record_list = []  # [chain, resseq, resname, distance, property]
                        for residue in residues:
                            source_atom = residue['CB'] if residue.has_id('CB') else residue['CA']
                            record_list.append([residue.parent.get_id(), residue.get_id()[1], self.AA_3to1[residue.get_resname()], target_atom - source_atom, self.AA_group[self.AA_3to1[residue.get_resname()]]])                        
                            
                        df_residue = pd.DataFrame(np.array(record_list), columns=['chain', 'resseq', 'resname', 'distance', 'property'])
                        df_residue['distance'] = df_residue['distance'].astype('float64')

                        if method == 'AAC_type1':
                            _, tmp_code = self.AAC_type1(df_residue)                        
                        elif method == 'AAC_type2':
                            _, tmp_code = self.AAC_type2(df_residue)
                        elif method == 'GAAC_type1':
                            _, tmp_code = self.GAAC_type1(df_residue)
                        elif method == 'GAAC_type2':
                            _, tmp_code = self.GAAC_type2(df_residue)                    
                        else:
                            return False, None
                        encodings.append([target_chain + '_' + target_resname + '_' + str(target_resseq)] + tmp_code)                        
                        header = ['Sample'] + _        
            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=header[1:], index=np.array(encodings)[:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_atom_descriptor(self, method='AC_type1'):       # target_list: 2d list, [[chain, serial_number, name], [chain, serial_number, name]]
        try:        
            tmp_atoms = self.model.get_atoms()
            atoms = {}                              # use dict
            for atom in tmp_atoms:
                if atom.parent.get_id()[0] != 'W':  # reomove water residue
                    atoms[atom.parent.parent.id + str(atom.serial_number) + re.sub(' ', '', atom.element)] = atom
            
            target_list = []
            for atom in atoms:
                if atoms[atom].name == 'CA':
                    target_list.append([atoms[atom].parent.parent.id, atoms[atom].serial_number, re.sub(' ', '', atoms[atom].element)])
            
            header = []
            encodings = []
            for item in target_list:
                target_chain, target_serial_number, target_element = item[0], item[1], item[2]
                if target_chain + str(target_serial_number) + target_element in atoms:
                    target = atoms[target_chain + str(target_serial_number) + target_element]                
                    record_list = []  # [chain, serial_number, element, distance]
                    for key in atoms:
                        atom = atoms[key]
                        record_list.append([atom.parent.parent.id, atom.serial_number, re.sub(' ', '', atom.element), target - atom])
                    df_atom = pd.DataFrame(np.array(record_list), columns=['chain', 'serial_number', 'element', 'distance'])
                    df_atom['distance'] = df_atom['distance'].astype('float64')

                    if method == 'AC_type1':
                        _, tmp_code = self.AC_type1(df_atom)                    
                    elif method == 'AC_type2':
                        _, tmp_code = self.AC_type2(df_atom)
                    else:
                        return False, None
                    encodings.append([target_chain + '_' + re.sub(' ', '', target_element) + '_' + str(target_serial_number)] + tmp_code)                        
                    header = ['Sample'] + _                

            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=header[1:], index=np.array(encodings)[:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_residue_depth(self):
        try:
            msms_status = True
            try:
                rd = ResidueDepth(self.model)
            except Exception as e:
                self.error_msg = 'Residue depth calculate failed. Please check whether msms was installed?'
                msms_status = 0
                return False, None

            key_array = rd.keys()
            
            encodings = []
            for key in key_array:
                residue_depth, ca_depth = rd[key]            
                encodings.append([key[0]+'_'+str(key[1][1]), residue_depth, ca_depth])

            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=['Residue_depth', 'CA_depth'], index=np.array(encodings)[:, 0])        
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
       
    def get_HSE_CA(self):
        try:        
            hse = HSExposureCA(self.model)
            encodings = []
            for e in hse:
                encodings.append([e[0].parent.id+'_'+e[0].resname+'_'+str(e[0].id[1]), e[1][0], e[1][1], e[1][2]])
            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=['HSE_CA_value1', 'HSE_CA_value2', 'HSE_CA_value3'], index=np.array(encodings)[:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
    
    def get_HSE_CB(self):
        try:
            hse = HSExposureCB(self.model)
            encodings = []
            for e in hse:
                encodings.append([e[0].parent.id+'_'+e[0].resname+'_'+str(e[0].id[1]), e[1][0], e[1][1], e[1][2]])
            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=['HSE_CB_value1', 'HSE_CB_value2', 'HSE_CB_value3'], index=np.array(encodings)[:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def AAC_type1(self, df):
        shell = self.__default_para_dict['AAC_type1']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]            
            
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0                
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i,2] in AA_dict:
                    AA_dict[df_tmp.iloc[i,2]] += 1
            for key in AA_dict:
                if len(df_tmp) == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= len(df_tmp)
        
            code += AA_dict.values()
            m += 1
        return header, code
    
    def AAC_type2(self, df):
        shell = self.__default_para_dict['AAC_type2']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]            
            
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0                
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i,2] in AA_dict:
                    AA_dict[df_tmp.iloc[i,2]] += 1
            for key in AA_dict:
                if len(df_tmp) == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= len(df_tmp)
        
            code += AA_dict.values()
            m += 1
        return header, code

    def GAAC_type1(self, df):
        shell = self.__default_para_dict['GAAC_type1']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'aliphatic': 0,
                'aromatic': 0,
                'positive charged': 0,
                'negative charged': 0,
                'uncharged': 0
            }
            group_list = ['aliphatic', 'aromatic', 'positive charged', 'negative charged', 'uncharged']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 4] in group_dict:
                    group_dict[df_tmp.iloc[i, 4]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def GAAC_type2(self, df):
        shell = self.__default_para_dict['GAAC_type2']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'aliphatic': 0,
                'aromatic': 0,
                'positive charged': 0,
                'negative charged': 0,
                'uncharged': 0
            }
            group_list = ['aliphatic', 'aromatic', 'positive charged', 'negative charged', 'uncharged']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 4] in group_dict:
                    group_dict[df_tmp.iloc[i, 4]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def SS8_type1(self, df):
        shell = self.__default_para_dict['SS8_type1']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 5] in group_dict:
                    group_dict[df_tmp.iloc[i, 5]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS8_type2(self, df):
        shell = self.__default_para_dict['SS8_type2']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 5] in group_dict:
                    group_dict[df_tmp.iloc[i, 5]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS3_type1(self, df):
        shell = self.__default_para_dict['SS3_type1']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 6] in group_dict:
                    group_dict[df_tmp.iloc[i, 6]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS3_type2(self, df):
        shell = self.__default_para_dict['SS3_type2']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 6] in group_dict:
                    group_dict[df_tmp.iloc[i, 6]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def AC_type1(self, df):
        shell = self.__default_para_dict['AC_type1']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            AA = 'CNOS'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0
            
            sum = 0
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 2] in AA_dict:
                    AA_dict[df_tmp.iloc[i, 2]] += 1
                    sum += 1
            for key in AA_dict:
                if sum == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= sum
            
            code += AA_dict.values()
            m += 1           
        return header, code
    
    def AC_type2(self, df):
        shell = self.__default_para_dict['AC_type2']['shell']
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            AA = 'CNOS'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0
            
            sum = 0
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 2] in AA_dict:
                    AA_dict[df_tmp.iloc[i, 2]] += 1
                    sum += 1
            for key in AA_dict:
                if sum == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= sum
            
            code += AA_dict.values()
            m += 1           
        return header, code

    def get_network_descriptor(self):
        try:
            distance_cutoff = self.__default_para_dict['Network-based index']['distance']
            tmp_residues = list(self.model.get_residues())        
            residues = []
            for residue in tmp_residues:   # remove hetfield, only residues are saved.
                residue_id = residue.get_id()
                if residue_id[0] == ' ':
                    residues.append(residue)
            
            target_list = []
            for residue in residues:
                target_list.append([residue.parent.id, residue.id[1], residue.resname])

            G = nx.Graph()     
            # add graph nodes
            for i in range(len(residues)):
                node = residues[i].resname + '_' + residues[i].parent.id + str(residues[i].id[1])
                G.add_node(node)

            CB_coord_dict = {}
            
            # add graph edges
            for i in range(len(residues)):
                node = residues[i].resname + '_' + residues[i].parent.id + str(residues[i].id[1])            
                atom = residues[i]['CB'] if residues[i].has_id('CB') else residues[i]['CA']
                CB_coord_dict[node] = atom.coord            
                for j in range(i+1, len(residues)):
                    node_2 = residues[j].resname + '_' + residues[j].parent.id + str(residues[j].id[1])
                    atom_2 = residues[j]['CB'] if residues[j].has_id('CB') else residues[j]['CA']
                    distance = atom - atom_2
                    if distance <= distance_cutoff:
                        G.add_edge(node, node_2)
            
            net_dict = {}
            net_dict['average_clustering'] = nx.average_clustering(G)        
            net_dict['diameter'] = nx.diameter(G)
            net_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                
            net_dict['degree_centrality'] = nx.degree_centrality(G)                       # degree centrality        
            net_dict['betweenness_centrality'] = nx.betweenness_centrality(G)             # betweenness
            net_dict['clustering'] = nx.clustering(G)                                     # clustering coefficient
            net_dict['closeness_centrality'] = nx.closeness_centrality(G)                 # closeness
            net_dict['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G)       # centrality        

            encodings = []
            for item in target_list:
                id = item[2]+'_'+item[0]+str(item[1])
                tmp_code = [id]
                if id in G.nodes:
                    tmp_code += [G.degree(id), net_dict['degree_centrality'].get(id, 'NA'), net_dict['betweenness_centrality'].get(id, 'NA'), net_dict['clustering'].get(id, 'NA'), net_dict['closeness_centrality'].get(id, 'NA'), net_dict['eigenvector_centrality'].get(id, 'NA')]            
                    encodings.append(tmp_code)
            self.encodings = pd.DataFrame(np.array(encodings)[:, 1:].astype(float), columns=['degree', 'degree_centrality', 'betweenness', 'clustering_coefficient', 'closeness', 'centrality'], index=np.array(encodings)[:, 0])        
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
 
    def save_descriptor(self, data, file_name):
        try:
            if not data is None:
                if file_name.endswith(".tsv"):
                    np.savetxt(file_name, data.values[:, 1:], fmt='%s', delimiter='\t')                    
                    return True
                if file_name.endswith(".tsv_1"):                    
                    data.to_csv(file_name, sep='\t', header=True, index=False)
                    return True
                if file_name.endswith(".csv"):
                    np.savetxt(file_name, data.values[:, 1:], fmt='%s', delimiter=',')
                    return True
                if file_name.endswith(".svm"):
                    with open(file_name, 'w') as f:
                        for line in data.values:
                            f.write('0')
                            for i in range(1, len(line)):
                                f.write('  %d:%s' % (i, line[i]))
                            f.write('\n')
                    return True
                if file_name.endswith(".arff"):
                    with open(file_name, 'w') as f:
                            f.write('@relation descriptor\n\n')
                            for i in data.columns:
                                f.write('@attribute %s numeric\n' % i)
                            f.write('@attribute play {yes, no}\n\n')
                            f.write('@data\n')
                            for line in data.values:
                                for fea in line[1:]:
                                    f.write('%s,' % fea)
                                f.write('no\n')
                    return True
            else:
                return False
        except Exception as e:
            return False
       
    def import_parameters(self, file):
        if os.path.exists(file):
            with open(file) as f:
                records = f.read().strip()
            try:
                self.__default_para_dict = json.loads(records)
                print('File imported successfully.')
            except Exception as e:
                print('Parameter file parser error.')

    def get_descriptor(self, descriptor='AAC_type1'):
        if descriptor in self.__cmd_dict:
            cmd = self.__cmd_dict[descriptor]
            status = eval(cmd)
            # print(status)
        else:
            print('The descriptor type does not exist.')

    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')

class iLigand():
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> ligand = iFeatureOmegaCLI.iLigand("./data_examples/Chemical_SMILES.txt")

    # display available feature descriptor methods
    >>> ligand.display_feature_types()    

    # calculate feature descriptors. Take "Constitution" as an example.
    >>> ligand.get_descriptor("Constitution")

    # display the feature descriptors
    >>> print(ligand.encodings)

    # save feature descriptors
    >>> ligand.to_csv("Ligand_constitution.csv", "index=False", header=False)
    """
    def __init__(self, file):
        self.file = file
        self.encodings = None
        self.__default_para_dict = {
            'Constitution': ['nhyd', 'nhal', 'nhet', 'nhev', 'ncof', 'ncocl', 'ncobr', 'ncoi', 'ncarb', 'nphos', 'nsulph', 'noxy', 'nnitro', 'nring', 'nrot', 'ndonr', 'naccr', 'nsb', 'ndb', 'ntb', 'naro', 'nta', 'AWeight', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'],
            'Topology': ['AW', 'J', 'Thara', 'Tsch', 'Tigdi', 'Platt', 'Xu', 'Pol', 'Dz', 'Ipc', 'BertzCT', 'GMTI', 'ZM1', 'ZM2', 'MZM1', 'MZM2', 'Qindex', 'diametert', 'radiust', 'petitjeant', 'Sito', 'Hato', 'Geto', 'Arto'],
            'Connectivity': ['Chi0', 'Chi1', 'mChi1', 'Chi2', 'Chi3', 'Chi4', 'Chi5', 'Chi6', 'Chi7', 'Chi8', 'Chi9', 'Chi10', 'Chi3c', 'Chi4c', 'Chi4pc', 'Chi3ch', 'Chi4ch', 'Chi5ch', 'Chi6ch', 'Chiv0', 'Chiv1', 'Chiv2', 'Chiv3', 'Chiv4', 'Chiv5', 'Chiv6', 'Chiv7', 'Chiv8', 'Chiv9', 'Chiv10', 'dchi0', 'dchi1', 'dchi2', 'dchi3', 'dchi4', 'Chiv3c', 'Chiv4c', 'Chiv4pc', 'Chiv3ch', 'Chiv4ch', 'Chiv5ch', 'Chiv6ch', 'knotpv', 'knotp'],
            'Kappa': ['kappa1', 'kappa2', 'kappa3', 'kappam1', 'kappam2', 'kappam3', 'phi'],
            'EState': ['value', 'max', 'min', 'Shev', 'Scar', 'Shal', 'Shet', 'Save', 'Smax', 'Smin', 'DS'],
            'Autocorrelation-moran': ['MATSm1', 'MATSm2', 'MATSm3', 'MATSm4', 'MATSm5', 'MATSm6', 'MATSm7', 'MATSm8',
               'MATSv1', 'MATSv2', 'MATSv3', 'MATSv4', 'MATSv5', 'MATSv6', 'MATSv7', 'MATSv8',
               'MATSe1', 'MATSe2', 'MATSe3', 'MATSe4', 'MATSe5', 'MATSe6', 'MATSe7', 'MATSe8',
               'MATSp1', 'MATSp2', 'MATSp3', 'MATSp4', 'MATSp5', 'MATSp6', 'MATSp7', 'MATSp8',],
            'Autocorrelation-geary': ['GATSm1', 'GATSm2', 'GATSm3', 'GATSm4', 'GATSm5', 'GATSm6', 'GATSm7', 'GATSm8',
               'GATSv1', 'GATSv2', 'GATSv3', 'GATSv4', 'GATSv5', 'GATSv6', 'GATSv7', 'GATSv8',
               'GATSe1', 'GATSe2', 'GATSe3', 'GATSe4', 'GATSe5', 'GATSe6', 'GATSe7', 'GATSe8',
               'GATSp1', 'GATSp2', 'GATSp3', 'GATSp4', 'GATSp5', 'GATSp6', 'GATSp7', 'GATSp8'],
            'Autocorrelation-broto': ['ATSm1', 'ATSm2', 'ATSm3', 'ATSm4', 'ATSm5', 'ATSm6', 'ATSm7', 'ATSm8',
               'ATSv1', 'ATSv2', 'ATSv3', 'ATSv4', 'ATSv5', 'ATSv6', 'ATSv7', 'ATSv8',
               'ATSe1', 'ATSe2', 'ATSe3', 'ATSe4', 'ATSe5', 'ATSe6', 'ATSe7', 'ATSe8',
               'ATSp1', 'ATSp2', 'ATSp3', 'ATSp4', 'ATSp5', 'ATSp6', 'ATSp7', 'ATSp8'],
            'Molecular properties': ['LogP', 'MR', 'LabuteASA', 'TPSA', 'Hy', 'UI'],            
            'Charge': ['SPP', 'LDI', 'Rnc', 'Rpc', 'Mac', 'Tac', 'Mnc', 'Tnc', 'Mpc', 'Tpc', 'Qass', 'QOss', 'QNss', 'QCss', 'QHss', 'Qmin', 'QOmin', 'QNmin', 'QCmin', 'QHmin', 'Qmax', 'QOmax', 'QNmax', 'QCmax', 'QHmax'],
            'Moe-Type descriptors': ['LabuteASA', 'TPSA', 'slogPVSA', 'MRVSA', 'PEOEVSA', 'EstateVSA', 'VSAEstate'],
            'Daylight-type fingerprints': ['topological'],
            'MACCS fingerprints': ['MACCS'],
            'Atom pairs fingerprints': ['atompairs'],
            'Morgan fingerprints': ['morgan'],
            'TopologicalTorsion fingerprints': ['torsions'],
            'E-state fingerprints': ['Estate'],
            'Basak': ["CIC0", "CIC1", "CIC2", "CIC3", "CIC4", "CIC5", "CIC6", "SIC0", "SIC1", "SIC2", "SIC3", "SIC4", "SIC5", "SIC6", "IC0", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6"],
            'Burden': ['bcutp', 'bcute', 'bcutv', 'bcutm'],
            'Pharmacophore': ['CalcCATS'],
            'Morgan-ECFP4 fingerprints': ['ECFP4'],
            'Morgan-ECFP6 fingerprints': ['ECFP6'],
            'Morgan-FCFP4 fingerprints': ['FCFP4'],
            'Morgan-FCFP6 fingerprints': ['FCFP6'],
        }
        self.error_msg = None
        self.fps = None

        with open(self.file) as f:
            self.smiles_list = f.read().strip().split('\n')

        self.mol_list = []
        no_error = True
        for mol in self.smiles_list:
            molObj = Chem.MolFromSmiles(mol)
            if molObj is not None:
                self.mol_list.append(molObj)
            else:
                no_error = False

    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        Basak
        Burden
        Pharmacophore
        Constitution
        Topology
        Connectivity
        Kappa
        EState
        Autocorrelation-moran
        Autocorrelation-geary
        Autocorrelation-broto
        Molecular properties
        Charge
        Moe-Type descriptors
        MACCS fingerprints
        Morgan-ECFP4 fingerprint
        Morgan-ECFP6 fingerprint
        Morgan-FCFP4 fingerprint
        Morgan-FCFP6 fingerprint
        E-state fingerprints

        '''
        
        print(info)
 
    def get_descriptor(self, descriptor, **kwargs):
        if descriptor in self.__default_para_dict:
            self.fps = self.__default_para_dict[descriptor]
            df = pd.DataFrame()
            for i, mol in enumerate(self.mol_list):
                for fp in self.fps:
                    coder = eval(fp)
                    code = coder(mol, **kwargs)
                    if type(code) == list or type(code) == np.ndarray:
                        for j, c in enumerate(code):
                            df.loc[i, fp+str(j)] = c
                    else:
                        df.loc[i, fp] = code
            self.encodings = pd.DataFrame(df.values.astype(float), columns=df.columns, index=self.smiles_list)
        else:
            self.error_msg = 'Descriptor does not exist.'
            return False

    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')

class iAnalysis():
    """
    # Running examples:
    
    >>> import iFeatureOmegaCLI
    >>> import pandas as pd
    
    # read data to pandas DataFrame
    >>> df = pd.read_csv('data_examples/data_without_labels.csv', sep=',', header=None, index_col=None)    

    # Initialize a instance of iAnalysis
    >>> data = iFeatureOmegaCLI.iAnalysis(df)
    
    # kmeans clustering
    >>> data.kmeans(nclusters=3)
    >>> data.cluster_result

    # other clustering algorithm
    >>> data.MiniBatchKMeans(nclusters=2)   # Mini-Batch K-means
    >>> data.GM()                           # Gaussian mixture
    >>> data.Agglomerative()                # Agglomerative
    >>> data.Spectral()                     # Spectral
    >>> data.MCL()                          # Markov clustering
    >>> data.hcluster()                     # Hierarchical clustering
    >>> data.APC()                          # Affinity propagation clustering 
    >>> data.meanshift()                    # Mean shift
    >>> data.DBSCAN()                       # DBSCAN

    # save cluster result
    >>> data.cluster_to_csv(file='cluster_result.csv')

    # dimensionality reduction
    >>> data.t_sne(n_components=2)          # t-distributed stochastic neighbor embedding
    >>> data.PCA(n_components=2)            # Principal component analysis
    >>> data.LDA(n_components=2)            # Latent dirichlet allocation

    # save dimensionality reduction result
    >>> data.dimension_to_csv(file="dimension_reduction_result.csv")

    # feature normalization
    >>> data.ZScore()                       # ZScore
    >>> data.MinMax()                       # MinMax

    # save feature normalization result
    >>> data.normalization_to_csv(file="feature_normalization_result.csv")

    """
    def __init__(self, df):
        self.dataframe = df
        self.datalabel = np.zeros(len(df))
        self.cluster_result = None
        self.cluster_plots = None
        self.dimension_reduction_result = None
        self.feature_normalization_data = None
        self.error_msg = None

    # cluster methods
    def kmeans(self, nclusters=2):
        try:
            if not self.dataframe is None:
                cluster_res = KMeans(n_clusters=nclusters).fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MiniBatchKMeans(self, nclusters=2):
        try:
            if not self.dataframe is None:
                cluster_res = MiniBatchKMeans(n_clusters=nclusters).fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def GM(self, nclusters=2):
        try:
            if not self.dataframe is None:
                cluster_res = GaussianMixture(n_components=nclusters).fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Agglomerative(self, nclusters=2):
        try:
            if not self.dataframe is None:
                cluster_res = AgglomerativeClustering(n_clusters=nclusters).fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Spectral(self, nclusters=2):
        try:
            if not self.dataframe is None:
                cluster_res = SpectralClustering(n_clusters=nclusters).fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MCL(self, expand=2.0, inflate=2.0, multiply=2.0, max_loop=1000):
        try:
            if not self.dataframe is None:
                cluster_res = MarkvCluster(self.dataframe.values, int(expand), float(inflate), float(multiply), max_loop).cluster_array               
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True               
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def hcluster(self):
        try:
            if not self.dataframe is None:
                disMat = sch.distance.pdist(self.dataframe.values, 'euclidean')
                Z = sch.linkage(disMat, method='average')
                cluster_res = sch.fcluster(Z, 1, 'inconsistent')
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def APC(self):
        try:
            if not self.dataframe is None:
                cluster_res = AffinityPropagation().fit_predict(self.dataframe.values)
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def meanshift(self):
        try:
            if not self.dataframe is None:
                bandwidth = estimate_bandwidth(self.dataframe)
                try:
                    cluster_res = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(self.dataframe.values)                    
                except Exception as e:
                    cluster_res = np.zeros(len(self.dataframe))                    
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DBSCAN(self):
        try:
            if not self.dataframe is None:
                data = StandardScaler().fit_transform(self.dataframe.values)
                cluster_res = DBSCAN().fit_predict(data)            
                self.cluster_result = pd.DataFrame(cluster_res, index=self.dataframe.index, columns=['cluster'])
                # rd_data, ok = self.t_sne(2)
                # self.cluster_plots = self.generate_plot_data(cluster_res, rd_data)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    # dimensionality reduction
    def t_sne(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = 'The reduced dimension number is out of range.'
                    return None, False            
                self.dimension_reduction_result = TSNE(n_components=n_components, method='exact', learning_rate=100).fit_transform(self.dataframe.values)            
                
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False
    
    def PCA(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = 'The reduced dimension number is out of range.'
                    return None, False
                self.dimension_reduction_result = PCA(n_components=n_components).fit_transform(self.dataframe.values)            
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LDA(self, n_components=2):
        try:
            if not self.dataframe is None:
                if n_components >= self.dataframe.shape[1]:
                    self.error_msg = 'The reduced dimension number is out of range.'
                    return None, False
                lda = LatentDirichletAllocation(n_components=n_components).fit(self.dataframe.values, self.datalabel)
                self.dimension_reduction_result = lda.transform(self.dataframe.values)
                return True
            else:
                self.error_msg = 'Data is null.'
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generate_plot_data(self, label, rd_data):
            plot_data = []
            clusters = sorted(set(label))
            for c in clusters:
                plot_data.append([c, rd_data[np.where(label == c)]])            
            return plot_data

    def ClusteringScatterPlot(self, method, file):
        data = self.cluster_plots        
        plt.style.use('default')
        try:
            fontdict = {
                'family': 'Arial',
                'size': 14,
                'color': '#282828',
            }
            colorlist = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF',
                            '#999999']
            marklist = ['o'] * 9 + ['v'] * 9 + ['^'] * 9 + ['+'] * 9
            prefix = 'Cluster:' if method == 'Clustering' else 'Sample category:'
            fig = plt.figure(0, facecolor='w')
            plt.grid(False)
            plt.title(method, fontdict=fontdict)
            plt.xlabel("PC 1", fontdict=fontdict)
            plt.ylabel("PC 2", fontdict=fontdict)
            plt.xticks(fontproperties='Arial', size=12)
            plt.yticks(fontproperties='Arial', size=12)
            for i, item in enumerate(data):
                plt.scatter(item[1][:, 0], item[1][:, 1], color=colorlist[i % len(colorlist)], s=70,
                                marker=marklist[i % len(marklist)], label='%s %s' % (prefix, item[0]),
                                edgecolor="w")
            ax = plt.gca()
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['right'].set_color('white')
            ax.spines['top'].set_color('white')
            plt.legend()
            plt.tight_layout()
            plt.savefig(file)
            plt.close(0)
            return True
        except Exception as e:
            return False

    def cluster_to_csv(self, file='cluster_result.csv'):
        if not self.cluster_result is None:
            self.cluster_result.to_csv(file, index=True, header=True)

    def dimension_to_csv(self, file="dimension_reduction_result.csv"):
        if not self.dimension_reduction_result is None:
            np.savetxt(file, self.dimension_reduction_result, fmt='%f', delimiter=',')

    def normalization_to_csv(self, file="feature_normalization_result.csv", header=False, index=False):
        if not self.feature_normalization_data is None:
            self.feature_normalization_data.to_csv(file, sep=',', header=header, index=index)

    # feature normalization
    def ZScore(self):
        try:
            data = self.dataframe.values
            std_array = np.std(data, axis=0)
            mean_array = np.mean(data, axis=0)            
            for i in range(len(mean_array)):
                if std_array[i] != 0:
                    data[:, i] = (data[:, i] - mean_array[i]) / std_array[i]
                else:
                    data[:, i] = 0            
            self.feature_normalization_data = pd.DataFrame(data, columns=self.dataframe.columns)
            # self.feature_normalization_data.insert(0, 'Labels', self.datalabel)
            
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MinMax(self):
        try:
            data = self.dataframe.values        
            for i in range(len(data[0])):
                maxValue, minValue = np.max(data[:, i]), np.min(data[:, i])
                data[:, i] = (data[:, i] - minValue) / (maxValue - minValue)
            # replace NaN value with mean
            data = self.fill_ndarray(data.T).T
            self.feature_normalization_data = pd.DataFrame(data, columns=self.dataframe.columns)
            # self.feature_normalization_data.insert(0, 'Labels', self.datalabel)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def fill_ndarray(self, t1):
        for i in range(t1.shape[1]):
            temp_col = t1[:, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]                
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        return t1

class MarkvCluster(object):
    def __init__(self, data, expand_factor=2, inflate_factor=2.0, mult_factor=2.0, max_loop=200):
        super(MarkvCluster, self).__init__()
        M = np.corrcoef(data)
        M[M < 0] = 0
        for i in range(len(M)):
            M[i][i] = 0
        import networkx as nx
        G = nx.from_numpy_matrix(M)
        self.M, self.clusters = self.networkx_mcl(G, expand_factor=expand_factor, inflate_factor=inflate_factor,
                                                  max_loop=max_loop, mult_factor=mult_factor)
        self.cluster_array = self.get_array()

    def get_array(self):
        array = []
        for key in self.clusters:
            for value in self.clusters[key]:
                array.append([value, key])
        df = pd.DataFrame(np.array(array), columns=['Sample', 'Cluster'])
        df = df.sort_values(by='Sample', ascending=True)
        return df.Cluster.values

    def networkx_mcl(self, G, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
        import networkx as nx
        A = nx.adjacency_matrix(G)
        return self.mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)

    def mcl(self, M, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
        M = self.add_diag(M, mult_factor)
        M = self.normalize(M)
        for i in range(max_loop):
            # logging.info("loop %s" % i)
            M = self.inflate(M, inflate_factor)
            M = self.expand(M, expand_factor)
            if self.stop(M, i): break

        clusters = self.get_clusters(M)
        return M, clusters

    def add_diag(self, A, mult_factor):
        return A + mult_factor * np.identity(A.shape[0])

    def normalize(self, A):
        column_sums = A.sum(axis=0)
        new_matrix = A / column_sums[np.newaxis, :]
        return new_matrix

    def inflate(self, A, inflate_factor):
        return self.normalize(np.power(A, inflate_factor))

    def expand(self, A, expand_factor):
        return np.linalg.matrix_power(A, expand_factor)

    def stop(self, M, i):
        if i % 5 == 4:
            m = np.max(M ** 2 - M) - np.min(M ** 2 - M)
            if m == 0:
                # logging.info("Stop at iteration %s" % i)
                return True
        return False

    def get_clusters(self, A):
        clusters = []
        for i, r in enumerate((A > 0).tolist()):
            if r[i]:
                clusters.append(A[i, :] > 0)
        clust_map = {}
        for cn, c in enumerate(clusters):
            for x in [i for i, x in enumerate(c) if x]:
                clust_map[cn] = clust_map.get(cn, []) + [x]
        return clust_map

class iPlot():
    """
    # Running examples:
    
    >>> import iFeatureOmegaCLI
    >>> import pandas as pd

    # read data to pandas DataFrame
    >>> df = pd.read_csv('data_examples/data_with_labels.csv', sep=',', header=None, index_col=None)

    # Initialize an instance of iPlot
    >>> myPlot = iFeatureOmegaCLI.iPlot(df, label=True)

    # scatter plot
    myPlot.scatterplot(method="PCA", xlabel="PC1", ylabel="PC2", output="scatterplot.pdf")

    # histogram and kernel density
    myPlot.hist()

    # headmap
    myPlot.heatmap(samples=(0, 50), descriptors=(0, 100), output='heatmap.pdf')

    # line chart
    myPlot.line(output='linechart.pdf')

    # boxplot
    myPlot.boxplot(descriptors=(0, 10), output='boxplot.pdf')
    """
    def __init__(self, df, label=False):
        self.dataframe = df
        self.label = label
        self.font_label = {'fontfamily': 'Arial', 'fontsize': 'medium', 'color': '#282828'}
        self.font_tick = {'fontfamily': 'Arial', 'fontsize': 'small', 'color': '#282828'}

    def hist(self, output='histKD.pdf'):
        color =['#4169E1', '#FF0099', '#008080', '#FF9900', '#660033']
        data = copy.deepcopy(self.dataframe.values)
        if data.shape[0] * (data.shape[1]-1) > 32000:
            sel_num = int(32000 / (data.shape[1]-1))
            random_index = random.sample(range(0, len(data)), sel_num)
            data = data[random_index]

        fig = plt.figure(0)
        ax = fig.add_subplot(111)   #这个很重要
        if self.label:
            categories = sorted(set(data[:, 0]))
            max_value = np.max(data[:, 1:].reshape(-1))
            min_value = np.min(data[:, 1:].reshape(-1))           
            if max_value == min_value and max_value == 0:
                pass
            else:
                bins = np.linspace(min_value, max_value, 20)
                for i, c in enumerate(categories):
                    tmp_data = data[np.where(data[:, 0]==c)][:, 1:].reshape(-1)
                    plt.hist(tmp_data, bins=bins, stacked=True, density=True, facecolor=color[i % len(color)], alpha=0.5)
                    X_plot = np.linspace(min_value, max_value, 100)[:, np.newaxis]
                    bandwidth = (max_value - min_value) / 20.0
                    if bandwidth <= 0:
                        bandwidth = 0.1
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tmp_data.reshape((-1, 1)))
                    log_dens = kde.score_samples(X_plot)
                    plt.plot(X_plot[:, 0], np.exp(log_dens), color=color[i % len(color)], label='Category {0}'.format(int(c)))
                plt.legend(prop={'family': 'Arial', 'size': 10})
                plt.xlabel('Value bins', fontdict=self.font_label)
                plt.ylabel('Density', fontdict=self.font_label)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_color('#282828') for label in labels]
                [label.set_fontname('Arial') for label in labels]
                [label.set_size(10) for label in labels]
                plt.savefig(output)
                plt.close()
        else:
            max_value = np.max(data.reshape(-1))
            min_value = np.min(data.reshape(-1))
            if max_value == min_value and max_value == 0:
                pass
            else:
                bins = np.linspace(min_value, max_value, 20)
                tmp_data = data.reshape(-1)
                plt.hist(tmp_data, bins=bins, stacked=True, density=True, facecolor=color[0], alpha=0.5)
                X_plot = np.linspace(min_value, max_value, 100)[:, np.newaxis]
                bandwidth = (max_value - min_value) / 20.0
                if bandwidth <= 0:
                    bandwidth = 0.1
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tmp_data.reshape((-1, 1)))
                log_dens = kde.score_samples(X_plot)
                plt.plot(X_plot[:, 0], np.exp(log_dens), color=color[0])
                plt.xlabel('Value bins', fontdict=self.font_label)
                plt.ylabel('Density', fontdict=self.font_label)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_color('#282828') for label in labels]
                [label.set_fontname('Arial') for label in labels]
                [label.set_size(10) for label in labels]
                plt.savefig(output)
                plt.close()

    def heatmap(self, samples=(0, 50), descriptors=(0, 100), output='heatmap.pdf'):
        data = copy.deepcopy(self.dataframe)
        if self.label:
            data = data.drop(labels=0, axis=1)
        else:
            data = self.dataframe

        ymin, ymax = samples
        xmin, xmax = descriptors
        
        if ymin < 0:
            ymin = 0
        if ymax > data.shape[0] - 1:
            ymax = data.shape[0] - 1
        
        if xmin < 0:
            xmin = 0
        if xmax > data.shape[1] - 1:
            xmax = data.shape[1] - 1
        
        if ymax < ymin or xmax < xmin:
            return False
        else:
            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            plt.grid(False)
            im = plt.imshow(data.values[ymin: ymax, xmin: xmax], cmap=plt.cm.autumn, alpha=0.6)
            plt.colorbar(im)

            ax.tick_params(axis=u'both', which=u'both', right=False, top=False, width=0.6)
            if ymax - ymin <= 20:
                tick = range(ymax-ymin)
            else:
                tick = [int(i) for i in np.linspace(ymin, ymax-1, 20)]
            ax.set_yticks(tick)
            ax.set_yticklabels(data.index[tick])

            if xmax - xmin <= 20:
                tick = range(xmax-xmin)
            else:
                tick = [int(i) for i in np.linspace(xmin, xmax-1, 20)]                
            ax.set_xticks(tick)
            ax.set_xticklabels(data.columns[tick], rotation=90)

            plt.xlabel('Samples', fontdict=self.font_label)
            plt.ylabel('Descriptors', fontdict=self.font_label)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_color('#282828') for label in labels]
            [label.set_fontname('Arial') for label in labels]
            [label.set_size(6) for label in labels]

            plt.savefig(output)
            plt.close()
        return True

    def line(self, output='linechart.pdf'):
        colorlist = ['#3274A1', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']
        sample_categories = None

        fig = plt.figure(0)
        ax = fig.add_subplot(111)

        if self.label:
            sample_categories = sorted(set(self.dataframe.iloc[:, 0]))
            for c, category in enumerate(sample_categories):
                tmp_df = self.dataframe[self.dataframe.iloc[:, 0] == category]
                array_mean = tmp_df.iloc[:, 1:].mean().values.tolist()
                array_x = list(range(1, len(array_mean) + 1))
                bp1 = plt.plot(array_x, array_mean, c=colorlist[c % len(colorlist)], label='Sample category {0}'.format(category))
                plt.legend(prop={'family': 'Arial', 'size': 8})
        else:
            array_mean = self.dataframe.mean().values.tolist()        
            array_x = list(range(1, len(array_mean) + 1))
            bp1 = plt.plot(array_x, array_mean, c='#3274A1')
        
        plt.xlabel('Descriptors', fontdict=self.font_label)
        plt.ylabel('Mean value', fontdict=self.font_label)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(10) for label in labels]
        plt.savefig(output)
        plt.close()
        
    def boxplot(self, descriptors=(0, 10), output='boxplot.pdf'):
        xmin, xmax = int(descriptors[0]), int(descriptors[1])        
        colorlist = ['#3274A1', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']
        sample_categories = None
        fig = plt.figure(0)
        ax = fig.add_subplot(111)

        ax.set_facecolor('white')
        ax.tick_params(axis=u'both', which=u'both', top=False, right=False)
        ax.set_xlabel('Descriptors', fontdict=self.font_label)
        ax.set_ylabel('Value', fontdict=self.font_label)
        ax.spines['left'].set_color('#585858')
        ax.spines['bottom'].set_color('#585858')

        if self.label:
            sample_categories = sorted(set(self.dataframe.iloc[:, 0]))
            data = self.dataframe.iloc[:, 1:]
            if xmin < 0:
                xmin = 0
            if xmax > data.values.shape[1] - 1:
                xmax = data.values.shape[1] - 1
            
            scale_ls = list(range(1, int(xmax)-int(xmin) + 1))
            index_ls = data.columns[int(xmin): int(xmax)+1]
            index_dict = {}
            for i in range(len(scale_ls)):
                index_dict[index_ls[i]] = scale_ls[i]
            
            
            for c, category in enumerate(sample_categories):
                tmp_df = self.dataframe[self.dataframe.iloc[:, 0] == category].iloc[:, 1:]
                tmp_positions = index_dict.values()
                positions_range = [list(np.linspace(p-0.4, p+0.4, len(sample_categories)+2)) for p in tmp_positions]
                positions = [p[c+1] for p in positions_range]
                tmp_width = 0.8 / (len(sample_categories) + 2)

                bp2 = plt.boxplot(tmp_df.values[:, int(xmin): int(xmax)].astype(float), widths=tmp_width, boxprops=dict(lw=0.8), medianprops=dict(lw=0.8),
                                    capprops=dict(lw=0.8), whiskerprops=dict(lw=0.8, linestyle='dashed'), showmeans=True, positions=positions, showfliers=False,
                                    patch_artist=True)
            
                for k, box in enumerate(bp2['boxes']):
                    box.set(edgecolor=colorlist[c % len(colorlist)], facecolor='#FFFFFF')
                for k, median in enumerate(bp2['medians']):
                    median.set(color=colorlist[c % len(colorlist)])

                for elem in ['whiskers', 'caps']:
                    for k, item in enumerate(bp2[elem]):
                        j = k // 2
                        item.set(color=colorlist[c % len(colorlist)])
                for k, flier in enumerate(bp2['fliers']):
                    flier.set(marker='o', markeredgecolor=colorlist[c % len(colorlist)], markeredgewidth=1)
                for k, mean in enumerate(bp2['means']):
                    mean.set(marker='^', markeredgecolor=colorlist[c % len(colorlist)], markeredgewidth=1, markerfacecolor='#FFFFFF')
        else:
            data = self.dataframe
            if xmin < 0:
                xmin = 0
            if xmax > data.values.shape[1] - 1:
                xmax = data.values.shape[1] - 1
            
            scale_ls = list(range(1, int(xmax)-int(xmin) + 1))
            index_ls = data.columns[int(xmin): int(xmax)+1]
            index_dict = {}
            for i in range(len(scale_ls)):
                index_dict[index_ls[i]] = scale_ls[i]

            bp2 = plt.boxplot(self.dataframe.values[:, int(xmin): int(xmax)].astype(float), widths=0.5, boxprops=dict(lw=0.8), medianprops=dict(lw=0.8),
                                    capprops=dict(lw=0.8), whiskerprops=dict(lw=0.8, linestyle='dashed'), showmeans=True, showfliers=False,
                                    patch_artist=True)
            for k, box in enumerate(bp2['boxes']):
                box.set(edgecolor=colorlist[0], facecolor='#FFFFFF')
            for k, median in enumerate(bp2['medians']):
                median.set(color=colorlist[0])

            for elem in ['whiskers', 'caps']:
                for k, item in enumerate(bp2[elem]):
                    j = k // 2
                    item.set(color=colorlist[0])
            for k, flier in enumerate(bp2['fliers']):
                flier.set(marker='o', markeredgecolor=colorlist[0], markeredgewidth=1)
            for k, mean in enumerate(bp2['means']):
                mean.set(marker='^', markeredgecolor=colorlist[0], markeredgewidth=1, markerfacecolor='#FFFFFF')

        ax.set_xticks(list(range(1, int(xmax)-int(xmin) + 1)))
        ax.set_xticklabels(index_dict, rotation=45)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]
        plt.subplots_adjust(bottom=0.25, top=0.95)
        plt.savefig(output)
        plt.close()
            
    def circularplot(self, samples=True, sample_range=(0, 30), cutoff=0.5, output='circularplot.pdf'):
        if self.label:
            data = self.dataframe.iloc[:, 1:]
        else:
            data = self.dataframe
        
        if not samples:
            data = data.T
        
        data = data.iloc[sample_range[0]: sample_range[1], :]
        df, theta_g = self.generate_data(data.values)

        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor('white')
        ax.grid(False)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_xticks(theta_g)
        ax.set_xticklabels(data.index)
        t=np.linspace(0, 1, 100)
        
        for i in range(len(df)):
            if df.iloc[i]['xielv1'] != 0:
                x_c=(1-t)*(1-t)*df.iloc[i]['Startx']+2*t*(1-t)*df.iloc[i]['control_x']+t*t*df.iloc[i]['Endx']
                y_c=(1-t)*(1-t)*df.iloc[i]['Starty']+2*t*(1-t)*df.iloc[i]['control_y']+t*t*df.iloc[i]['Endy']
                
                r=np.power(y_c*y_c+x_c*x_c,0.5)
                theta=np.arctan2(y_c,x_c)
                tmp_color = self.get_color(df.iloc[i]['corr'])
                if abs(df.iloc[i]['corr']) >= cutoff:
                    plt.plot(theta, r, color=tmp_color, alpha=1, linewidth=0.8)
        plt.bar(x=theta_g, height=[0.2], bottom=5.1, width=0.01,color='#3399FF',alpha=0.8)

        for i in range(len(theta_g)):
            ax.text(
                theta_g[i], 6, 
                data.index[i],
                ha='center', 
                va= 'center',
                fontsize=4,
                color=(0.1, 0.2, 0.5),             
                rotation= (360/len(theta_g)*(i)),
                alpha=0.5
            )

        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('#ffffff') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(1) for label in labels]
        plt.subplots_adjust(bottom=0.25)

        ax1 = fig.add_axes([0.20, 0.05, 0.60, 0.03])
        category_names = ['[-1, -0.8)', '[-0.8, -0.6]', '[-0.6, -0.4)', '[-0.4, -0.2)', '[-0.2, 0)', '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1]'][::-1]
        results = {
            'Color range': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]    
        }
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))
        
        ax1.invert_yaxis()
        ax1.xaxis.set_visible(False)
        ax1.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax1.barh(labels, widths, left=starts, height=0.1, label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # ax1.bar_label(rects, label_type='center', color=text_color)
        ax1.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', prop={'family': 'Arial', 'size': 3})        
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_color('#000000') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(5) for label in labels]
        plt.savefig(output)
        plt.close()
        
    def generate_data(self, data):
        corr_res = np.corrcoef(data)
        
        theta = 2 * np.pi * np.linspace(0, 1, corr_res.shape[0] + 1)
        r = np.array([5] * (corr_res.shape[0] + 1))
        X = []
        Y = []

        for i in range(len(theta)):
            res = cmath.rect(r[i], theta[i])
            X.append(res.real)
            Y.append(res.imag)
        
        X = np.array(X)
        Y = np.array(Y)

        tmp = []

        for i in range(len(corr_res)):
            for j in range(i+1, len(corr_res)):
                # if abs(corr_res[i][j]) >= cutoff:
                tmp.append([X[i], Y[i], X[j], Y[j], i, j, corr_res[i][j]]) # [StartX, StartY, EndX, EndY, NodeX, NodeY, corr]
                    
        df = pd.DataFrame(np.array(tmp), columns=['Startx', 'Starty', 'Endx', 'Endy', 'NodeX', 'NodeY', 'corr'])
        df['xielv1']=(df['Starty']-df['Endy'])/(df['Startx']-df['Endx'])
        df['xielv2']=-1/df['xielv1']
        df['central_point_x']=(df['Startx']+df['Endx'])/2
        df['central_point_y']=(df['Starty']+df['Endy'])/2
        df['axes_x']=df['central_point_x']-df['central_point_x']*(0.05)
        df['axes_y']=df['axes_x']*df['xielv2']
        df['control_x']=df['central_point_x']-df['axes_x']
        df['control_y']=df['central_point_y']-df['axes_y']

        return df, theta[0: -1]

    def get_color(self, val):
        if -1 <= val < -0.8:
            return '#3FAA59'
        elif -0.8 <= val < -0.6:
            return '#78C565'
        elif -0.6 <= val < -0.4:
            return '#A9D06C'
        elif -0.4 <= val < -0.2:
            return '#D1EC86'
        elif -0.2 <= val < 0:
            return '#F1F9AC'
        elif 0 <= val < 0.2:
            return '#FFF3AC'
        elif 0.2 <= val < 0.4:
            return '#FED884'
        elif 0.4 <= val < 0.6:
            return '#FDB163'
        elif 0.6 <= val < 0.8:
            return '#F67F4B'
        elif 0.8<= val <= 1:
            return '#E54E35'
        else:
            return '#778899' 

    def scatterplot(self, method='PCA', xlabel='PC1', ylabel='PC2', output='scatterplot.pdf'):
        colorlist = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']
        marklist = ['o'] * 9 + ['v'] * 9 + ['^'] * 9 + ['+'] * 9
        fontdict = {
            'family': 'Arial',
            'size': 8,
            'color': '#282828',
        }
        prefix = 'Category: '

        if self.label:
            data = self.dataframe.iloc[:, 1:]
            datalabel = self.dataframe.iloc[:, 0].values
        else:
            data = self.dataframe
            datalabel = np.zeros(len(self.dataframe))
        
        analysis = iAnalysis(data)
        if method == 'PCA':
            analysis.PCA(2)
        elif method == 't_sne':
            analysis.t_sne(2)
        elif method == 'LDA':
            analysis.LDA(2)
        else:
            analysis.PCA(2)        

        plot_data = []
        categories = sorted(set(datalabel))
        for c in categories:
            tmp_data = analysis.dimension_reduction_result[np.where(datalabel == c)]
            plot_data.append([c, tmp_data])

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')        
        ax.set_xlabel(xlabel, fontdict=fontdict)
        ax.set_ylabel(ylabel, fontdict=fontdict)
       
        for i, item in enumerate(plot_data):
            plt.scatter(item[1][:, 0], item[1][:, 1], color=colorlist[i % len(colorlist)], s=70,
                             marker=marklist[i % len(marklist)], label='%s %s' % (prefix, int(item[0])),
                             edgecolor="w")

        plt.legend(prop={'family': 'Arial', 'size': 8})
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(10) for label in labels]
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        plt.savefig(output)
        plt.close()

        
 
if __name__ == '__main__':
    # dna = iDNA('./data_examples/DNA_sequences.txt')
    # dna.import_parameters('parameters/DNA_parameters_setting.json')
    # status = dna.get_descriptor('Kmer')
    # dna.display_feature_types()
    # print(dna.encodings)
    # print(dna.error_msg)


    # plot = iPlot(dna.encodings, label=False)
    # plot.scatterplot()
    # dna.to_svm('tmp.svm')
    # data = iAnalysis(dna.encodings)
    # data.PCA()
    # data.cluster_to_csv()
    # data.ZScore()
    # data.dimension_to_csv()
    
    
    # print(data.feature_normalization_data)
    # ok = data.ClusteringScatterPlot('Clustering', 'cluster.pdf')
    # print(ok)
    # print(data.error_msg)

    # df = pd.read_csv('data_examples/data_with_labels.csv', sep=',', header=None, index_col=None)   
    
    # myPlot = iPlot(df, label=True)
    # myPlot.scatterplot()
    # myPlot.hist()
    # myPlot.heatmap(samples=(0, 50), descriptors=(0, 100), output='heatmap.pdf')
    # myPlot.line(output='linechart.pdf')
    # myPlot.boxplot(descriptors=(0, 10), output='boxplot.pdf')
    # myPlot.circularplot(samples=True, sample_range=(0, 30), cutoff=0.5, output='circularplot.pdf')

    ligand = iLigand('data_examples/Chemical_SMILES.txt')
    status = ligand.get_descriptor('Basak')
    print(ligand.encodings)


