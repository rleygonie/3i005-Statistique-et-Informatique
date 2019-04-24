import io
import math
import numpy as np


#Définition des nucléotides
nucleotide = {'A':0,'C':1,'G':2,'T':3}
nucleotide_indetermine = {'A':0,'C':1,'G':2,'T':3,'N':-1}

def decode_sequence(sequence):
    inv_nucleotide = {v:k for k, v in nucleotide_indetermine.items()}
    to_str = ""
    for i in sequence:
        if(i in inv_nucleotide):
            to_str += inv_nucleotide[i]
        else:
            to_str += 'N'
    return to_str
    

def encode_sequence(string):
    to_list = []
    for base in string:
        if(base in nucleotide_indetermine):
            to_list.append(nucleotide_indetermine[base])
    return to_list

def read_fasta(fasta_filepath):
    fasta_file = io.open(fasta_filepath, 'r')
    current_sequence = ""
    sequences_dict = {}
    for line in fasta_file.readlines():
        if(line[0] == '>'):
            current_sequence = line
            sequences_dict[line] = []
        else:
            for nucl in line:
                if(nucl in nucleotide_indetermine):
                    sequences_dict[current_sequence].append((int) (nucleotide_indetermine[nucl]))

    return sequences_dict
    
def nucleotide_count(sequence):
    count = [0 for k in nucleotide]
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
    return count

def nucleotide_frequency(sequence):
    count = [0 for k in nucleotide]
    n_nucl = 0.
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
            n_nucl += 1.
    return count/(np.sum(count))

def logproba(liste_entiers,m):
    total = 0
    for lettre in liste_entiers:
        total+= math.log(m[lettre])
    return total

def logprobafast(compte_lettres,m):
    return sum([math.log(m[i])*compte_lettres[i] for i in range(len(m))])
        
    
def code(m,k):
    i = k-1
    res = 0
    for lettre in m:
        res += lettre * (4 ** i)
        i -= 1
    return res
    
def inverse_code(i,k):
    res = []
    for ind in range(k-1,-1,-1):
        res.append(i // (4**ind))
        i = i % (4**ind)
    return res
    
def compte_freq_mots(sequence,taille):
    resTmp = {}
    for i in range(len(sequence)-taille+1):
        sousSeq = code(sequence[i:i+taille],taille)
        if not sousSeq in resTmp.keys():
            resTmp[sousSeq] = 1
        else:
            resTmp[sousSeq] += 1 
    res = {}
    for k in resTmp.keys():
        lTmp = inverse_code(k,taille)
        buff = decode_sequence(lTmp)
        res[buff] = resTmp[k]
    return res

def comptage_attendu(freq,taille,longueur):
    return 0


def affGraph2D(sequence,taille):
    return 0

import random

def simule_sequence(lg,m):
    res = []
    for i in range(lg):
        r = random.random()
        res.append((int)(r >= m[0]) +(int)(r >= m[0]+m[1]) +(int)(r >= m[0]+m[1]+m[2]))
    return res
    
    
    
    
    
    
    
    
    
    
    
    