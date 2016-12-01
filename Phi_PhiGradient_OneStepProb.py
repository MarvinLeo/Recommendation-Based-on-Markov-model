
# coding: utf-8

# In[1]:

import numpy as np

'''
def get_phi_logLinear(Cu, Au ,j):
    phi = Cu * j^(-Au)
    return phi

def get_phi_exp(Cu, Au ,Bu, j):
    phi = Cu * j^(-Au) * np.exp(-Bu*j)
    return phi

def get_phi_hyper(Cu, Au ,j):
    phi = Cu / (j - Au)
    return phi

def gradient_phi_logLinear_Au(Cu, Au ,j):
    grad = -Cu * j^(-Au) * np.log(j)
    return grad

def gradient_phi_logLinear_Cu(Cu, Au ,j):
    grad = j^(-Au)
    return grad

def gradient_phi_exp_Au(Cu, Au ,Bu, j):
    grad = -Cu * j^(-Au) * np.exp(-Bu*j) * np.log(j)
    return grad

def gradient_phi_exp_Bu(Cu, Au ,Bu, j):
    grad = -Cu * j^(1-Au) * np.exp(-Bu*j)
    return grad

def gradient_phi_exp_Cu(Cu, Au ,Bu, j):
    grad = j^(-Au) * np.exp(-Bu*j)
    return grad

def gradient_phi_hyper_Au(Cu, Au ,j):
    grad = Cu / (j-Au)^2
    return grad

def gradient_phi_hyper_Cu(Cu, Au ,j):
    grad = 1 / (j-Au)
    return grad
'''


def get_phi (Cu, Au ,Bu, j, method = "logLinear"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "logLinear": 
        phi = Cu * j^(-Au)
    elif method == "Exponential":
        phi = Cu * j^(-Au) * np.exp(-Bu*j)
    elif method == "Hyperbolic":
        phi = Cu / (j - Au)
    return phi

def phi_gradient_Cu (Cu, Au ,Bu, j, method = "logLinear"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "logLinear": 
        grad = j^(-Au)
    elif method == "Exponential":
        grad = j^(-Au) * np.exp(-Bu*j)
    elif method == "Hyperbolic":
        grad = 1 / (j-Au)
    return grad

def phi_gradient_Au (Cu, Au ,Bu, j, method = "logLinear"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "logLinear": 
        grad = -Cu * j^(-Au) * np.log(j)
    elif method == "Exponential":
        grad = -Cu * j^(-Au) * np.exp(-Bu*j) * np.log(j)
    elif method == "Hyperbolic":
        grad = Cu / (j-Au)^2
    return grad

def phi_gradient_Bu (Cu, Au ,Bu, j, method = "Exponential"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "Exponential":
        grad = -Cu * j^(1-Au) * np.exp(-Bu*j)
    return grad



# In[62]:

# if a is in the sample
# sample is a 1*n ndarray
def findNum (sample, a):
    result = np.argwhere(sample == a)
    if result.size == 0:
        return False
    else:
        return True
    
# if sequence ab is in the sample
# sample is a 1*n ndarray
def findSequenceAB (sample, a, b):
    if findNum (sample, a) and findNum (sample, b):
        PosiA = np.argwhere(sample == a)
        PosiB = np.argwhere(sample == b)
        if a != b:
            if PosiB[0]> np.max(PosiA):
                return True
            else:
                return False
        else:
            if PosiA.size/2 > 1:
                return True
            else:
                return False
    else:
        return False

# calculate one step Probability
# sequence is a 1*n ndarray which only contain the sequence of music listening record
def get_one_step_Prob (a, b, sequence):
    Ia = 0;
    Iab = 0;
    recordLength = sequence.size
    for j in range(0, 1000):
        sampleLength = np.random.random_integers(1, high = recordLength-1)
        samplePosition = np.random.random_integers(0, high = recordLength-1, size = sampleLength)
        samplePosition = np.sort(np.unique(samplePosition))
        sample = np.array([])
        for i in samplePosition:
            sample = np.append(sample, sequence[i])
        if findNum (sample, a):
            Ia = Ia + 1
        if findSequenceAB (sample, a, b):
            Iab = Iab + 1
    P = float(Iab) / float(Ia)
    return P
        
    


# In[61]:


x = np.array([5,7,9,10,4,2,1,0,8])
print get_one_step_Prob (7, 1, x)

