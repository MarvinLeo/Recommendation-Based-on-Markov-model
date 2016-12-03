
# coding: utf-8

# In[5]:

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
        phi = Cu * j**(-Au)
    elif method == "Exponential":
        phi = Cu * j**(-Au) * np.exp(-Bu*j)
    elif method == "Hyperbolic":
        phi = Cu / (j - Au)
    return phi

def phi_gradient_Cu (Cu, Au ,Bu, j, method = "logLinear"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "logLinear": 
        grad = j**(-Au)
    elif method == "Exponential":
        grad = j**(-Au) * np.exp(-Bu*j)
    elif method == "Hyperbolic":
        grad = 1 / (j-Au)
    return grad

def phi_gradient_Au (Cu, Au ,Bu, j, method = "logLinear"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "logLinear": 
        grad = -Cu * j**(-Au) * np.log(j)
    elif method == "Exponential":
        grad = -Cu * j**(-Au) * np.exp(-Bu*j) * np.log(j)
    elif method == "Hyperbolic":
        grad = Cu / (j-Au)**2
    return grad

def phi_gradient_Bu (Cu, Au ,Bu, j, method = "Exponential"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "Exponential":
        grad = -Cu * j**(1-Au) * np.exp(-Bu*j)
    return grad



# In[14]:

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
    for j in range(0, 100):
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
    if Ia == 0:
        P = 0
    else:
        P = float(Iab) / float(Ia)
    return P
        
def get_gamma(phi_u, f_ut, method = "logistic"):
    '''
    build experience gamma with two different methods: logistic,
    rational
    
    Based on:
    function of the frequency: f_ut: user's consumption on item x by time t
    personalized parameter for experience: phi_u
    '''
#check the constraint of parameters
    if f_ut < 0 or phi_u <0:
        raise ValueError("parameter is lower than 0")
    if method == "logistic": 
        gamma = 1 + np.exp(-phi_u * f_ut)
        gamma = 2/gamma
    elif method == "rational":
        #check the constraint of parameters
        if f_ut > 1 or phi_u > 1:
            raise ValueError("parameter is larger than 1")
        gamma = 1 + f_ut^phi_u   
    return gamma
    
def get_gamma_gradient(phi_u, f_ut, method = "logistic"):
    if f_ut < 0 or phi_u <0:
        raise ValueError("parameter is lower than 0")
    if method == "logistic":
        gammaGradient = 2 * f_ut* np.exp(-phi_u*f_ut)
        gammaGradient = gammaGradient/ (1 + np.exp(-phi_u * f_ut)) ** 2
    elif method == "rational":
        #check the constraint of parameters
        if f_ut > 1 or phi_u > 1:
            raise ValueError("parameter is larger than 1")
        gammaGradient = f_ut ** phi_u
        gammaGradient = gammaGradient * np.log(f_ut)
    return gammaGradient 

def get_f_ut (sequence, x):
    Posi = np.argwhere(sequence == x)
    f_ut = float(Posi.size) / float(sequence.size)
    return f_ut

def get_gradient_theta (record, theta, t):
    phi_u = theta[0]
    Cu = theta[1]
    Au = theta[2]
    Bu = theta[3]
    sequence = record[range(0,t-1)]
    target = record[t]
    numerator_phi_u = np.zeros(t)
    numerator_Cu = np.zeros(t)
    numerator_Au = np.zeros(t)
    numerator_Bu = np.zeros(t)

    denominator = np.zeros(t)


    for k in range(1, t):
        f_ut = get_f_ut(sequence, record[t-k])
        numerator_phi_u[k-1] = get_gamma_gradient(phi_u, f_ut) * get_phi(Cu, Au ,Bu, k) * get_one_step_Prob(record[t-k], target, sequence)
        numerator_Cu[k-1] = phi_gradient_Cu (Cu, Au ,Bu, k) * get_gamma(phi_u, f_ut) * get_one_step_Prob(record[t-k], target, sequence)
        numerator_Au[k-1] = phi_gradient_Au (Cu, Au ,Bu, k) * get_gamma(phi_u, f_ut) * get_one_step_Prob(record[t-k], target, sequence)
        numerator_Bu[k-1] = phi_gradient_Bu (Cu, Au ,Bu, k) * get_gamma(phi_u, f_ut) * get_one_step_Prob(record[t-k], target, sequence)
        denominator[k-1] = get_gamma(phi_u, f_ut) * get_phi(Cu, Au ,Bu, k) * get_one_step_Prob(record[t-k], target, sequence)
        
    result = np.zeros(4)
    
    print(numerator_phi_u)
    print(numerator_Cu)
    print(numerator_Au)
    print(numerator_Bu)
    print(denominator)
    #result[0] = numerator_phi_u.sum / float(denominator.sum)
    #result[1] = numerator_Cu.sum / float(denominator.sum)
    #result[2] = numerator_Au.sum / float(denominator.sum)
    #result[3] = numerator_Bu.sum / float(denominator.sum)
    #return result

def re_gradient(theta, gre_fun, record, t, n_iter = 50000, convergence = None, alpha = 0.005):
    '''
    compute gradient scent algorithm with convergence or iterate times
    where gre_fun is function
    theta are parameters
    record are the listening series
    t is how long of time series for traing you select
    '''
    gradient_theta = gre_fun(record, theta, t)
    if convergence == None:
        for _ in range(n_iter):
            theta = theta + alpha * gradient_theta
            gradient_theta = gre_fun(record, theta, t)
    else:
        while np.linalg.norm(gradient_theta) > convergence:
            theta = theta + alpha * gradient_theta
            gradient_theta = gre_fun(record, theta, t)
    return theta
    


# In[8]:


x = np.array([5,7,9,10,4,2,1,0,8])
print get_one_step_Prob (7, 1, x)


# In[15]:

import pandas as pd
df = pd.read_csv('user_000007.csv')
record = df.item_index[range(0,df.item_index.size-1)]
theta = np.array([0.5, 0.5, 0.5, 0.5])
get_gradient_theta (record, theta, 100)


# In[ ]:



