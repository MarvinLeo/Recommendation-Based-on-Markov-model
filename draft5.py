
# coding: utf-8

# In[2]:

import numpy as np

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
        phi = Cu / (j+1 - Au)
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
        grad = 1 / (j+1-Au)
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
        grad = Cu / (j+1-Au)**2
    return grad

def phi_gradient_Bu (Cu, Au ,Bu, j, method = "Exponential"):
    if Cu <= 0 or Au < 0 or Bu < 0:
        raise ValueError("parameter is lower than 0")
    if Cu > 1 or Au > 1 or Bu > 1:
        raise ValueError("parameter is greater than 1")
    if method == "Exponential":
        grad = -Cu * j**(1-Au) * np.exp(-Bu*j)
    return grad

# calculate one step Probability
# sequence is a 1*n ndarray which only contain the sequence of music listening record
def get_one_step_Prob (a, b, sequence):
    P = 0
    PosiA = np.argwhere(sequence == a)
    PosiB = np.argwhere(sequence == b)
    if PosiA.size == 0 or PosiB.size == 0:
        P = 0
    else:
        NumA = PosiA.size
        for i in range (0, NumA):
            RightB = PosiB[PosiB > PosiA[i]]
            NumRightB = RightB.size
            P = P + 1.0 / NumA *(1 - 0.5**NumRightB)
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

def get_gradient_theta (record, theta, t, method2 = "logLinear", method1 = "logistic"):
    phi_u = theta[0]
    Cu = theta[1]
    Au = theta[2]
    #Bu = theta[3]
    sequence = record[range(0,t)]
    target = record[t]
    numerator_phi_u = np.zeros(t)
    numerator_Cu = np.zeros(t)
    numerator_Au = np.zeros(t)
    #numerator_Bu = np.zeros(t)

    denominator = np.zeros(t)


    for k in range(1, t):
        f_ut = get_f_ut(sequence[:t-k], record[t-k])
        Bu = 0.5

        numerator_phi_u[k-1] = get_gamma_gradient(phi_u, f_ut, method1) * get_phi(Cu, Au ,Bu, k, method2) * get_one_step_Prob(record[t-k], target, record)
        numerator_Cu[k-1] = phi_gradient_Cu (Cu, Au ,Bu, k, method2) * get_gamma(phi_u, f_ut, method1) * get_one_step_Prob(record[t-k], target, record)
        numerator_Au[k-1] = phi_gradient_Au (Cu, Au ,Bu, k, method2) * get_gamma(phi_u, f_ut, method1) * get_one_step_Prob(record[t-k], target, record)
        #numerator_Bu[k-1] = phi_gradient_Bu (Cu, Au ,Bu, k) * get_gamma(phi_u, f_ut, method1) * get_one_step_Prob(record[t-k], target, record)
        denominator[k-1] = get_gamma(phi_u, f_ut, method1) * get_phi(Cu, Au ,Bu, k, method2) * get_one_step_Prob(record[t-k], target, record)
        
    result = np.zeros(3)
    
    #print(numerator_phi_u)
    #print(numerator_Cu)
    #print(numerator_Au)
    #print(numerator_Bu)
    #print(np.sum(denominator))
    result[0] = sum(numerator_phi_u) / float(sum(denominator))
    result[1] = sum(numerator_Cu) / float(sum(denominator))
    result[2] = sum(numerator_Au) / float(sum(denominator))
    #result[3] = sum(numerator_Bu) / float(sum(denominator))
    return result

def sum_gradient (record, theta, t, method2 = "logLinear", method1 = "logistic"):
    sum_g = 0
    for i in range(int(0.2*t),t):
        sum_g = sum_g + get_gradient_theta (record, theta, i, method2, method1)
    return sum_g


def re_gradient(theta, gre_fun, record, t, n_iter = 10000, convergence = None, alpha = 0.0005, method2 = "logLinear", method1 = "logistic"):
    '''
    compute gradient scent algorithm with convergence or iterate times
    where gre_fun is function
    theta are parameters
    record are the listening series
    t is how long of time series for traing you select
    '''
    
        
    gradient_theta = gre_fun(record, theta, t, method2)
    if convergence == None:
        for _ in range(n_iter):
            theta = theta + alpha * gradient_theta
            if theta[1] <= 0:
                theta[1] = 0.001
            if theta[2] < 0:
                theta[2] = 0
                
            if theta[1] > 1:
                theta[1] = 1
            if theta[2] > 1:
                theta[2] = 1
                
            if theta[0] < 0:
                theta[0] = 0
                
            if method2 == "Hyperbolic":
                if theta[2] >= 1:
                    theta[2] = 0.999
                    
            if method1 == "rational":
                if theta[0] > 1:
                    theta[0] = 1
            print(_)
            print(gradient_theta)
            print(theta)
            gradient_theta = gre_fun(record, theta, t, method2)
    else:
        while np.linalg.norm(gradient_theta) > convergence:
            theta = theta + alpha * gradient_theta
            gradient_theta = gre_fun(record, theta, t, method2)
    return theta



# In[3]:

import pandas as pd
df = pd.read_csv('user_000007.csv')
record = df.item_index[range(0,df.item_index.size-1)]


# In[4]:

import pandas as pd
df = pd.read_csv('user_000007.csv')
record = df.item_index[range(0,df.item_index.size-100)]

theta1 = np.array([4.0, 1.0, 0.0])
x_list = np.sort(np.unique(record))

def predict(theta, x_list, seq):
    '''
    where theta is trained parameters
    x_list is the item list
    seq is previous listening record
    '''
    phi_u = theta[0]
    Cu = theta[1]
    Au = theta[2]
    #Bu = theta[3]
    Bu = 0.5
    prob = np.zeros(x_list.size)
    
    for i in range(x_list.size):
        for k in range(seq.size-1):
            xk = seq[seq.size-1-k]
            f_ut = get_f_ut(seq[:seq.size-1-k], xk)
            prob[i] += get_phi(Cu, Au, Bu, k) * get_gamma(phi_u, f_ut) * get_one_step_Prob(xk, x_list[i], seq)
        print(i)
        print(prob[i])
    return prob


# In[5]:

print(predict(theta1, x_list, record))


# In[ ]:

theta = np.array([4, 0.5, 0.9])
print(re_gradient(theta, sum_gradient, record, 100, alpha = 0.01))

