import numpy as np

def get_gamma(phi_u, f_ut, method = "logistic"):
    '''
    build experience gamma with two different methods: logistic,
    rational
    
    Based on:
    function of the frequency: f_ut: user's consumption on item x by time t
    personalized parameter for experience: phi_u
    '''
#check the constraint of parameters
    if f_ut.all() < 0 or phi_u.all() <0:
        raise ValueError("parameter is lower than 0")
    if method == "logistic": 
        gamma = 1 + np.exp(-phi_u * f_ut)
        gamma = 2/gamma
    elif method == "rational":
        #check the constraint of parameters
        if f_ut.all() > 1 or phi_u.all() > 1:
            raise ValueError("parameter is larger than 1")
        gamma = 1 + f_ut^phi_u   
    return gamma
    
def get_gamma_gradient(phi_u, f_ut, method = "logistic"):
    if f_ut.all() < 0 or phi_u.all() <0:
        raise ValueError("parameter is lower than 0")
    if method == "logictic":
        gammaGradient = 2 * f_ut* np.exp(-phi_u*f_ut)
        gammaGradient = gammaGradient/ (1 + np.exp(-phi_u * f_ut)) ** 2
    elif method == "rational":
        #check the constraint of parameters
        if f_ut.all() > 1 or phi_u.all() > 1:
            raise ValueError("parameter is larger than 1")
        gammaGradient = f_ut ** phi_u
        gammaGradient = gammaGradient * np.log(f_ut)
    return gammaGradient 
