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