
#%%


'pass_project_code'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""""   Function build_data()
The function gives us for the name Seymour and Diane from the Database the nuber of newborns for each years 
Parameters:
----------
t : a year from 1880 to 2021

output:
-------
d : a list for each years up to t the nuber of newborns with the first name Diane
s : the same list with the first name Seymour
d_cumulate : a list which contains the number of Diane from 1880 to t
s_cumulate : the same liste with the name Seymour 


"""

path = "names(1)/yob"
def build_data(t):
    d = []
    d_cumulate = [0]
    s = []
    s_cumulate = [0]
    for i in range(1880, t+1):
        filename = path+str(i)+".txt"
        f = open(filename, 'r')
        lines = f.readlines()
        n = len(lines)
        nd = 0
        ns = 0
        for i in range(n):
            
            name, _, number = lines[i].split(',')
            if name == 'Diane':
                nd += int(number)
            elif name == 'Seymour':
                ns += int(number)
        
        f.close()

        d.append(nd)
        d_cumulate.append(nd+d_cumulate[-1])
        s.append(ns)
        s_cumulate.append(ns+s_cumulate[-1])

    
    d_cumulate.pop(0)
    s_cumulate.pop(0)

    return d, s, d_cumulate, s_cumulate






#%%

# We take all data for 1880
t = 2021
diane, seymour, d_cumulate, s_cumulate = build_data(t)


"""""   Function Bass_model()
The function of the Bass Model which gives the number of adopters at time t 
Parameters:
---------
t : Time
q, etha_0 and K : parameters of the Bass model

"""
def Bass_model(t, q, etha_0, K):
    return K* 1/(1 + (1-etha_0)/etha_0 * np.exp(-q * (t-1880)))


# We choose a sample of all data in order to provide the curve fit function
time_period  = np.array(range(1880, 2022))
learning_data = np.array(d_cumulate)

# The curve function will give us best constants by minimizing the error 
out_1, _ = curve_fit(Bass_model, time_period, learning_data)
q, etha_0, K = out_1
print(q)

# The logistic function is the solution of the differential equation of the Bass Model
def logistic(t, q, etha_0):
    return 1/(1 + (1-etha_0)/etha_0 * np.exp(-q * (t-1880)))


""""" Function prediction()
The prediction function is which gives from the constant of the model the product K * derivative of 
the logistic function

"""
def prediction(t, q, etha_0, K):
    return K * (1-etha_0)/etha_0 * q * np.exp(- q * (t - 1880)) * logistic(t, q, etha_0)**2

# Sigmoide 
y_bass_model = np.array(Bass_model(time_period, q, etha_0, K))
plt.plot(time_period, y_bass_model)
plt.show()

# prediction with Whole database wich is the best prediction we can have
y_fit = np.array(prediction(time_period, q, etha_0, K))
plt.plot(time_period, y_fit, color = 'black', label = 'Fit curve')
plt.plot(time_period, diane, 'ro', label = 'Name Diane')
plt.xlabel('Time (from 1880 to 2021)')
plt.ylabel('Number of Baby')
plt.plot(time_period, seymour, color = 'blue', label = 'Name Seymour')
plt.legend()
plt.title('Learning the Bass model on the whole Database')
plt.show()


# %%

#---------------------------------Prediction----------------------------------------------------

# Get all data from 1880 to 1960
diane_p, _,d_p_cumulate, _ = build_data(1960) 

learning_time  = np.array(range(1880, 1961))
learning_data = np.array(d_p_cumulate) # we fit with the data frome 1880 to 1960

out_1, _ = curve_fit(Bass_model, learning_time, learning_data)
q, etha_0, K = out_1



# plot the fit and the real incidence
y_fit = np.array(prediction(time_period, q, etha_0, K))
plt.plot(time_period, y_fit, color = 'black', label = 'Prediction')
plt.plot(time_period, diane, 'ro', label = 'Real distribution ')
plt.xlabel('Time (from 1880 to 2021)')
plt.ylabel('Number of Baby')
plt.legend()
plt.title('Provide with data from 1880 to 1960, (peak=1953,newborns=22975)')
plt.show()




print(diane.index(max(diane))+1880, max(diane)) # the peak and the time of the peak of the first name diane
print(y_fit.tolist().index(max(y_fit))+1880, max(y_fit)) # the peak and the time of the peak of the first name diane with the prediction


#%%
# ---------------------------RFIM model-----------------------------
#The code comes from the TD1

class RFIM_rational(object):
    
    def __init__(self, J, h, N, delta_q):

        self.J = J
        self.h = h
        self.N = N
        self.S= np.random.choice([-1,1], size = N)
        if delta_q == 0: 
            self.f = np.zeros(N)
        else:
            self.f = np.random.normal(loc = 0, scale = delta_q, size = N)
        
    def moy(self):
        return sum(self.S)/self.N
        
                                   
    def p(self, i):
        '''Computes the local opinion polarization for agent i p_i
        inputs: agent label i
        returns: the field p_i'''
        local_m = (self.S.sum() - self.S[i]) / (self.N)
        return self.f[i] + self.h + self.J * local_m
    
    def flip(self,i):
        '''Changes S_i -> -S_i according to the dynamic rule
        input: agent label i
        returns: 0 if the agent didn't change opinion
                 1 if the agent changed opinion
        '''
        should_flip = self.S[i] * np.sign(self.p(i))
        if should_flip == -1:
            self.S[i] *= -1
            return 1
        else:
            return 0
    
    def sweep(self):
        '''Goes through all N agents and tries to flip them
        input: none
        returns: number of agents that changed their mind
        '''
        s = 0
        for i in range(N):
            s += self.flip(i)
        return s
    
    def equilibrate(self): # Needs modification for finite temperature case
        '''Find the equilibrium of the system'''
        flips = self.sweep()
        while flips > 0:
            flips = self.sweep()

J = 1
delta_q = 1/0.18 #Comes from the Bass model fit on the whole data set
N = 100
h = np.linspace(-10, 10, 30)

m = np.zeros(len(h))
for i in range(len(h)):
    sim = RFIM_rational(J,h[i],N, delta_q)
    sim.equilibrate()
    m[i] = sim.moy()

plt.plot(h, m)
plt.xlabel('h : The Media coverage')
plt.ylabel('Average')
plt.title('The average adoption')
plt.show()
    





# %%
