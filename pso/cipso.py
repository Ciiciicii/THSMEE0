import numpy as np 
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, fitness, LB, UB, swarm_size, epochs, args, params, normalize):
        self.fitness = fitness
        self.LB = LB
        self.UB = UB
        self.swarm_size = swarm_size
        self.epochs = epochs
        self.args = args
        self.params = params
        self.normalize = normalize

        self.sol = []
        self.assignment = 0
        self.lbest = []
        self.lfit = np.zeros((epochs, swarm_size))
        self.gbest = np.zeros((epochs, len(LB)))
        self.gfit = []
        self.assigned = []

        self.optimize()
        

    def optimize(self):
        num_dims = len(self.LB)

        if self.normalize:
            orig_LB = self.LB.copy()
            orig_UB = self.UB.copy()
            self.LB = np.zeros(num_dims)
            self.UB = np.ones(num_dims)
            orig_bounds = orig_LB, orig_UB
        else:
            orig_bounds = []
        
        if type(self.params["Vmax"]) is tuple:
            V1 = (self.params["Vmax"])[0]
            V2 = (self.params["Vmax"])[1]
            # UPDATE THIS LATER
            Vmin = -V1 * (self.UB - self.LB)
            Vmax = V1
        else:
            Vmin = self.params["Vmax"]
            Vmax = self.params["Vmax"]
        
        # LOGISTIC MAP
        position = np.zeros((self.swarm_size, num_dims))
        
        position[0, :] = np.random.rand(1, num_dims)
        for i in range(1, self.swarm_size):
            position[i, :] = 4 * position[i - 1, :] * (1 - position[i - 1, :])
        # position = self.LB + position * (self.UB - self.LB)

        # VELOCITY, ENSURE STAYING WITHIN THE POSITION
        velocity = np.zeros((self.swarm_size, num_dims))
        velocity[0, :] = np.random.rand(1, num_dims) 
        for i in range(1, self.swarm_size):
            velocity[i, :] = 4 * velocity[i - 1, :] * (1 - velocity[i - 1, :])  
        velocity = (self.LB - position) + velocity * (self.UB - self.LB)
        # RESTRICT TO VMIN VMAX
        velocity = np.fmin(np.fmax(velocity, Vmin), Vmax)

        
        # OBTAIN FITNESS
        fitness, assigned = self.get_fitness(self.fitness, position, self.args, orig_bounds, self.normalize)
        # INITIAL LOCAL BEST, GLOBAL BEST
        self.lbest = position.copy()
        self.lfit[0, :] = fitness.copy()
        lassigned = assigned.copy()


        # INITIAL GBEST
        idx = np.argmin(self.lfit[0, :])
        self.gbest[0, :] = self.lbest[idx]
        self.gfit.append((self.lfit[0, :])[idx])
        self.assigned.append(lassigned[idx])
        tiled_gbest = np.tile(self.gbest[0, :], (self.swarm_size, 1))

        if type(self.params["w"]) is tuple:
            w = max(self.params["w"])
        else:
            w = self.params["w"]
        if type(self.params["c1"]) is tuple:
            c1 = max(self.params["c1"])
        else: 
            c1 = self.params["c1"]
        if type(self.params["c2"]) is tuple:
            c2 = min(self.params["c2"])
        else:
            c2 = self.params["c2"]
        
        for it in range(1, self.epochs):
            velocity = w * velocity + np.random.random() * c1 * (self.lbest - position) + np.random.random() * c2 * (tiled_gbest - position)
            velocity = np.fmin(np.fmax(velocity, Vmin), Vmax)
            
            # CHECK POSITION NEW
            position_tmp = position + velocity
            # out = np.logical_not((position_tmp > self.LB) * (position_tmp < self.UB))

            # VELOCITY CONFINEMENT
            # vel_conf = self.random_back_conf(velocity)
            # velocity = np.where(out, vel_conf, velocity)
            # position += velocity

            # CONFINE POSITION
            # position = np.fmin(np.fmax(position, self.LB), self.UB)

            # UPDATE FITNESS
            fitness, assigned = self.get_fitness(self.fitness, position, self.args, orig_bounds, self.normalize)

            # UPDATE BEST POSITION
            better = fitness < self.lfit[it - 1, :]
            self.lbest[better, :] = position[better, :]
            self.lfit[it, :] = self.lfit[it - 1, :]
            self.lfit[it, better] = fitness[better]
            lassigned = lassigned
            lassigned[better] = assigned[better]


            # UPDATE GBEST
            idx = np.argmin(self.lfit[it, :])
            if self.lfit[it, idx] < self.gfit[it - 1]:
                self.gfit.append(self.lfit[it, idx])
                self.gbest[it, :] = self.lbest[idx, :]
                self.assigned.append(lassigned[idx])
            else:
                self.gfit.append(self.gfit[it - 1])
                self.gbest[it, :] = self.gbest[it - 1, :]
                self.assigned.append(self.assigned[it - 1])
            
            # UPDATE PARAMETERS
            if type(self.params["w"]) is tuple:
                wmin, wmax  = self.params["w"]
                w = wmax - it*(wmax - wmin)/self.epochs
            if type(self.params["c1"]) is tuple:
                cmin, cmax = self.params["c1"]
                c1 = cmax - it*(cmax - cmin)/self.epochs
            if type(self.params["c2"]) is tuple:
                cmin, cmax = self.params["c2"]
                c2 = cmin + it*(cmax - cmin)/self.epochs
            

        if self.normalize:
            self.gbest = orig_LB + self.gbest * (orig_UB - orig_LB) 
    
    def extract_answer(self):
        return self.gbest[-1], self.gfit[-1], self.assigned[-1]

    def get_fitness(self, f, position, args, orig_bounds, normalize):
        if normalize:
            orig_LB, orig_UB = orig_bounds
            position = orig_LB + position * (orig_UB - orig_LB)
        fitness, assigned = f(position, args)
        return fitness, assigned
    
    def random_back_conf(self, velocity):
        x, y = velocity.shape
        vel_conf = -np.random.rand(x, y) * velocity

        return vel_conf
        
