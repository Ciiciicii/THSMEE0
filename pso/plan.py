import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import cv2 as cv
import os
import numpy as np
from cipso import PSO
from scipy.interpolate import interp1d
import time

# 1 PIXEL = X DISTANCE IN CM
PDR = 1


# INPUT: image file, show image (default: FALSE) OUTPUT: bounds (w, h), obs
def map_from_image(file, max_height, showim=False):
    if os.path.exists(file):
        image = cv.imread(file)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        im_h, im_w = gray.shape
        _, thres = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        boxes = []

        for cnt in contours:
            # Base rectangles
            # in center, width, height
            rect = cv.minAreaRect(cnt)
            
            centers, wh, angle = rect
            w, h = wh
            rect = centers, (w + PDR * 30, h + PDR * 30), angle

            # convert to points of coordinates
            box = cv.boxPoints(rect)
            # turn to int
            box = np.int0(box)
            

            if showim:
                cv.drawContours(image, [box], 0, (255, 0, 0), 2)
            boxes.append(box)

        
        # cv.drawContours(image, contours, -1, (0,255,0), 3)
        if showim:
            plt.imshow(image)
            plt.show()

        return (im_w, im_h, max_height), boxes

def observe_map(U, bounds, obs, task):
    img = np.zeros((bounds[1], bounds[0], 1), np.uint8)

    for ob in obs:
        cv.drawContours(img, [ob], 0, (255, 0, 0), 2)
        

    plt.imshow(img, cmap="gray")
    for u in U:
        plt.scatter(u[0], u[1], c="#39D37F")
    plt.scatter(task[0], task[1])
    plt.show()

def fitness(position, args):
    uavs = args["uavs"]
    task = args["task"]
    obs = args["obs"]
    obz = args["obz"]

    num_dims = len(uavs[0])
    swarm_size, num_vars = position.shape
    wpts = num_vars // num_dims
    
    lengths = np.zeros((len(uavs), swarm_size))
    colls = np.zeros((len(uavs), swarm_size))

    for u, U in enumerate(uavs):
        coords = np.zeros((num_dims, swarm_size, wpts + 2))
        for dim in range(num_dims): 
            # create a matrix for start and goal, repeat value for number of agents
            startM = np.ones([swarm_size, 1]) * U[dim]
            goalM = np.ones([swarm_size, 1]) * task[dim]

            # list indices from the matrix X that match that dimension
            idx = [i for i in range(num_vars) if (i % num_dims == dim)]
            # merge start mat then X then goal
            coords[dim] = np.block([startM, position[:, idx], goalM])

        
        t = np.linspace(0, 1, wpts + 2)

        PATHS = np.zeros([num_dims, swarm_size, 100])
        print(t.shape, coords[dim].shape)
        
        for dim in range(num_dims):
            # interp1d creates a function given x and y values, for us to interpolate new x 
            CS = interp1d(t, coords[dim], axis=1, kind='cubic', assume_sorted=True)
            splice = np.linspace(0, 1, 100)
            PATHS[dim] = CS(splice)
        
        # COLLISIONS 
        colls[u, :] = collision_check(PATHS, obs)
        
        # PATH OPTIMALITY / LEMGTH
        path_diffs = np.zeros([num_dims, swarm_size, 99])
        # Get length difference between spline points 
        for dim in range(num_dims):
            path_diffs[dim] = np.diff(PATHS[dim], axis=1)

        # Get resultant length, example: sqrt(diffX^2 + diffY^2) then sum all
        L_sq = np.zeros([swarm_size, 99])
        for dim in range(num_dims):
            L_sq = L_sq + path_diffs[dim] ** 2

        L = np.sqrt(L_sq).sum(axis=1)
        lengths[u, :] = L

    fitness = lengths * np.exp(colls)
    assigned = np.argmin(fitness, axis=0)
    FIT = np.zeros((swarm_size, ))
    for i, a in enumerate(assigned):
        FIT[i] = fitness[a, i]
    # 1D ARRAY OF (SWARM_SIZE)
    return FIT, assigned
    
def path_generate(wp, assigned, args):
    task = args["task"] 
    start = args["uavs"][assigned]
    num_vars = wp.shape[0]
    
    X = [i for i in range(num_vars) if (i % 3 == 0)]
    Y = [i for i in range(num_vars) if (i % 3 == 1)]
    Z = [i for i in range(num_vars) if (i % 3 == 2)]

    cx = np.block([start[0], wp[X], task[0]])
    cy = np.block([start[1], wp[Y], task[1]])
    cz = np.block([start[2], wp[Z], task[2]])

    t = np.linspace(0, 1, num_vars//3 + 2)
    CSx = interp1d(t, [cx], axis=1, kind='cubic', assume_sorted=True)
    CSy = interp1d(t, [cy], axis=1, kind='cubic', assume_sorted=True)
    CSz = interp1d(t, [cz], axis=1, kind='cubic', assume_sorted=True)
    splice = np.linspace(0, 1, 100)
    px = CSx(splice)
    py = CSy(splice)
    pz = CSz(splice)
    return px[0], py[0], pz[0]

    
def collision_check(paths, obs):
    num_dims = len(paths)
    swarm_size, n_pts = paths[0].shape

    fitnessc = np.zeros(paths.shape[1])
    for ob in obs:
        X = ob[:, 0]
        Y = ob[:, 1]
        sX, bX = min(X), max(X)
        sY, bY = min(Y), max(Y)
        btab = np.logical_and(np.logical_and(sX < paths[0], paths[0] < bX), np.logical_and(sY < paths[1], paths[1] < bY))
        fit = np.where(btab, 5, 0)
        fitnessc += np.nanmean(fit, axis=1)
    return fitnessc


class PP: 
    # INPUTS: U - array of initial positions of UAVs, S - swarm size (particles), W - num_waypoints
    def __init__(self, uavs, task, bounds, obs, obz):
        self.uavs = uavs
        self.task = task
        self.bounds = bounds
        self.obs = obs
        self.obz = obz

        self.sol = []
        self.sol_obj = []

    def __repr__(self):
        return "Path planning object: \
            \nCoordinates as x, y, z; Origin at west, north, bottom \
            \n start = {} \
            \n goal = {} \
            \n bounds = {} \
            \n obs_count = {}" \
            .format(self.uavs, self.task, self.bounds, len(self.obs))
    def optimize(self, swarm_size, num_wpts, fitness, w=(0.4, 0.9), c1=(0.5, 3.5), c2=(0.5, 3.5), Vmax=(0.5, 0.1), epochs=1000, normalize=True):
        num_dims = len(bounds)

        # Number of variabes in a particle: num_uavs * dims * wpts
        num_vars = num_dims * num_wpts
        args = {
            "uavs": self.uavs,
            "task": self.task,
            "obs": self.obs,
            "obz": self.obz
        }
        params = {
            "w": w,
            "c1": c1,
            "c2": c2,
            "Vmax": Vmax
        }

        LB = np.zeros(num_vars)
        UB = np.zeros(num_vars)
        for v in range(num_vars):
            UB[v] = self.bounds[v % num_dims]

        pso_sol = PSO(fitness, LB, UB, swarm_size, epochs, args, params, normalize)
        gb, gf, ga = pso_sol.extract_answer()
        px, py, pz = path_generate(gb, ga, args)
        self.sol_obj = pso_sol
        self.sol = px, py, pz
        # print(gf, gb, ga)


    def plot_sol(self, file):
        fig, ax = plt.subplots()
        ax.set_xlim(0,1000)
        ax.set_ylim(600,0)
        image = cv.imread(file)
        plt.imshow(image)
        for i in self.uavs:
            plt.scatter(i[0], i[1], c="#E96151")
        
        plt.plot(self.sol[0], self.sol[1])
        plt.show()
    
    def plot_cost_hist(self):
        plt.plot(range(1000), self.sol_obj.gfit)
        plt.show()
    def plot_g_hist(self, file):
        fig, ax = plt.subplots()
        ax.set_xlim(0,1000)
        ax.set_ylim(600, 0)
        args = {
            "uavs": self.uavs,
            "task": self.task,
            "obs": self.obs,
            "obz": self.obz
        }
        ct = 0 
        image = cv.imread(file)
        plt.imshow(image)
        for i in self.uavs:
            plt.scatter(i[0], i[1], c="#E96151")
        
        for i in range(0, 1000, 50):
            px, py, pz = path_generate(self.sol_obj.gbest[i, :], self.sol_obj.assigned[i], args)
            plt.plot(px, py, '#296B73', alpha=0.4 + ct * 0.02)
            ct += 1
        plt.plot(self.sol[0], self.sol[1], "#0084FF")
        plt.show()


if __name__ == "__main__":
    # OBS BOUNDS HEIGHT = 400, OBS HEIGHT = 300
    bounds, obs = map_from_image('img/map1-01.png', 400, False)
    
    # POSITIONS, with (0,0,0) ESU coords
    uavs = [[48, 53, 0], [48, 548, 20], [952, 53, 10], [952, 548, 0]]
    
    task = [460, 236, 100]
    # observe_map(uavs, bounds, obs, task)
    
    
    obheight = 300
    plan = PP(uavs, task, bounds, obs, obheight)

    fitness = fitness
    S = 40
    W = 3
    w = (0.4, 0.9)
    c1 = (0.5, 3.5)
    c2 = (0.5, 3.5)
    Vmax = (0.5, 0.1)
    
    times = []
    for i in range(100):
        st = time.time()
        plan.optimize(S, W, fitness, w, c1, c2, Vmax) # FINAL WPTS, WHICH UAV, CURVE
        times.append(time.time() - st)

    print("Mean: {}, Standard Deviation: {}".format(np.mean(times), np.std(times)))
    """
    plan.plot_sol('img/map1-01.png')
    plan.plot_cost_hist()
    plan.plot_g_hist('img/map1-01.png')
    """



        