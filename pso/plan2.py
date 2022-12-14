import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import cv2 as cv
import os
import numpy as np
from cipso import PSO
from scipy.interpolate import interp1d
from scipy.special import comb
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
    img = np.zeros([bounds[1], bounds[0], 1], np.uint8)

    for ob in obs:
        cv.drawContours(img, [ob], 0, (255, 0, 0), 2)
        

    plt.imshow(img, cmap="gray")
    for u in U:
        plt.scatter(u[0], u[1], c="#39D37F")
    plt.scatter(task[0], task[1])
    plt.show()

def fitness(position, args, ns=50):
    uavs = args["uavs"]
    task = args["task"]
    obs = args["obs"]
    obz = args["obz"]

    num_dims = len(uavs[0])
    swarm_size, num_vars = position.shape
    wpts = num_vars // num_dims
    
    FITNESS_MAT = np.zeros([len(uavs), swarm_size])

    for u, UAV in enumerate(uavs):

        """
       
        s_mat = np.tile(UAV, (40,1))
        t_mat = np.tile(task, (40,1))
        position_mat = np.block([s_mat, position, t_mat])

        bezier(position_mat)
        """
        b_mat = np.zeros([swarm_size, ns * 3])
        for p, particle in enumerate(position):
            particle_with_start_end = np.block([np.array(UAV), particle, np.array(task)])
            # RESHAPE TO [X Y Z]; [X Y Z] ...
            particle_pts = np.reshape(particle_with_start_end, (wpts + 2, 3))

            x, y, z = particle_pts[:,0], particle_pts[:,1], particle_pts[:,2]
            bx, by, bz = bezier(particle_pts)
            b_mat[p, :] = np.block([bx, by, bz])
        

        FITNESS_MAT[u, :] = bezier_fitness(b_mat, obs)    

    
    ASSIGNED = np.argmin(FITNESS_MAT, axis=0)
    FITNESS = np.zeros(swarm_size)
    for i, UAV_num in enumerate(ASSIGNED):
        FITNESS[i] = FITNESS_MAT[UAV_num, i]
    return FITNESS, ASSIGNED

"""
def bezier(position, ns=50):

    mat = True
    if len(position.shape) == 2:
        lent = position.shape[1]
    else:
        lent = position.shape[0]
        mat = False

    n_wpts = lent // 3

    bezier = bezier_solve(position, n_wpts, mat)
    t = np.linspace(0, 1, ns)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, ns)])
    print(new_points.shape)
    return new_points[:,0], new_points[:,1], new_points[:,2]

def bezier_solve(position, n_wpts, mat): 
    if mat:
        separated = np.split(position, n_wpts, axis=1)
    else:
        separated = np.split(position, n_wpts)
    print(separated[0])
    n = n_wpts - 1
    return lambda t: sum(
        comb(n, i) * t ** 1 * (1 - t) ** (n - i) * separated[i]
        for i in range(n_wpts)
    )
    

"""
def bezier_fitness(bezier_pts, obs):
    x, y, z = np.split(bezier_pts, 3, axis=1)
    LENGTH = path_length(x, y, z)
    COLLISIONS = collision_check(x, y, z, obs)
    # HEIGHT_RES = height_check(z)
    return LENGTH  * np.exp(COLLISIONS)

    # 1D ARRAY OF (SWARM_SIZE)

def bezier(points, ns=50):
    bezier = bezier_solve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, ns)])
    return new_points[:,0], new_points[:,1], new_points[:,2]
 

def bezier_solve(points):
    n = len(points) - 1
    return lambda t: sum(
        comb(n, i) * t**i * (1 - t)**(n - i) * points[i]
        for i in range(n + 1)
    )

def comb(n, i):
    return np.math.factorial(n) // (np.math.factorial(i) * np.math.factorial(n - i))

def path_length(x, y, z):
    return np.sqrt(np.sum(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2, axis=1))
    
def collision_check(x, y, z, obs):
    

    COLLISIONS = np.zeros(x.shape[0])
    M = 10

    for ob in obs:
        ob_X = ob[:, 0]
        ob_Y = ob[:, 1]
        within_X = np.logical_and(min(ob_X) <= x, x <= max(ob_X))
        within_Y = np.logical_and(min(ob_Y) <= y, y <= max(ob_Y))
        inside = np.logical_and(within_X, within_Y)
        collisions = inside.any(axis=1)
        
        collisions = np.where(collisions, M, 0)
        COLLISIONS = np.maximum(COLLISIONS, collisions)
        if np.mean(COLLISIONS) == M:
            break

    return COLLISIONS

def height_check(z):
    zmin, zmax = 190, 400
    HEIGHT_RES = np.zeros(z.shape[0])
    return HEIGHT_RES


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
        self.sol_obj = pso_sol
        gb, gf, ga = pso_sol.extract_answer()
        pos = np.block([np.array(self.uavs[ga]), np.array(gb), np.array(self.task)])
        path = bezier(np.reshape(pos, (num_wpts + 2,3)))
        self.sol = path

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

    def plot_sol_D(self):
        ax = plt.axes(projection='3d')
        plt.plot(self.sol[0], self.sol[1], self.sol[2], '-',  c="#006233")
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
        
        """
        for i in range(0, 1000, 50):
            px, py, pz = path_generate(self.sol_obj.gbest[i, :], self.sol_obj.assigned[i], args)
            plt.plot(px, py, '#296B73', alpha=0.4 + ct * 0.02)
            ct += 1
        plt.plot(self.sol[0], self.sol[1], "#0084FF")
        plt.show()
        """


if __name__ == "__main__":
    # OBS BOUNDS HEIGHT = 400, OBS HEIGHT = 300
    st = time.time()
    bounds, obs = map_from_image('img/map1-01.png', 400, False)
    # POSITIONS, with (0,0,0) ESU coords
    uavs = [[48, 53, 0], [48, 548, 20], [952, 53, 10], [952, 548, 0]]
    
    task = [460, 236, 200]
    # observe_map(uavs, bounds, obs, task)
    
    
    obheight = 300
    plan = PP(uavs, task, bounds, obs, obheight)

    fitness = fitness
    S = 40
    W = 5
    w = (0.4, 0.9)
    c1 = (0.5, 3.5)
    c2 = (0.5, 3.5)
    Vmax = (0.5, 0.1)
    
    plan.optimize(S, W, fitness, w, c1, c2, Vmax, epochs=500)
    
    print(time.time() - st)
    plan.plot_sol('img/map1-01.png')
    plan.plot_sol_D()

    # print("Mean: {}, Standard Deviation: {}".format(np.mean(times), np.std(times)))



        