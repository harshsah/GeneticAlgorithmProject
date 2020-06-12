"""Ant Colony Optimization"""

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv

ALPHA = 1 # alpha(pheromone factor) >= 0
BETA = 1  # beta(visibility factor) >= 1
PHEROMONE_EVAPORATION_RATE = 0.05
MAXIMUM_NUMBER_OF_ITERATIONS = 10000
NUMBER_OF_ANTS = 10
PARAMETER_Q = 1
Q_NOT = 0.4
VISUALIZATION = False
SCREEN_SIZE =512
ELITE_NUMBER = 1

class ACO:
    def __init__(self,
                 distance_matrix = [], 
                 points = [],
                 ants = NUMBER_OF_ANTS, 
                 iterations = MAXIMUM_NUMBER_OF_ITERATIONS, 
                 evaporation_rate = PHEROMONE_EVAPORATION_RATE, 
                 alpha = ALPHA, 
                 beta = BETA,
                 parameter_q = PARAMETER_Q,
                 q_not = Q_NOT,
                 visualization = VISUALIZATION
                ):
        if distance_matrix == []:
            if points == []:
                print("No points or distance supplied")
                return
            distance_matrix = self.generate_distance_matrix(points)
        self.distance_matrix = distance_matrix
        self.points = points
        self.ants = ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.parameter_q = parameter_q
        self.q_not = q_not
        self.visualization = visualization
        self.visibility = 1/distance_matrix
        self.initialize_pheromone()
        self.solution = self.start()
    
    def start(self):
        elite_path = None  #To store best path for each iteration
        overall_elite_path = (range(len(self.distance_matrix)),np.inf) #Just a placeholder for best path overall
        for i in range(self.iterations):
            colony_path = self.generate_colony_path()
            self.update_pheromone(colony_path[:,0], overall_elite_path[0])
            elite_path = self.find_shortest_path(colony_path)
            if overall_elite_path[1]>elite_path[1]:
                overall_elite_path = elite_path
            if i%100 == 0:
                print("Iteration ",i+1,": ", overall_elite_path[1])
                if self.visualization == True:
                    self.view_graph(colony_path,elite_path, overall_elite_path)
        if self.visualization == True:
            cv.waitKey(0)
            cv.destroyAllWindows()
        return overall_elite_path
            
    def view_graph(self, colony_path = None, elite_path = None, overall_elite_path = None, wait = 1):
        screen_size = SCREEN_SIZE
        edge =0.2
        points = np.copy(self.points)
        points = points.transpose()
        x = points[0]
        y = points[1]
        maxx = max(x)
        minx = min(x)
        maxy = max(y)
        miny = min(y)
        temp = max(maxx-minx, maxy-miny)
        scaling_factor = (screen_size)/temp
        x = x * (scaling_factor)
        y = y * (scaling_factor )
        x = x - scaling_factor * minx
        y = y - scaling_factor * miny
        x = x.astype(int)
        y = y.astype(int)
        image = np.zeros(shape=[SCREEN_SIZE,SCREEN_SIZE,3], dtype=np.uint8)
        cv.imshow("ss", image)
#         if colony_path != None:
#             density_matrix = self.generate_density_matrix(np.copy(self.pheromone))
#             for i in range(len(colony_path)):
#                 for j in range(len(colony_path[i][0])-1):
#                     green = int(density_matrix[colony_path[i][0][j]][colony_path[i][0][j+1]])
# #                     print(green)
#                     image = cv.line(image, 
#                                     (x[colony_path[i][0][j]], y[colony_path[i][0][j]]),
#                                     (x[colony_path[i][0][j+1]], y[colony_path[i][0][j+1]]),
#                                     color = (0,green,0),
#                                     thickness =5
#                                     )
        for j in range(len(overall_elite_path[0])-1):
            image = cv.line(image, 
                            (x[overall_elite_path[0][j]], y[overall_elite_path[0][j]]),
                            (x[overall_elite_path[0][j+1]], y[overall_elite_path[0][j+1]]),
                            color = (0,255,0),
                            thickness = 5
                            )
        for i in range(len(x)):
            cv.circle(image, (x[i],y[i]),7,(0,0,255),-1)
        cv.imshow("ss", image)
        cv.waitKey(wait)
        #cv.destroyAllWindows()
        
        
    def generate_density_matrix(self, pheromone):
        temp_max = np.amax(pheromone)
        temp_min = np.amin(pheromone)
        if temp_min == temp_max:
            pheromone = pheromone*0
        else:
            pheromone = (pheromone - temp_min) * 255 / (temp_max-temp_min)
        pheromone = pheromone.astype(int)
#         print(pheromone)
        return pheromone
    
    def find_shortest_path(self, path):
        temp = 0
        for i in range(len(path)):
            if path[i][1] < path [temp][1]:
                temp = i
        return np.copy(path[i])
            
    def update_pheromone(self, path, elite):
        self.pheromone = (1-self.evaporation_rate)* self.pheromone
        for i in range(len(path)):
            for j in range(len(path[i])-1):
                self.pheromone[path[i][j]][path[i][j+1]] += self.parameter_q / self.distance_matrix[path[i][j]][path[i][j+1]]
        for i in range(ELITE_NUMBER):
            for j in range(len(elite)-1):
                self.pheromone[elite[j]][elite[j+1]] += self.parameter_q / self.distance_matrix[elite[j]][elite[j+1]]
                
        
            
    def generate_colony_path(self):
        path = []
        for i in range(self.ants):
            temp = self.generate_individual_path()
            path.append([temp, self.find_distance_of_path(temp)])
        path = np.array(path)
        #print(path)
        return path
    
    def find_distance_of_path(self, path):
        distance = 0 
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i+1]]
        return distance
    
    def generate_individual_path(self):
        visited_city = []
        visited_city.append(0)
        next_city = 0
        for i in range(len(self.distance_matrix)-1):
            next_city = self.choose_city(visited_city, self.pheromone[next_city],  self.visibility[next_city])
            visited_city.append(next_city)
        visited_city.append(0)
        visited_city = np.array(visited_city)
        #print(visited_city)
        return visited_city
        
    
    def choose_city(self, visited_city, _visibility, _pheromone):
        visibility = np.copy(_visibility)
        pheromone = np.copy(_pheromone)
        visibility[list(visited_city)] = 0
        pheromone[list(visited_city)] = 0
        #print(pheromone, visibility)
        probability_list = ((pheromone) ** (self.alpha)) * (visibility ** (self.beta))
        #print("probability_list \n", probability_list)
        probability_list = probability_list/ probability_list.sum()
        if np.random.normal() < self.q_not:
            return np.random.choice(range(len(probability_list)), 1, p=probability_list)[0]
        return probability_list.argmax()
        
    def initialize_pheromone(self):
        #self.pheromone = np.random.rand(len(self.distance_matrix), len(self.distance_matrix[0]))
        self.pheromone = np.zeros([len(self.distance_matrix), len(self.distance_matrix[0])]) + 1 /(len(self.distance_matrix)**2)
    
    def generate_distance_matrix(self, points):
        matrix = np.zeros((len(points), len(points)))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = self.find_dist(points[i], points[j])
        return matrix
    
    def find_dist(self, A,B):
        return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def main():
    points = np.array([
        [1,(1+math.sqrt(2))],
        [1,-(1+math.sqrt(2))],
        [-1,(1+math.sqrt(2))],
        [-1,-(1+math.sqrt(2))],
        [(1+math.sqrt(2)),1],
        [0,0],
        [-2,2],
        [-0.5,0.5],
        [0.5,0.5],
        [0.5,-0.5],
        [(1+math.sqrt(2)),-1],
        [-(1+math.sqrt(2)),1],
        [-(1+math.sqrt(2)),-1],
    ])
    points = np.random.rand(100,2)*100 - 50
    points = np.zeros((100,2))
    for i in range(10):
        for j in range(10):
            points[i*10+j] = [i,j]
    #print(points)
    visualization = True
    sample = ACO(points=points,visualization=visualization)
    image = np.zeros(shape=[SCREEN_SIZE,SCREEN_SIZE,3], dtype=np.uint8)
    if visualization == True:
        cv.namedWindow("ss",cv.WINDOW_NORMAL)
        cv.imshow("ss",image)
        print(sample.solution)
        sample.view_graph(overall_elite_path=sample.solution, wait =0)
        cv.destroyAllWindows()
    
if __name__ == "__main__":
        main()
