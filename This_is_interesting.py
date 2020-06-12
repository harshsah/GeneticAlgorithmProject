import cv2 as cv
import numpy as np
import scipy
from scipy.signal import convolve2d
import seaborn as sns
import matplotlib

SCREEN_SIZE = 30
OBSTACLES = [[ [20,20], [30,40]],
			 # [ [10,50], [20,80]],
			 # [ [70,10], [80,80]],
			 # [ [10,70], [80,80]],
			 [ [50,60], [90,70]]
			]
ANTS_NUMBER = 1000
ALPHA = 2
BETA = 2
PARAMETER_Q = 1
EVAPORATION_RATE = 0.01
ITERATIONS = 1000
crossover_matrix = [[1]]
crossover_matrix_size = 5
mu = 1
sigma = 1
WAIT = 1000


class define_map:
	def __init__(self, 
				 screen_size, 
				 rectangle_obstacle_position = []
				 ):
		self.screen_size = screen_size
		self.map = np.zeros((screen_size,screen_size)).astype(int)
		self.crossover_matrix = self.return_crossover_matrix(int(screen_size/crossover_matrix_size))
		if len(rectangle_obstacle_position) == 0:
			self.draw_obstacle()
		else:
			self.draw_rectangle_obstacles( rectangle_obstacle_position)

	def draw_rectangle_obstacles(self, position_pair):
		for pair in position_pair:
			x1,y1,x2,y2 = (pair.flatten() * self.screen_size /100).astype(int)
			self.map[x1:x2, y1:y2] = 1
		# print(self.map)
	def return_crossover_matrix(self, size):
		size_ = 2*size + 1
		x = scipy.stats.norm.pdf(np.arange(size_) - size)
		x = x.reshape((1,len(x)))
		x = np.matmul(x.transpose(),x)
		x = ( x/x.max() * self.screen_size).astype(int)
		# print(x)
		return x

class apithy():
	def __init__(self,
				 obstacles = OBSTACLES,
				 screen_size = SCREEN_SIZE,
				 start_point = [],
				 end_point = [],
				 alpha = ALPHA,
				 beta = BETA,
				 parameter_q = PARAMETER_Q,
				 n_ants = ANTS_NUMBER,
				 iteratons = ITERATIONS,
				 evaporation_rate = EVAPORATION_RATE
				 ):
		obs = np.array(obstacles)
		self.map = define_map(screen_size,obs)
		if len(start_point) == 0 :
			start_point = [0,0]
		if len(end_point) == 0:
			end_point = [100, 100]
		self.start_point = (np.array(start_point)*screen_size/100).astype(int)
		self.end_point = (np.array(end_point)*screen_size/100).astype(int)
		self.evaporation_rate = evaporation_rate
		self.n_ants = n_ants
		self.alpha = alpha
		self.beta = beta
		self.parameter_q = parameter_q
		self.iteratons = iteratons
		self.pheromone = self.generate_pheromone_matrix()
		self.heuristic = self.generate_heuristics_matrix()
		# self.weight = self.generate_weight_matrix()
		self.best_path = [np.zeros((self.map.screen_size,self.map.screen_size)).astype(int), np.inf]
		self.path_found = False
		self.window = cv.namedWindow('map', cv.WINDOW_NORMAL)
		self.start()
		self.print_map()
		cv.waitKey(100000)
		cv.destroyAllWindows()

	def start(self):
		traverse_matrix = []
		# print(self.path_found, self.pheromone, self.heuristic, self.weight, traverse_matrix, self.best_path,sep='\n')
		for i in (range(self.n_ants)):
			# print(self.pheromone)
			print(i+1)
			# self.print_map()
			if i%1 == 0:
				print(i+1)
				self.print_map()
			flag = False
			traverse_matrix = self.return_traverse_matrix()
			# print("ant")
			x_current, y_current = self.start_point
			j = 0
			length = 0
			path_followed = []
			path_followed.append(self.start_point+1)
			for j in range(self.iteratons):
				weight = self. generate_weight_matrix(x_current,y_current)
				# print(weight)
				x_offset, y_offset = self.choose_next_point(traverse_matrix[x_current:x_current+3,y_current:y_current+3].copy(),
															 weight
															 )
				if x_offset == 2:
					break
				length += np.sqrt(x_offset**2 + y_offset**2)
				(x_current, y_current) = (x_current+x_offset, y_current+y_offset)
				# print(x_current, y_current)
				path_followed.append([x_current+1, y_current+1])
				traverse_matrix[x_current+1, y_current+1] = j+2
				if (x_current, y_current) == tuple(self.end_point):
					flag = True
					print("FOUND")
					self.path_found = True
					break
			if flag == True:
				self.pheromone_update(traverse_matrix,length)
				traverse_matrix, length = self.trim_traverse_matrix(traverse_matrix, path_followed,j+2)
				if self.best_path[1] > length:
					self.best_path = self.return_best_path(traverse_matrix.copy(), length)
				self.pheromone_update(traverse_matrix,length)
				# self.print_map()
			else:
				self.pheromone_update(traverse_matrix,self.iteratons)
				traverse_matrix, length = self.trim_traverse_matrix(traverse_matrix, path_followed,j+2)
				self.pheromone_update(traverse_matrix,self.iteratons/10)
			print('j=',j)
			# self.print_map()
		# print(self.path_found, self.pheromone, self.heuristic, np.around(self.weight,4), traverse_matrix, self.best_path,
			  # (self.pheromone==1).astype(int).sum(),sep='\n')
		print(traverse_matrix,length, self.pheromone, sep='\n')
		# sns.heatmap(traverse_matrix, self.pheromone)

	def trim_traverse_matrix(self, traverse_matrix, path_followed, number):
		x_current, y_current = self.start_point + 1
		i = 0
		length = 0
		# print(traverse_matrix)
		while i<number-2:
			x_current, y_current = path_followed[i]
			matrix = np.copy(traverse_matrix[x_current-1:x_current+2, y_current-1:y_current+2])
			x, y = path_followed[i+1]
			# print(matrix,(matrix>matrix[1,1]).astype(int).sum())
			if (matrix>matrix[1,1]).astype(int).sum() > 1:
				matrix = matrix.flatten()
				choice = np.argmax(matrix)
				x = x_current + choice //3 - 1
				y = y_current + choice % 3 - 1
				choice = traverse_matrix[x, y]
				# print('hrer\n',choice)
				while i<choice-2:
					x, y = path_followed[i+1]
					traverse_matrix[x,y] = 0
					i+=1
				# print(traverse_matrix)
				# i-=1
			length += np.sqrt((x_current-x)**2 + (y_current-y)**2)
			i += 1
		return (traverse_matrix, length)

	def return_best_path(self, traverse_matrix, distance):
		traverse_matrix = traverse_matrix[1:self.map.screen_size+1,1:self.map.screen_size+1]
		traverse_matrix = (traverse_matrix > 0).astype(int) * traverse_matrix
		return [traverse_matrix, distance]

	def pheromone_update(self, matrix, length):
		traverse_matrix = matrix.copy()
		self.pheromone = (1-self.evaporation_rate) * self.pheromone
		traverse_matrix = traverse_matrix[1:self.map.screen_size+1, 1:self.map.screen_size+1]
		# print(self.map.crossover_matrix)
		self.pheromone += self.parameter_q/length/self.map.screen_size*convolve2d((traverse_matrix>0).astype(int), self.map.crossover_matrix,mode='same',boundary='fill')
		# print(self.pheromone)
		# self.weight = self.generate_weight_matrix()

	def choose_next_point(self, traverse_matrix, weight):
		weight = (traverse_matrix == 0 ).astype(int) * weight
		# print(weight, traverse_matrix)
		# print(weight)
		weight = weight.flatten()
		if weight.sum() == 0:
			weight = (traverse_matrix != -1).astype(int).flatten()
			weight = weight/weight.sum()
			choice = np.random.choice(a=range(len(weight)), size=1, p=weight)[0]
			while choice == 4:
				choice = np.random.choice(a=range(len(weight)), size=1, p=weight)[0]
			return (2,2)
		if weight.sum() == np.inf:
			choice = np.argmax(weight)
		else:
			weight = weight/ weight.sum()
			# print(len(weight))
			choice = np.random.choice(a=range(len(weight)), size=1, p=weight)[0]
		y = choice % 3
		x = choice //3
		# print(x-1,y-1)
		return (x-1,y-1)

	def return_traverse_matrix(self):
		traverse_matrix = np.zeros((self.map.screen_size+2,self.map.screen_size+2)).astype(np.int)
		x_current, y_current = self.start_point + 1
		traverse_matrix[1:self.map.screen_size+1, 1:self.map.screen_size+1] = self.map.map * (-1)
		traverse_matrix[x_current][y_current] = 1
		traverse_matrix[:,0] = -1
		traverse_matrix[:,self.map.screen_size+1] = -1
		traverse_matrix[0, :] = -1
		traverse_matrix[self.map.screen_size+1,:] = -1
		return traverse_matrix


	def print_map(self, wait = WAIT):
		screen_size = self.map.screen_size
		# print(self.map.map.sum())
		img = np.zeros(shape = [screen_size,screen_size,3], dtype = np.uint8)
		img[:,:,1] += self.map.map.astype(np.uint8)*100
		# img[:,:,0] += (self.pheromone/self.pheromone.max()*100).astype(np.uint8)
		img[:,:,0] += (np.log(self.pheromone/self.pheromone.max()+1)*100).astype(np.uint8)
		img[:,:,2] += ((self.best_path[0]>0)*200).astype(np.uint8)
		cv.circle(img,tuple(self.start_point),1,(255,255,255),-1)
		cv.circle(img,tuple(self.end_point),1,(0,255,255),-1)
		cv.imshow('map',img)
		cv.waitKey(wait)

	def generate_weight_matrix(self, x, y):
		matrix = np.zeros((3,3))
		# matrix = (self.pheromone[]**self.alpha)*(self.heuristic[]**self.beta)
		for i in range(3):
			for j in range(3):
				x_ = (x-1+i)%self.map.screen_size
				y_ = (y-1+j)%self.map.screen_size
				matrix[i][j] += (self.pheromone[x_][y_]**self.alpha)*(self.heuristic[x_][y_]**self.beta)
		return matrix

	def generate_pheromone_matrix(self):
		matrix = np.ones((self.map.screen_size,self.map.screen_size))
		# return matrix
		x, y = self.start_point
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				matrix[i][j] = np.sqrt((i-x)**2+(j-y)**2)
		return matrix/matrix.max()

	def generate_heuristics_matrix(self):
		matrix = np.ones((self.map.screen_size,self.map.screen_size))
		x,y = self.end_point
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				matrix[i][j] = 1/np.sqrt((i-x)**2+(j-y)**2)
		matrix[i][j] = 0.5
		return matrix




obj = apithy(start_point = [0,0],
			 end_point = [99,99]
			 )














# drawing = False 
# mode = True 
# ix,iy = -1,-1

# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),-1)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),-1)

# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()