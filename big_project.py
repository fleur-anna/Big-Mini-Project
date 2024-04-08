import numpy as np
import sys
#!{sys.executable} -m pip install opencv-python
import cv2
import random
import matplotlib.pyplot as plt

'''
#define a function that takes velocity and position and finds how long it will take for the particle to hit the next wall
def wall(s, v):
    s1 = s[0]
    s2 = s[1] 
    v1 = v[0]
    v2 = v[1] 
    
    t1 = (50-s1)/v1
    t2 = (50-s2)/v2
    t3 = (550-s1)/v1
    t4 = (550-s2)/v2
    
    t_vals = [t1,t2,t3,t4]
    t_vals2 = []
    
    for i in t_vals:
        if i > 0:
            t_vals2.append(i)
    
    return min(t_vals2)
    

#define a function that returns the velocity and position of particle at time t, given the initial velocity and position
def particle(s, v, t):
    
    wall_hits = []
    
    #find out if the time given is greater than the time taken to hit first wall
    #if it is, return the position 
    if t <= wall(s,v):
        return [s[0] + t*v[0], s[1] + t*v[1]], v
    
    else:
        #we know that t > wall(s,v)
        wall_hits.append(wall(s,v))
        
        while t > sum(wall_hits):
            
            #find the position when hits the coming wall
            s = (s[0]+v[0]*wall(s,v), s[1]+v[1]*wall(s,v))

            #find velocity after hitting the coming wall
            if s[0] == 50 or s[0] == 550:
                v[0] = -v[0] 

            if s[1] == 50 or s[1] == 550:
                v[1] = -v[1]
            
            #append time it takes to hit next wall
            wall_hits.append(wall(s,v))
        
        #if while loop fails then we need to consider the current s and v, which occurs before last appended time, so must delete this
        del wall_hits[-1]
        
        #need to find the difference between the given time and the time at which it hit previous wall
        time_diff = t - sum(wall_hits)
        
        #return the position andd velocity at given time
        return [s[0] + time_diff*v[0], s[1] + time_diff*v[1]], v
    
particle([1,1],[2,1], 3)

'''

#redo above but now considering the radius of the particle
def r_wall(s, v, r):
    

    #now the particle hits the wall at 0+r and 5-r
    t1 = (50+r-s[0])/v[0]
    t2 = (50+r-s[1])/v[1]
    t3 = (550-r-s[0])/v[0]
    t4 = (550-r-s[1])/v[1]
    
    t_vals = [t1,t2,t3,t4]
    t_vals2 = []
    
    for i in t_vals:
        if i > 0:
            t_vals2.append(i)
    
    return min(t_vals2)
    

#change the previous function to know use r_wall instead of wall
def r_particle(s, v, t, r):
    
    #the particle shouldn't be within r of the edge of the box, but if it is:
    if s[0] < 50 + r:
        dist1 = 50 + r - s[0]
        s[0] = s[0] + dist1

    if s[0] > 550 - r:
        dist2 = s[0] - (550 - r)
        s[0] = s[0] - dist2

    if s[1] < 50 + r:
        dist3 = 50 + r - s[1]
        s[1] = s[1] + dist3

    if s[1] > 550 - r:
        dist4 = s[1] - 550 + r
        s[1] = s[1] - dist4


    wall_hits = []
    
    if t <= r_wall(s,v,r):
        return [s[0] + t*v[0], s[1] + t*v[1]], v
    
    else:
        wall_hits.append(r_wall(s,v,r))
       
        while t > sum(wall_hits):
            
            s = (s[0]+v[0]*r_wall(s,v,r), s[1]+v[1]*r_wall(s,v,r))
            
            #the velocity changes when the radius hits the wall
            if s[0] == 50+r or s[0] == 550-r:
                v[0] = -v[0] 

            if s[1] == 50+r or s[1] == 550-r:
                v[1] = -v[1]
            
            wall_hits.append(r_wall(s,v,r))
        
        del wall_hits[-1]
        
        time_diff = t - sum(wall_hits)
        
        return [s[0] + time_diff*v[0], s[1] + time_diff*v[1]], v

#def generate1():
    #p_list = [[100, 300], [100, 300], [100, -100], [100,-100], [17.5, 17.5], [2, 2]]
    #return p_list
    

#generate a function that creates lists to describe all the particles where A is the number of particles of type A, B number of type B and C number of type C
def generate(A, B, C):
    
    #create a list of lists where lsists are x coordinate of position, y coordinate of position, x coordinate of velocity, y coordinate of velocity, radius, mass
    p_list = [[],[],[],[],[],[]]
    
    #while len x value of position is less than A then continue appending to list
    while len(p_list[0]) < A:
        
        #generate x and y position of position
        position_x = random.uniform(60, 540)
        position_y = random.uniform(60, 540)
        
        j = 0
        
        #check over the positions already generated
        for i in range (len(p_list[0])):
            
            #if the length between the generated position and a previously generated one is less than the sum of the radii then break the for loop
            if np.linalg.norm([position_x - p_list[0][i], position_y - p_list[1][i]]) <= 10 + p_list[4][i]:
                
                break
            
            #count how many times for loop runs
            j = j+1    
        
        #if j equals the range of loop then it completed it rather than break so the generated position is okay
        if j == len(p_list[0]):
            
            #append position to list
            p_list[0].append(position_x)
            p_list[1].append(position_y)
            
            #append the velocity, radius and mass at the same time
            p_list[2].append(random.uniform(-200,200))
            p_list[3].append(random.uniform(-200,200))
            p_list[4].append(10)
            p_list[5].append(1)
        
        #else the for loop wasnt completed and so need to retry generating position
        else:
            continue
            
            
    while len(p_list[0]) < A+B:
        
        position_x = random.uniform(67.5, 532.5)
        position_y = random.uniform(67.5, 532.5)
        
        j = 0
        
        for i in range (len(p_list[0])):
            
            if np.linalg.norm([position_x - p_list[0][i], position_y - p_list[1][i]]) <= 17.5 + p_list[4][i]:
                
                break
            
            j = j+1    
        
        if j == len(p_list[0]):
            
            p_list[0].append(position_x)
            p_list[1].append(position_y)
            
            p_list[2].append(random.uniform(-200,200))
            p_list[3].append(random.uniform(-200,200))
            p_list[4].append(17.5)
            p_list[5].append(2)
        
        else:
            continue
            
    while len(p_list[0]) < A+B+C:
        
        position_x = random.uniform(75, 525)
        position_y = random.uniform(75, 525)
        
        j = 0
        
        for i in range (len(p_list[0])):
            
            if np.linalg.norm([position_x - p_list[0][i], position_y - p_list[1][i]]) <= 25 + p_list[4][i]:
                
                break
            
            j = j+1    
        
        if j == len(p_list[0]):
            
            p_list[0].append(position_x)
            p_list[1].append(position_y)
            
            p_list[2].append(random.uniform(-200,200))
            p_list[3].append(random.uniform(-200,200))
            p_list[4].append(25)
            p_list[5].append(3)
        
        else:
            continue
            
    return p_list

#vector minus function
def minus(a, b):
    return [a[0] - b[0], a[1] - b[1]]


#cteate a function that takes p_list (so it has position, velocity and mass of particles) and the position of two particles in the list and returns their changed velocities
def collision(List, p1, p2):
    
    #compute the scalar    2m_2/(m_1 + m_2)      dot product of       (v1                         )-(v2                          ) and   (x1                          )- (x2                         ) divided by norm of         (x1                      )- (x2                      ) squared
    calc1 = (2*List[5][p2]/(List[5][p1] + List[5][p2]))*(np.dot(minus([List[2][p1],List[3][p1]],[List[2][p2],List[3][p2]]), minus([List[0][p1],List[1][p1]], [List[0][p2],List[1][p2]])))/((np.linalg.norm([minus([List[0][p1], List[1][p1]], [List[0][p2], List[1][p2]])]))**2)
    
    #v1 =      (v1                      )- ([calc1 times ((x1                      )- (x2                     ))[0], calc1 times ((x1                      )- (x2                      ))[1]])
    v1 = minus([List[2][p1], List[3][p1]], [calc1* minus([List[0][p1], List[1][p1]], [List[0][p2], List[1][p2]])[0], calc1* minus([List[0][p1], List[1][p1]], [List[0][p2], List[1][p2]])[1]])
    
    #compute the scalar    2m_1/(m_1 + m_2) times dot product of      (v2                         )-(v1                          ) and   (x2                          )- (x1                         ) divided by norm of         (x2                      )- (x1                      ) squared
    calc2 = (2*List[5][p1]/(List[5][p1] + List[5][p2]))*(np.dot(minus([List[2][p2],List[3][p2]],[List[2][p1],List[3][p1]]), minus([List[0][p2],List[1][p2]], [List[0][p1],List[1][p1]])))/((np.linalg.norm([minus([List[0][p2], List[1][p2]], [List[0][p1], List[1][p1]])]))**2)
    
    #v2 =      (v2                      )- ([calc2 times ((x2                      )- (x1                     ))[0], calc2 times ((x2                      )- (x1                      ))[1]])
    v2 = minus([List[2][p2], List[3][p2]], [calc2* minus([List[0][p2], List[1][p2]], [List[0][p1], List[1][p1]])[0], calc2* minus([List[0][p2], List[1][p2]], [List[0][p1], List[1][p1]])[1]])

    #we know that p1 and p2 are colliding so need to change their radius to the other
    if ((List[4][p1] == 10 and List[4][p2] == 17.5) or (List[4][p1] == 17.5 and List[4][p2] == 10)):
        List[4][p1] = 25
        List[4][p2] = 25

    if (List[4][p1] == 10 and List[4][p2] == 25) or (List[4][p1] == 25 and List[4][p2] == 10):
        List[4][p1] = 17.5
        List[4][p2] = 17.5

    if (List[4][p1] == 17.5 and List[4][p2] == 25) or (List[4][p1] == 25 and List[4][p2] == 17.5):
        List[4][p1] = 10
        List[4][p2] = 10

    #need to return the radius
    r1 = List[4][p1]
    r2 = List[4][p2]

    return v1, v2, r1, r2


#create a function that moves all particles forward after time t, then checks if any are colliding and changes the volocity if so
def move(p_list, t):
    
    #nevery particle needs to move forward
    for i in range (len(p_list[0])):
        
        #need to store initial position and velocity otherwise it will used already changed ones
        x_pos = p_list[0][i]
        y_pos = p_list[1][i]
        x_vel = p_list[2][i]
        y_vel = p_list[3][i]
        
        #change the values of velocity and position to be that after moving t forward
        p_list[0][i] = r_particle([x_pos, y_pos], [x_vel, y_vel], t, p_list[4][i])[0][0]
        p_list[1][i] = r_particle([x_pos, y_pos], [x_vel, y_vel], t, p_list[4][i])[0][1]
        p_list[2][i] = r_particle([x_pos, y_pos], [x_vel, y_vel], t, p_list[4][i])[1][0]
        p_list[3][i] = r_particle([x_pos, y_pos], [x_vel, y_vel], t, p_list[4][i])[1][1]
    
    Pairs = []

    #need to check every pair of positions
    for i in range (len(p_list[0])):
        for j in range (len(p_list[0])):
            
            #don't want to check one against itself
            if i != j and (j,i) not in Pairs:

                Pairs.append((i,j))
                
                #if they're less than sum of radii they are colliding so change their velocities
                if np.linalg.norm([p_list[0][i] - p_list[0][j], p_list[1][i] - p_list[1][j]]) <= p_list[4][i] + p_list[4][j]:
                   
                    #compute the new velocities of i and j respectfully
                    v1, v2, r1, r2 = collision(p_list, i, j)
                    
                    p_list[2][i] = v1[0]
                    p_list[3][i] = v1[1]
                    p_list[2][j] = v2[0]
                    p_list[3][j] = v2[1]
                    p_list[4][i] = r1
                    p_list[4][j] = r2

                    #there is a problem if we get two 25 radius particles because now their distance between centres is less than 50, so they are stuck on top of each other
                    if (r1 and r2) == 25:

                        #so we want to try and move them apart from each other 
                        s_1, v_1 = r_particle([p_list[0][i], p_list[1][i]], [p_list[2][i], p_list[3][i]], 50/np.linalg.norm(v1), 25)
                        p_list[0][i] = s_1[0]
                        p_list[1][i] = s_1[1]
                        p_list[2][i] = v_1[0]
                        p_list[3][i] = v_1[1]

                        s_2, v_2 = r_particle([p_list[0][j], p_list[1][j]], [p_list[2][j], p_list[3][j]], 50/np.linalg.norm(v2), 25)
                        p_list[0][j] = s_2[0]
                        p_list[1][j] = s_2[1]
                        p_list[2][j] = v_2[0]
                        p_list[3][j] = v_2[1]
                    
    return p_list

def draw():

    # create a 500x500 px image

    img = np.zeros((600,600,3),dtype='uint8')

    # set all pixels to white

    for i in range(600):
        for j in range(600):
            img[i][j]=(255,255,255)

    # draw box
            
    img = cv2.line(img, (50,50),(50,550),(0,0,0),2)
    img = cv2.line(img, (50,50),(550,50),(0,0,0),2)
    img = cv2.line(img, (50,550),(550,550),(0,0,0),2)
    img = cv2.line(img, (550,50),(550,550),(0,0,0),2)


    for i in range(len(p_list[0])):

        #A

        if p_list[4][i] == 10:
                                        #x position                     
            img = cv2.circle(img, (int(p_list[0][i]), int(p_list[1][i])), int(p_list[4][i]), (255, 122, 128) , -1)

        #B
        elif p_list[4][i] == 17.5:
                                        #x position                     
            img = cv2.circle(img, (int(p_list[0][i]), int(p_list[1][i])), int(p_list[4][i]), (128, 255, 128) , -1)


        #C
        elif p_list[4][i] == 25:
                                        #x position                     
            img = cv2.circle(img, (int(p_list[0][i]), int(p_list[1][i])), int(p_list[4][i]), (255, 128, 255) , -1)
 

    # add labels etc.

 

    #img = cv2.putText(img, "t = " + str(round(t,1)), (50,30), cv2.FONT_HERSHEY_COMPLEX,0.75, BLACK, 1, cv2.LINE_AA)
 

    # show and add to list for video processing

 

    vid.write(img)
    
    # cv2.imshow("MA3K7", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
                             


#generate particles: 4 type A, 3 type B, 2 type C
p_list = generate(4,1,1)



time_sum = 0

framerate = 1/0.05
vid = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),framerate, (600,600))

# find number of each particle

N_A = 0
N_B = 0
N_C = 0

for i in range(len(p_list[4])):
    if p_list[4][i] == 10:
        N_A += 1
    elif p_list[4][i] == 17.5:
        N_B += 1
    elif p_list[4][i] == 25:
        N_C += 1


file = open("log.txt", "w")
file.write(str(round(time_sum,2)) + " " + str(N_A) + " " + str(N_B) + " " + str(N_C) + " " + "\n")
file.close()

while time_sum <40:
    
    draw()
    p_list = move(p_list, 0.05)
    
    time_sum = time_sum + 0.05

    file = open("log.txt", "a")

    file.write(str(round(time_sum,2)) + " " + str(N_A) + " " + str(N_B) + " " + str(N_C) + " " + "\n")

    file.close()


print("video released")
vid.release()


# plot

file = open("log.txt", "r")
data = file.readlines()

time = []
A = []
B = []
C = []

for i in range(len(data)):

    if i%10 == 0:

        parsed = data[i].split(" ")

        time.append(float(parsed[0]))

        A.append(int(parsed[1]))

        B.append(int(parsed[2]))

        C.append(int(parsed[3]))

 



plt.plot(time, A, label = "A")

plt.plot(time, B, label = "B")

plt.plot(time, C, label = "C")

 

plt.locator_params(axis='y', nbins=6)

 

plt.xlabel("time")


 

plt.legend()

plt.show()

