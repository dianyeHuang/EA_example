import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 2           # 2 real numbers
POP_SIZE = 100
N_GENERATION = 200
N_REPRODUCE = 50
DNA_BOUND=[[13,100],[0,100]]
MUT_STRENGTH = 2.0

class EA:
  def __init__(self, DNA_size, POP_size):
    self.DNA_size = DNA_size
    self.POP_size = POP_size
    # initialize the population
    # self.POP = dict(DNA=80*np.random.rand(1,DNA_size).repeat(POP_size,axis=0),mut_strength=np.random.rand(POP_size,DNA_size)) # DNA+mut_strength
    self.POP = dict(DNA=80*np.random.rand(POP_size,DNA_size),mut_strength=np.random.rand(POP_size,DNA_size)) # DNA+mut_strength

  # fitness function
  def fun(self,x,y):
    return (x-10)**3+(y-20)**3

  def fitness(self,X,Y):
    pred = np.empty_like(X)
    i=0
    for x,y in zip(X,Y):
      if ((x-5)**2+(y-5)**2-100 < 0) or ((x-6)**2+(y-5)**2-82.81 < 0):
        pred[i] = 0
      elif x > 100 or x < 13:
        pred[i] = 0
      elif y < 0 or y > 100:
        pred[i] = 0
      else:
        pred[i] = np.exp(-((x-10)**3+(y-20)**3)/1000000)
      i += 1
    return pred

  def scalarfitness(self,x,y):
    if ((x-5)**2+(y-5)**2-100 < 0) or ((x-6)**2+(y-5)**2-82.81 < 0):
      pred = 0
    elif x > 100 or x < 13:
      pred = 0
    elif y < 0 or y > 100:
      pred = 0
    else:
      pred = np.exp(-((x-10)**3+(y-20)**3)/1000000)
    return pred
  
  def reproduce(self): 
    kids = {'DNA':np.empty((N_REPRODUCE, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
      # crossover(roughly half p1 and half p2)
      p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
      cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool) # crossover points
      kv[cp]  = self.POP['DNA'][p1, cp]
      kv[~cp] = self.POP['DNA'][p2, ~cp]
      ks[cp]  = self.POP['mut_strength'][p1, cp]
      ks[~cp] = self.POP['mut_strength'][p2, ~cp]

      # mutate (change DNA based on normal distribution)
      ks[:] = np.maximum(ks + (MUT_STRENGTH*np.random.rand(*ks.shape)-0.5), 0.)
      # ks[:] = ks + MUT_STRENGTH*np.random.rand(*ks.shape)
      # print (ks)
      #     # must > 0
      kv += ks * np.random.randn(*kv.shape)
      kv[0] = np.clip(kv[0], *DNA_BOUND[0])    # clip the mutated value
      kv[1] = np.clip(kv[1], *DNA_BOUND[1])   
      # print (kv)

    return kids
  
  def eliminate(self,kids): 
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        self.POP[key] = np.vstack((self.POP[key], kids[key]))

    fitness = self.fitness(self.POP['DNA'][:,0],self.POP['DNA'][:,1])            # calculate global fitness
    
    idx = np.arange(self.POP['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    print(fitness[fitness.argsort()[-1]])
    Point=self.POP['DNA'][fitness.argsort()[-1],:]
    minimum = -((Point[0]-10)**3+(Point[1]-20)**3)
    print('The optimal Point of this generation is: (%f,%f)\n\rThe minimum of f(x,y) is %f' % (Point[0],Point[1],minimum)),

    
    for key in ['DNA', 'mut_strength']:
        self.POP[key] = self.POP[key][good_idx]
    return self.POP


# main loop
ea = EA(DNA_SIZE, POP_SIZE)

# something about plotting (can be ignored)
num = 400
x = np.linspace(-20, 120, num)
y = np.linspace(-20, 120, num)
X, Y = np.meshgrid(x, y)
Z1 = np.zeros_like(X)
Z2 = np.zeros_like(X)
for i in range(num):
    for j in range(num):
        Z1[i, j] = ea.fun(x[i],y[j])
        Z2[i, j] = ea.scalarfitness(x[i],y[j])
        
# plt.figure(1)       
# plt.contourf(X, Y, Z1, 100, cmap=plt.cm.rainbow)
# plt.xlim(-20, 120); plt.ylim(-20, 120)
# plt.ion(); plt.colorbar(); plt.show(1)
plt.figure(2)
plt.contourf(Y, X, Z2, 100, cmap=plt.cm.rainbow)
plt.xlim(-20, 120); plt.ylim(-20, 120)
plt.ion(); plt.colorbar()


for g in range(N_GENERATION):  
  # plotting update
  if 'sca' in globals(): sca.remove()
  sca = plt.scatter(ea.POP['DNA'][:,0], ea.POP['DNA'][:,1], s=30, c='k',alpha=0.5);plt.pause(0.01)
  # update distribution parameters
  kids = ea.reproduce()
  ea.POP = ea.eliminate(kids)
  print ('No. %d generation' % g),


print('Finished'); plt.ioff(); plt.show(2)
