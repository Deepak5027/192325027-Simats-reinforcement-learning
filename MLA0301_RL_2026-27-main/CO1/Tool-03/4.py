import numpy as np

grid = 4
gamma = 0.9

V = np.zeros((grid,grid))

goal = (3,3)

for iteration in range(100):
    newV = V.copy()

    for i in range(grid):
        for j in range(grid):

            if (i,j)==goal:
                continue

            values=[]

            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:

                ni=max(0,min(grid-1,i+dx))
                nj=max(0,min(grid-1,j+dy))

                reward=-1

                values.append(reward+gamma*V[ni,nj])

            newV[i,j]=max(values)

    V=newV

print(V)