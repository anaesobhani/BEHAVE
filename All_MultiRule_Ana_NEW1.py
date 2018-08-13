# python 56.py (5:agent_mutirule , 6:hybrid_mutirule) k j (0/1) number_of_rows_with_plus(-1 if don't want to specify) mu manual_chosen_stkn(0/1) gamma
#
import sys
import numpy as np
import random

agent_naive_cynical = int(sys.argv[1])
k = int(sys.argv[2])
j = int(sys.argv[3])
m = int(sys.argv[4])
if sys.argv[4] == '1':
	manual = True
	
else:
	manual = False
	
if sys.argv[5]=='0':
	sp_kj=False
else:
	sp_kj=True

number_of_plus = -1
try:
    number_of_plus = int(sys.argv[6])
except:
    pass
	

mu = 1
try:
    mu = float(sys.argv[7])
except:
    pass
manual_chosen_stkn = 0
try:
    manual_chosen_stkn = int(sys.argv[8])
except:
    pass
gamma = -1
try:
    gamma = float(sys.argv[9])
except:
    pass

translate = {}
translate[0] = '0'
translate[1] = '+'
translate[-1] = '-'
translate_1 = {}
translate_1['0'] = 0
translate_1['+'] = 1
translate_1['-'] = -1

################ step 1
good_to_go = False
if not manual:
    while not good_to_go:
        first_matrix = []
        i = 0
        if number_of_plus == -1:
            while i < k:
                random_actions = np.random.randint(-1, 2, j).tolist()
                if random_actions.count(1) >= 1: # there is at least one + in the actions
                    index_plus_in_row = [index for index, value in enumerate(random_actions) if value == 1]
                    index_plus_to_keep = np.random.randint(0, len(index_plus_in_row))
                    random_actions = j * [-1]
                    random_actions[index_plus_in_row[index_plus_to_keep]] = 1
                if random_actions.count(-1) == j: # all the actions are -
                    index_to_change = np.random.randint(0, j)
                    random_actions[index_to_change] = np.random.randint(0, 2)
                first_matrix.append(random_actions)
                i += 1
        else:
            random_rows = random.sample(range(k), number_of_plus)
            while i < k:
                if i in random_rows:
                    random_actions = j * [-1]
                    random_actions[np.random.randint(0, j)] = 1
                else:
                    random_actions = np.random.randint(-1, 1, j).tolist()
                    if random_actions.count(-1) == j: # all the actions are -
                        index_to_change = np.random.randint(0, j)
                        random_actions[index_to_change] = 0
                first_matrix.append(random_actions)
                i += 1
        c = 0
        for column in map(list, zip(*first_matrix)):
            if column.count(-1) == k:
                for e in range(k):
                    row_to_change = np.random.randint(0, k)
                    if first_matrix[row_to_change].count(1) == 0:
                        good_to_go = True
                        break
                first_matrix[row_to_change][c] = 0
            c += 1
else:
    first_matrix = []
    i = 0
    while i < k:
        raw_row = input('row ' + str(i + 1) + ': ')
        row = []
        for a in raw_row:
            row.append(translate_1[a])
        first_matrix.append(row)
        i += 1
if not sp_kj: 
	print ('\n')
	print ('S_kj matrix:')
	for row in first_matrix:
		for action in row:
			print (translate[action], end=' ')
		print('')
	
	

#step 2,	cancelling step 1
if sp_kj:
	sp = []
	i = 0
	while i < k:
		raw_row = input('row ' + str(i + 1) + ': ')
		row = []
		for a in raw_row:
			aa=int(a)
			aaa=aa/10.0
			row.append(aaa)
		sp.append(row)
		i += 1
	
# step 2
if not sp_kj:
	sp = []
	i=0
	while i < k: 
		row=[]
		for jj in range(j):
			if first_matrix[i][jj]==1:
				row.append(random.uniform(0,3))
			elif first_matrix[i][jj]==0:   
				row.append(random.uniform(0,3))
			elif first_matrix[i][jj]==-1:
				row.append(0)	
		sp.append(row)
		i +=1 
#for i in range(k):
 #   sp.append([(x != -1) * 1 for x in first_matrix[i]])
#print ('\n')
#print ('Sp_kj matrix:')
#for row in sp:
#    for action in row:
#        print (action, end=' ')
#    print('')
print ('\n')
print ('Sp_kj matrix:')
for row in sp:
    for action in row:
        print (action, end=' ')
    print('')
	
	
	
	
	
# step 3
n = 2 ** k
st_t = []
i = n - 1
while i >= 0:
    binary_num = bin(i)[2:]
    binary_num = '0' * (k - len(binary_num)) + binary_num
    sti = []
    for l in binary_num:
        sti.append(int(l))
    st_t.append(sti)
    i -= 1
st = list(map(list, zip(*st_t)))
print ('\n')
print ('St_kn matrix:')
for row in st:
    for action in row:
        print (action, end=' ')
    print('')

# step 4
f_b = 1. / n
#print ('\n')
#print ('F(B):', f_b)

# step 5
stj = []
uj = []
for si in range(n):
    stj_t = []
    for i in range(len(st_t[si])):
        stj_t.append([x * st_t[si][i] for x in sp[i]])
    stj.append(stj_t)
    uj.append([mu * sum(x) for x in zip(*stj_t)])
#print ('\n')
#print ('Uj matrix:')
#rn = 1
#for row in uj:
#    print('St' + str(rn), end=' ')
#    for action in row:
#        print (action, end=' ')
#    print('')
#    rn += 1

# step 6
p_ab = []
for i in range(len(uj)):
    p_ab_t = []
    sum_exp = 0
    for l in range(j):
        sum_exp += np.exp(uj[i][l])
    for l in range(j):
        p_ab_t.append(np.exp(uj[i][l]) / sum_exp)
    p_ab.append(p_ab_t)
#print ('\n')
#print ('P(aj|B) matrix:')
#rn = 1
#for row in p_ab:
#    print('St' + str(rn), end=' ')
#    for action in row:
#        print (action, end=' ')
#    print('')
#    rn += 1

# step 7
f_bat = []
for i in range(j):
    f_ba_t = []
    sum_pab_fb = 0
    for l in range(len(p_ab)):
        sum_pab_fb += p_ab[l][i] * f_b
    for l in range(len(p_ab)):
        f_ba_t.append(p_ab[l][i] * f_b / sum_pab_fb)
    f_bat.append(f_ba_t)
f_ba = list(map(list, zip(*f_bat)))
#print ('\n')
#print ('f(b|aj) matrix:')
#rn = 1
#for row in f_ba:
#    print('St' + str(rn), end=' ')
#    for action in row:
#        print (action, end=' ')
#    print('')
#    rn += 1

# step 8
h_action_matrix = []
for col in f_bat:
    h = 0
    for v in col:
        if v != 0:
            h -= v * np.log10(v)
    h_action_matrix.append(h)
print('')
print ('H_j action matrix:')
for h in h_action_matrix:
    print (h, end=' ')
print('')

# step 5.9
if agent_naive_cynical == 5:
    aac_temp = np.argwhere(h_action_matrix == np.amax(h_action_matrix)).flatten().tolist()
    aac = []
    for a in aac_temp:
        aac.append(range(1, j + 1)[a])
    print('')
    print ('agent action choice:', aac)

# step 6.9
if agent_naive_cynical == 6:
    if manual_chosen_stkn:
        chosen_stkn = int(input('Chosen St_kn? ')) - 1
    else:
        chosen_stkn = np.random.randint(0, n)
        print('')
        print('Chosen St_kn:', chosen_stkn + 1)
# step 6.10
    if gamma == -1:
        gamma = np.random.uniform()
    print('')
    print('gamma:', gamma)
# step 6.11
    Upj = []
    h_min = min(h_action_matrix)
    h_max = max(h_action_matrix)
    uj_min = min(uj[chosen_stkn])
    uj_max = max(uj[chosen_stkn])
    for i in range(j):
        Upj.append(((1 - gamma) * (uj[chosen_stkn][i] - uj_min) / (uj_max - uj_min)) + (gamma * (h_action_matrix[i] - h_min) / (h_max - h_min)))
    print('')
    print ('Upj matrix:')
    for h in Upj:
        print (h, end=' ')
    aac_temp = np.argwhere(Upj == np.amax(Upj)).flatten().tolist()
    aac = []
    for a in aac_temp:
        aac.append(range(1, j + 1)[a])
    print('')
    print ('agent action choice:', aac)

	
	
	
	
RESULT=np.zeros((j+1,k+8))

sp_kj_matrix=np.array(sp)
st_matrix=np.array(st)
h_action_matrix_matrix=np.array(h_action_matrix)
Upj_matrix=np.array(Upj)
h_action_matrix_matrix_reverse=np.zeros((j,1))
Upj_matrix_reverse=np.zeros((j,1))

for jj in range(j):
	h_action_matrix_matrix_reverse[jj,0]=h_action_matrix_matrix[jj]
	Upj_matrix_reverse[jj,0]=Upj_matrix[jj]

	
	
	
	


for kk in range(k):#sp_kj
	for jj in range(j):
		RESULT[jj,kk]=sp_kj_matrix[kk,jj]	

for kk in range(k):#beta
	RESULT[j,kk]=st_matrix[kk,chosen_stkn]

for jj in range(j+1):#mu
	RESULT[jj,k]=mu

for jj in range(j+1): #gamma
	RESULT[jj,k+1]= gamma
		
for jj in range(j): #Hj
	RESULT[jj,k+2]= h_action_matrix_matrix_reverse[jj,0]

for jj in range(j): #Upj
	RESULT[jj,k+3]= Upj_matrix_reverse[jj,0]
	
for aacc in aac: #choice	
	RESULT[aacc,k+4]=1

for jj in range(j+1): # k=rule & J=action
	RESULT[jj,k+5]=k
	RESULT[jj,k+6]=j

for jj in range(j): #actions' order numbering
	RESULT[jj,k+7]=jj+1	

	
RESULT[j,k+7]=1000	#idetify the Beta row
	
'''
print ('\n')
print ('RESULT:')
for row in RESULT:
    for action in row:
        print (action, end=' ')
    print('')	
'''
		
###SAVE TO FILE###
out = open('agentMultiRule.csv', 'a')


for row in RESULT:
	for a in row:
		out.write('%s, ' % format(a))
	out.write('\n')



out.close()



