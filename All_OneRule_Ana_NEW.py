# python agent.py (1:agent , 2:naive , 3:cynical, 4:hybrid) (0:no_agent_at_the_end , 1:agent_at_the_end) k j manual(0/1) number_of_rows_with_plus(-1 if don't want to specify) chosen_k(optional) gamma(optional)

import sys
import numpy as np
import random

agent_naive_cynical = int(sys.argv[1])
agent_after_naive_cynical = int(sys.argv[2])
k = int(sys.argv[3])
j = int(sys.argv[4])
m = int(sys.argv[5])
if sys.argv[5] == '1':
    manual = True
else:
    manual = False
number_of_plus = -1
try:
    number_of_plus = int(sys.argv[6])
except:
    pass
chosen_k = -2
try:
    chosen_k = int(sys.argv[7]) - 1
except:
    pass
gamma = -1
try:
    gamma = float(sys.argv[8])
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

print ('\n')
print ('S_kj matrix:')
for row in first_matrix:
	for action in row:
		print (translate[action], end=' ')
	print('')

	
##################### Computing S_kj matrix, # Vertical 0/+ & # cross_compliances 
S_kj_V=np.zeros((j))
S_kj_CC=np.zeros((j))

#Number of Supporting Rules
first_matrix_matrix = np.array(first_matrix)
for jjj in range(j):
	for element in first_matrix_matrix[:,jjj]:
		if element == 1:
			S_kj_V[jjj]=S_kj_V[jjj]+1
		elif element == 0:
			S_kj_V[jjj]=S_kj_V[jjj]+1
			
print ('')
print ('# Supporting Rules:')			
print(S_kj_V)


#Cross-Compliance
for jjj in range(j):
	for kkk in range(k):
		if first_matrix_matrix[kkk,jjj]==1:
			for element in first_matrix_matrix[kkk,:]:
				if element == 1:
					S_kj_CC[jjj]=S_kj_CC[jjj]+1
				elif element == 0:
					S_kj_CC[jjj]=S_kj_CC[jjj]+1
		if first_matrix_matrix[kkk,jjj]==0:
			for element in first_matrix_matrix[kkk,:]:
				if element == 1:
					S_kj_CC[jjj]=S_kj_CC[jjj]+1
				elif element == 0:
					S_kj_CC[jjj]=S_kj_CC[jjj]+1
						
					
			
print ('')
print ('# Cross-Compliance:')			
print(S_kj_CC)



if agent_naive_cynical in [2, 3]: # either naive or cynical
    first_matrix_T = np.array(first_matrix).T.tolist()
    C = []
    c_trn = []
    for j1 in range(j):
        for j2 in range(j1 + 1, j):
            C.append([])
            C[-1].append(first_matrix_T[j1])
            C[-1].append(first_matrix_T[j2])
            C[-1] = np.array(C[-1]).T.tolist()
    E_H = []
    for Cn in C:
        probability_matrix = []
        ################ step 3,2
        for row in Cn:
            probability_row = []
            for action in row:
                if action == 1:
                    probability_row.append(1.)
                elif action == -1:
                    probability_row.append(0.)
                else:
                    probability_row.append(1. / row.count(0))
            probability_matrix.append(probability_row)
        ################ step 3,3
        p_rk = 1. / k
        ################ step 3,4
        post_probability_matrix = []
        for row in probability_matrix:
            post_probability_row = []
            c = 0
            for prob in row:
                s_p = sum([p_rk * i[c] for i in probability_matrix])
                if s_p == 0:
                    post_probability = 0
                else:
                    post_probability = (p_rk * prob) / s_p
                post_probability_row.append(post_probability)
                c += 1
            post_probability_matrix.append(post_probability_row)
        ################ step 3,5
        h_action_matrix = []
        for col in map(list, zip(*post_probability_matrix)):
            h = 0
            for v in col:
                if v != 0:
                    h -= v * np.log10(v)
            h_action_matrix.append(h)
        ################ step 4
        if agent_naive_cynical == 2: # Naive
            E = 0
            for ha in range(2):
                E += np.mean(np.array(probability_matrix).T.tolist()[ha]) * h_action_matrix[ha]
        else: # cynical
            E = max(h_action_matrix)
        E_H.append(E)
    print ('\n')
    print ('E(Hj|Cn) smallest to largest:')
    # print (np.sort(E_H).tolist())
    for eh in np.sort(list(set(E_H))).tolist():
        counter = 0
        for j1 in range(j):
            for j2 in range(j1 + 1, j):
                if E_H[counter] == eh:
                    print ('C', [j1 + 1, j2 + 1], ' = ', eh)
                counter += 1

    min_E_h = np.argwhere(E_H == np.amin(E_H)).flatten().tolist()
    counter = 0
    print ('\n')
    if agent_naive_cynical == 2:
        print ('Naive onlooker chosen choice set (Cn):')
    else:
        print ('Cynical onlooker chosen choice set (Cn):')
    for j1 in range(j):
        for j2 in range(j1 + 1, j):
            if counter in min_E_h:
                print ([j1 + 1, j2 + 1])
                c_trn.append([j1 + 1, j2 + 1])
            counter += 1


if agent_naive_cynical in [1, 4] or agent_after_naive_cynical == 1: # agent
    print('\n')
    ################ step 2
    if agent_naive_cynical in [1, 4]:
        min_E_h = [first_matrix]
        c_trn = [range(1, j + 1)]
    c_c = 0
    for mh in min_E_h:
        if agent_after_naive_cynical == 1:
            first_matrix = C[mh]
            print ('Matrix coming from Naive or Cynical:')
            for row in first_matrix:
            	for action in row:
            		print (translate[action], end=' ')
            	print('')
        probability_matrix = []
        for row in first_matrix:
            probability_row = []
            for action in row:
                if action == 1:
                    probability_row.append(1.)
                elif action == -1:
                    probability_row.append(0.)
                else:
                    probability_row.append(1. / row.count(0))
            probability_matrix.append(probability_row)
        # print
        # print ('P(a_j|r_k)matrix:')
        # for row in probability_matrix:
        # 	#print (row)
        #     for prob in row:
        #         print (prob, end=' ')
        #     print('')
        # print ('\n')
        ################ step 3
        p_rk = 1. / k
        # print
        # print ('P(r_k) =', p_rk)
        # print ('\n')
        ################ step 4
        post_probability_matrix = []
        for row in probability_matrix:
            post_probability_row = []
            c = 0
            for prob in row:
                post_probability = (p_rk * prob) / sum([p_rk * i[c] for i in probability_matrix])
                post_probability_row.append(post_probability)
                c += 1
            post_probability_matrix.append(post_probability_row)
        # print
        # print ('P(r_k|a_j) matrix:')
        # for row in post_probability_matrix:
        #  	#print (row)
        # 	for prob in row:
        # 		print (prob, end=' ')
        # 	print('')
        ################ step 5
        h_action_matrix = []
        for col in map(list, zip(*post_probability_matrix)):
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

        ################ step 6
        if agent_naive_cynical == 4:
            # step 6
            if chosen_k == -2:
                chosen_k = np.random.randint(0, k)
                print('')
                print('Random Chosen k:', chosen_k + 1)
            # step 7
            if gamma == -1:
                gamma = np.random.uniform()
                print('')
                print('Random gamma:', gamma)
            # step 7.1
            sp = [(x != -1) * 1 for x in first_matrix[chosen_k]]
            # step 8
            Up = []
            h_min = min(h_action_matrix)
            h_max = max(h_action_matrix)
            for i in range(j):
                Up.append(((1 - gamma) * sp[i]) + (gamma * (h_action_matrix[i] - h_min) / (h_max - h_min)))
            print('')
            print('Uj:', Up)
            print('')
            aac_temp = np.argwhere(Up == np.amax(Up)).flatten().tolist()
            aac = []
            for a in aac_temp:
                aac.append(c_trn[c_c][a])
            print ('agent action choice:', aac)

        else:
            print('')
            aac_temp = np.argwhere(h_action_matrix == np.amax(h_action_matrix)).flatten().tolist()
            aac = []
            # print (c_trn, aac_temp)
            for a in aac_temp:
                aac.append(c_trn[c_c][a])
            print ('agent action choice:', aac)
        c_c += 1

#soc=len(acc) #size of choices based on max-entropy
print('')
print('')
#Checking chosen actions based on H_j vs supporting rules (S_kj_V)
vcorrect=0                   #number of actions that is both chosen by maximizing H_j and S_kj_V
vmismatch=[]
for element in aac:
	if S_kj_V[element-1]== max(S_kj_V):
		vcorrect=vcorrect+1
	else:
		vmismatch.append(element)
		
if vcorrect == len(aac):
	print("Check Supporting Rules with H_j: yes")
	print('')

else:
	print("Check Supporting Rules with H_j: No, the mismatched H_j action(s) is (are)",vmismatch)
	print('')



#Checking chosen actions based on H_j vs cross_compliances (S_kj_CC)
ccorrect=0                   #number of actions that is both chosen by maximizing H_j and S_kj_CC
cmismatch=[]
for element in aac:
	if S_kj_CC[element-1]== max(S_kj_CC):
		ccorrect=ccorrect+1
	else:
		cmismatch.append(element)
		
if ccorrect == len(aac):
	print("Check Cross-Compliance with H_j: yes")
	print('')

else:
	print("Check Cross-Compliance with H_j:No, the mismatched H_j action(s) is (are)",cmismatch)	
	print('')
	
		
###SAVE TO FILE###
out = open('agentOneRule.csv', 'a')
#print step 1
out.write('New Run\n')
out.write('k =  {} \n'.format(k))
out.write('j =  {} \n'.format(j))
out.write('strong rule =  {} \n'.format(number_of_plus))
out.write('manual =  {} \n'.format(m))

out.write('\nS_kj matrix\n')
for row in first_matrix:
	for a in row:
		out.write('%s, ' % translate[a])
	out.write('\n')

#print step 2
out.write('\nP(a_j|r_k)matrix\n')
for row in probability_matrix:
	for a in row:
		out.write('%s, ' % str(format(a, '.3f')))
	out.write('\n')

#print step 3
out.write('\nP(r_k) ={}\n'.format(p_rk))

#print step 4
out.write('\nP(r_k|a_j) matrix\n')
for row in post_probability_matrix:
	for a in row:
		out.write('%s, ' % str(format(a, '.3f')))
	out.write('\n')

#print step 5
out.write('\nH_j action matrix\n')
for a in h_action_matrix:
	out.write('%s, ' % str(format(a, '.3f')))
out.write('\n')

#S_kj_V
out.write('\nSupporting Rules\n')
for a in S_kj_V:
	out.write('%s, ' % str(format(a, '.3f')))
out.write('\n')

if vcorrect == len(aac):
	out.write('\nCheck Supporting Rules with H_j: yes\n')
	
else:
	out.write('\nCheck Supporting Rules with H_j:No, the mismatched H_j action(s) = {}\n'.format(vmismatch))	

#S_kj_CC
out.write('\nCross-Compliance\n')
for a in S_kj_CC:
	out.write('%s, ' % str(format(a, '.3f')))
out.write('\n')

if ccorrect == len(aac):
	out.write('\nCheck Cross-Compliance with H_j: yes\n')
	
else:
	out.write('\nCheck Cross-Compliance with H_j:No, the mismatched H_j action(s) = {}\n'.format(cmismatch))	
	



#print step 6
out.write('\nagent action choice ={}  \n'.format(np.argmax(h_action_matrix) + 1))
out.write('\n\n\n')

out.close()
