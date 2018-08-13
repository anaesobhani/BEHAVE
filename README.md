# BEHAVE

-1python	All_OneRule_Ana.py	4		0			5	5	0			-1					-1			0.3
									
					1:agent		0:no_agent_at_the_end	k	j	0: random S_kj		optional				optional		optional
					2:naive		1:agent_at_the_end			1: manual S_kj		number_of_rows_with_plus		chosen_k		gamma_value 
					3:cynical									-1: random_number_of_rows_with_plus	-1: random_chosen_k
					4:hybrid							





python	All_MultiRule_Ana.py	5		5	5	0			0			-1					1		0				0.3
										
				5:agent		k	j	0: random S_kj		0: random sp_kj		optional				optional	optional 			optional
				6:hybrid			1: manual S_kj		1: manual sp_kj		number_of_rows_with_plus		mu_default= 1	0:random_chosen_stkn		gamma_value 		
														-1: random_number_of_rows_with_plus	mu_value	1:manual_chosen_stkn



**************fOR LOOP_mnl**************

python	All_MultiRule_Ana.py	6		5	5	0			0			0					5			0
										
				5:agent		k	j	0: random S_kj		0: random sp_kj		optional				optional		optional
				6:hybrid			1: manual S_kj		1: manual sp_kj		number_of_rows_with_plus		mu_default= 1		gamma_value 		
														-1: random_number_of_rows_with_plus	mu_value	



********All_MultiRule_Ana.csv file heather*************

	R1	R2	R4	…Rk	Mu	Gamma	Hj	Up_j	Action chosen	K	J	Order of actions
A1												
A2												
A…j												
B												


********All_MultiRule_Ana_MEW1_Loop_MNL.csv file heather*************	

A1R1 A1R2... .A2R1 A2R2....AjRk		B1...Bk		Mu	Gamma		H1 H2 ...Hj		Up1...Upj 		Err1...Errk	Upp1...Uppj	Uppj-choice	K 	J			
