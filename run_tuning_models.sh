#!/bin/bash

#bin=/nfs/raid88/u10/users/bmin/bet-rtx8000-100/apps/venv/py-ml-type-inf/bin/python
bin=/h/milod/tmp/bin/python

for lr in 1e-3 5e-4 1e-4 5e-5  
do
	for epochs in 10 20 30
	do
		for emb_dim in 128 256
		do
			for hidden_units_lstm in 64 128 256
			do
				for hidden_units_dense in 64 128 256
				do
					# logfile = log.train_names_model.$lr.$epochs.$emb_dim.$hidden_units_lstm.$hidden_units_dense
					echo "===== Run experiments with lr=${lr}, epochs=${epochs}, emb_dim=${emb_dim}, hidden_units_lstm=${hidden_units_lstm}, hidden_units_dense=${hidden_units_dense}"
					$bin train_names_models.py -lr $lr -epochs $epochs -emb_dim $emb_dim -hidden_units_lstm $hidden_units_lstm -hidden_units_dense $hidden_units_dense
				done
			done
		done
	done
done

