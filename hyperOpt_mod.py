from Faraone_TF import run_MLP
import hyperopt
import tensorflow as tf
import re
import pickle


def objective(args):

	params = {}

	params['l1_reg'] = args['l1_reg']
	params['l2_reg'] = args['l2_reg']
	params['num_layers'] = int(args['num_layers'])
	params['layer_size'] = int(args['layer_size'])
	params['learning_rate'] = args['learning_rate']
	params['batch_size'] = args['batch_size']
	params['dropout_keep_probability'] = args['dropout_keep_probability']
	params['validation_window'] = args['validation_window']
	
	with tf.Graph().as_default():
		loss = run_MLP(params)
    
	return loss

#trials = hyperopt.Trials()
#trials = pickle.load(open("trial_obj.pkl", "rb"))

def optimize():

    save_trial = 1
    max_trials = 10

    space = {
        'l1_reg': hyperopt.hp.uniform('l1_reg', 0, 0.2),
        'l2_reg': hyperopt.hp.uniform('l2_reg', 0, 0.2),
        'learning_rate': hyperopt.hp.quniform('learning_rate', 0.001, 0.01,0.001),
        'num_layers': hyperopt.hp.quniform('num_layers', 1, 5, 1),
        'layer_size': hyperopt.hp.quniform('layer_size', 10, 2000, 10),
        'batch_size': hyperopt.hp.choice('batch_size', [32]),
        'dropout_keep_probability': hyperopt.hp.uniform('dropout_keep_probability', 0.1, 0.5),
        'validation_window': hyperopt.hp.choice('validation_window',[5])         
    }

    try:
        trials = pickle.load(open("trial_obj.pkl", "rb"))
        print("________Loading saved trials object__________")
        max_trials = len(trials.trials) + save_trial
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, save_trial))
    except:
        trials = hyperopt.Trials()



    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_trials)

    with open("trial_obj.pkl", "wb") as f:
        pickle.dump(trials, f)


    print(best_model)
    print("*"*150)
    print(hyperopt.space_eval(space, best_model))
    print("*"*150)
    f = open("trials.log","w")
    for i,tr in enumerate(trials.trials):
    	trail = tr['misc']['vals']
    	for key in trail.keys():
    		trail[key] = trail[key][0]
    	f.write("Trail no. : %i\n"%i)
    	f.write(str(hyperopt.space_eval(space, trail))+"\n")
    	f.write("Loss : "+str(tr['result']['loss'])+"\n")
    	f.write("*"*100+"\n")
    f.close()
    
while True:
    optimize()
