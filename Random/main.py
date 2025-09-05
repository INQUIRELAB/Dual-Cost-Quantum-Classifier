#Quantum classifier
#Journal title"Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions"
#arXiv Link: https://arxiv.org/pdf/2405.09377 (Accepted in Scientific Report)
#Sara Aminpour, Mike Banad, Sarah Sharif
#September 25th 2024

#School of Electrical and Computer Engineering/ Center for Quantum and Technology, University of Oklahoma, Norman, OK 73019 USA, 
###########################################################################


from big_functions import minimizer, painter, SGD_step_by_step_minimization, overlearning_paint
import datetime
qubits = 1  #integer, number of qubits
layers = 5 #integer, number of layers (time we reupload data)
chi = 'fidelity_chi' #Cost function; choose between ['fidelity_chi', 'trace_chi]
entanglement = 'y' #entanglement y/n
name = 'run' #However you want to name your files
seed = 30 #random seed
#epochs=3000 #number of epochs, only for SGD methods

        
problem=['circle', 'line'] #name of the problem, choose among ['circle', 'wavy circle', '3 circles', 'wavy lines', 'sphere', 'non convex', 'crown']
for problem in problem:
            
            method = ['l-bfgs-b', 'cobyla', 'nelder-mead', 'slsqp'] #minimization methods between ['l-bfgs-b', 'cobyla', 'nelder-mead', 'slsqp']
            for method in method:
                a=datetime.datetime.now()
                #SGD_step_by_step_minimization(problem, qubits, entanglement, layers, name)
                minimizer(chi, problem, qubits, entanglement, layers, method, name)
                painter(chi, problem, qubits, entanglement, layers, method, name, standard_test=True)
                #paint_world(chi, problem, qubits, entanglement, layers, method, name, standard_test=True)
                b=datetime.datetime.now()
                c=b-a
                
                text_file_nn = open('time.txt', mode='a+')
                text_file_nn.write(problem +'_'+ chi +'_'+ method +'_'+ str(qubits) +'Qubits_' + entanglement +'_'+ str(layers) +'Layers_' + method + "__" + 'total_time'+' = '+ str(c))
                text_file_nn.write('\n')
                text_file_nn.write('======================================================================')
                text_file_nn.write('\n')
                text_file_nn.close() 
