import argparse
import numpy as np
import tensorflow as tf
from convert_time import convert_time
from encoder import read_data, read_soln

import sys
import time

def load_graph(frozen_model_folder):

    frozen_graph_filename = frozen_model_folder + "frozen_model.pb"
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            producer_op_list=None
        )
    return graph



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_folder", default="../Model/", type=str, help="Model folder to export")
    parser.add_argument("--x_val", default="0.0", type=str, help="x-coordinate")
    parser.add_argument("--y_val", default="0.0", type=str, help="y-coordinate")
    #parser.add_argument("--frozen_model_filename", default="../Model/frozen_model_1.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    #graph = load_graph(args.frozen_model_filename)
    graph = load_graph(args.frozen_model_folder)

    # Display operators defined in graph
    #for op in graph.get_operations():
    #    print(op.name)

    # Define input and output nodes
    x = graph.get_tensor_by_name('prefix/Training_Data/x:0')
    y = graph.get_tensor_by_name('prefix/Training_Data/y:0')
    #y = graph.get_tensor_by_name('prefix/VAE_Net/prediction:0')
    y_masked = graph.get_tensor_by_name('prefix/VAE_Net_Masked/masked_prediction:0')

    # Define placeholders to store regularization parameters
    #drop = graph.get_tensor_by_name('prefix/Regularization/drop:0')
    #training = graph.get_tensor_by_name('prefix/Regularization/training:0')


    template = graph.get_tensor_by_name('prefix/Training_Data/template:0')

    template_file = '../Arrays/template.npy'
    template_array = np.load(template_file)
    template_array = np.array([template_array[:,:,0]])
    template_array = np.transpose(template_array,[1,2,0])

    
    # Specify number of plots
    #DATA_COUNT = 2325
    #DATA_COUNT= 2925
    DATA_COUNT= 25250


    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time
        #x_data = np.array( read_data(0,'../Setup/Data/') )
        #x_data = [ np.transpose(x_data, (1, 2, 0)) ]
        
        #SCALING = 10e3
        SCALING = 1.0/0.0056417417
        #x_data = np.zeros([1,2])
        #x_data = SCALING*np.array([[0.0,-0.015]])
        x_val = float(args.x_val)
        y_val = float(args.y_val)
        x_data = SCALING*np.array([[y_val,x_val]])
        y_data = np.expand_dims(template_array,0)
        true_soln = np.expand_dims(template_array,0)
        
        y_out = sess.run(y_masked, feed_dict={
            x: x_data,
            y: y_data,
            template: template_array
        })

        cv_indices = np.load('../cv_indices.npy')
        train_indices = np.concatenate( (cv_indices[0:0], cv_indices[1:5,:]) ).flatten()
        test_indices = cv_indices[0,:]

        """
        TEST = True
        if TEST:
            indices = test_indices
        else:
            indices = train_indices
        """

        print("\n [ Computing Training Loss ]\n")
        indices = train_indices
        step = 1
        times = np.zeros([len(indices)])
        errors = np.zeros([len(indices)])
        l1_errors = np.zeros([len(indices)])
        #for k in range(1,DATA_COUNT+1):
        for k in indices:
            #x_data = np.array( read_data(k,'../Setup/Data/') )
            #x_data = [ np.transpose(x_data, (1, 2, 0)) ]

            sys.stdout.write('\r   Progress:  {0:.2%}'.format(step/len(train_indices)))
            sys.stdout.flush()
            
            x_data = np.expand_dims(read_data(k), 0)
            true_soln = read_soln(k)[0,:,:]
                        
            start_time = time.time()
        
            y_out = sess.run(y_masked, feed_dict={
                x: x_data,
                y: y_data,
                template: template_array
            })

            end_time = time.time()
            time_elapsed = convert_time(end_time-start_time)
            #print('\nComputation Time:  '  + time_elapsed)

            prediction = y_out[0,:,:,:]
            soln = y_data[0,:,:,:]
            ext_indices = (soln == 0.0)
            z_tensor = 0.0*soln
            prediction[ext_indices] = 0.0
            #pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
            #pred_layered = np.transpose(pred_layered,(1,2,0))
            #Np.save('y_out',pred_layered)


            prediction = prediction[:,:,0]
            mse_error = 1.0/prediction.size*np.sum(np.sum(np.power(prediction-true_soln,2), axis=1), axis=0)
            l1_error = 1.0/prediction.size*np.sum(np.sum(np.abs(prediction-true_soln), axis=1), axis=0)
            #print('Data %d MSE Error:   %f' %(k,mse_error))

            times[step-1] = float(time_elapsed[:-1])
            errors[step-1] = mse_error
            l1_errors[step-1] = l1_error

            step += 1

        mean_comp_time_train = np.mean(times)
        mean_error_train = np.mean(errors)
        mean_l1_error_train = np.mean(l1_errors)


        print("\n\n\n [ Computing Validation Loss ]\n")
        indices = test_indices
        step = 1
        times = np.zeros([len(indices)])
        errors = np.zeros([len(indices)])
        l1_errors = np.zeros([len(indices)])
        #for k in range(1,DATA_COUNT+1):
        for k in indices:
            #x_data = np.array( read_data(k,'../Setup/Data/') )
            #x_data = [ np.transpose(x_data, (1, 2, 0)) ]

            sys.stdout.write('\r   Progress:  {0:.2%}'.format(step/len(test_indices)))
            sys.stdout.flush()
            
            x_data = np.expand_dims(read_data(k), 0)
            true_soln = read_soln(k)[0,:,:]
                        
            start_time = time.time()
        
            y_out = sess.run(y_masked, feed_dict={
                x: x_data,
                y: y_data,
                template: template_array
            })

            end_time = time.time()
            time_elapsed = convert_time(end_time-start_time)
            #print('\nComputation Time:  '  + time_elapsed)

            prediction = y_out[0,:,:,:]
            soln = y_data[0,:,:,:]
            ext_indices = (soln == 0.0)
            z_tensor = 0.0*soln
            prediction[ext_indices] = 0.0
            #pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
            #pred_layered = np.transpose(pred_layered,(1,2,0))
            #Np.save('y_out',pred_layered)


            prediction = prediction[:,:,0]
            mse_error = 1.0/prediction.size*np.sum(np.sum(np.power(prediction-true_soln,2), axis=1), axis=0)
            l1_error = 1.0/prediction.size*np.sum(np.sum(np.abs(prediction-true_soln), axis=1), axis=0)
            #print('Data %d MSE Error:   %f' %(k,mse_error))

            times[step-1] = float(time_elapsed[:-1])
            errors[step-1] = mse_error
            l1_errors[step-1] = l1_error
            step += 1

        mean_comp_time_test = np.mean(times)
        mean_error_test = np.mean(errors)
        mean_l1_error_test = np.mean(l1_errors)
        
    print('\n\n\nAVERAGE COMPUTATION TIME:\n')
    print('  %f s [Train]  /  %f s [Test]' %(mean_comp_time_train, mean_comp_time_test))
    print('\n\nAVERAGE MSE ERROR:\n')
    print('  %f [Train]  /  %f [Test]' %(mean_error_train, mean_error_test))
    print('\n\nAVERAGE L1 ERROR:\n')
    print('  %f [Train]  /  %f [Test]' %(mean_l1_error_train, mean_l1_error_test))
    print('\n')

