import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, read_data, convert_time

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
            op_dict=None,
            producer_op_list=None
        )
    return graph


# Recover values and alpha mask corresponding to data image
def read_data(data_ID, data_directory, **keyword_parameters):
    data_label = 'data_' + str(data_ID)
    data_file = data_directory + data_label + '.npy'
    vals = np.load(data_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)

    vals_array = np.array([vals])
    return vals_array



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
    PLOT_COUNT = 1

    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time
        #x_data = np.array( read_data(0,'../Setup/Data/') )
        #x_data = [ np.transpose(x_data, (1, 2, 0)) ]

        SCALING = 10e3
        #x_data = np.zeros([1,2])
        #x_data = SCALING*np.array([[0.0,-0.015]])
        x_val = float(args.x_val)
        y_val = float(args.y_val)
        x_data = SCALING*np.array([[y_val,x_val]])
        y_data = np.expand_dims(template_array,0)
        
        y_out = sess.run(y_masked, feed_dict={
            x: x_data,
            y: y_data,
            template: template_array
        })
        
        
        for k in range(0,PLOT_COUNT):
            #x_data = np.array( read_data(k,'../Setup/Data/') )
            #x_data = [ np.transpose(x_data, (1, 2, 0)) ]
            
            start_time = time.time()
        
            y_out = sess.run(y_masked, feed_dict={
                x: x_data,
                y: y_data,
                template: template_array
            })

            end_time = time.time()

            time_elapsed = convert_time(end_time-start_time)
            
            print('\nComputation Time:  '  + time_elapsed)





            prediction = y_out[0,:,:,:]
            soln = y_data[0,:,:,:]
            ext_indices = (soln == 0.0)
            z_tensor = 0.0*soln
            prediction[ext_indices] = 0.0
            pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
            pred_layered = np.transpose(pred_layered,(1,2,0))
            
            np.save('y_out',pred_layered)
            
            #plot_prediction(k, y_out, Model=0, CV=args.CV, Rm_Outliers=False, Filter=True, Plot_Error=False)

    print('\n')
