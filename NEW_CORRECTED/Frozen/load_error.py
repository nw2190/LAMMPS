import argparse
import numpy as np
import tensorflow as tf
import csv

from encoder import read_data

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_folder", default="../Model/", type=str, help="Model folder to export")
    parser.add_argument("--ID", default=1, type=int, help="ID of solution to plot")
    args = parser.parse_args()
    ID = args.ID


    data_directory = '../Data/'
    input_filename = data_directory + 'indenter_' + str(ID) +'.txt'
    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])
    data = np.array(indenter_data, dtype=np.float32)
    # Only use x,y coordinates
    data = data[0:2]
    print('\nPlotting Solution for:   (x,y)  = %f %f\n' %(data[1],data[0]))

    
    graph = load_graph(args.frozen_model_folder)
    x = graph.get_tensor_by_name('prefix/Training_Data/x:0')
    y = graph.get_tensor_by_name('prefix/Training_Data/y:0')
    y_masked = graph.get_tensor_by_name('prefix/VAE_Net_Masked/masked_y:0')
    pred_masked = graph.get_tensor_by_name('prefix/VAE_Net_Masked/masked_prediction:0')

    template = graph.get_tensor_by_name('prefix/Training_Data/template:0')
    template_file = '../Arrays/template.npy'
    template_array = np.load(template_file)
    template_array = np.array([template_array[:,:,0]])
    template_array = np.transpose(template_array,[1,2,0])

    
    with tf.Session(graph=graph) as sess:

        # Run initial session to remove graph loading time
        #x_data = np.array( read_data(0,'../Setup/Data/') )
        #x_data = [ np.transpose(x_data, (1, 2, 0)) ]

        #SCALING = 10e3
        SCALING = 1.0/0.0056417417
        #x_data = np.zeros([1,2])
        #x_data = SCALING*np.array([[0.0,-0.015]])
        x_val = data[1]
        y_val = data[0]
        x_data = SCALING*np.array([[y_val,x_val]])
        y_data = np.expand_dims(np.load('../Arrays/avg_damage_'+str(ID)+'.npy'),0)
        
        [y_out,pred_out] = sess.run([y_masked, pred_masked], feed_dict={
            x: x_data,
            y: y_data,
            template: template_array
        })
        
        

        #x_data = np.array( read_data(k,'../Setup/Data/') )
        #x_data = [ np.transpose(x_data, (1, 2, 0)) ]
            
        [y_out,pred_out] = sess.run([y_masked, pred_masked], feed_dict={
            x: x_data,
            y: y_data,
            template: template_array
        })

        prediction = pred_out[0,:,:,:]
        soln = y_out[0,:,:,:]
        ext_indices = (soln == 0.0)
        prediction[ext_indices] = 0.0
        pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
        pred_layered = np.transpose(pred_layered,(1,2,0))

        soln_layered = np.array([soln[:,:,0],soln[:,:,0],soln[:,:,0]])
        soln_layered = np.transpose(soln_layered,(1,2,0))


        error = np.abs(soln_layered-pred_layered)
        
        # Template storing locations of atoms before impact
        template = template_array
        template_tiled = np.array([template[:,:,0],template[:,:,0],template[:,:,0]])
        template_tiled = np.transpose(template_tiled,[1,2,0])

        print(template_tiled.shape)
        print(pred_layered.shape)
        # Find Interior Indices
        interior_indices = (template_tiled == 1.0)
        #zero_tensor = 0.0*soln_layered
        #eps_tensor = zero_tensor + 0.0001

        # Remove 'displaced' atoms from solution
        error[interior_indices * (error == 0.0)] = 0.0001


        np.save('y_error_out',error)
            
    print('\n')


    #soln = np.load('../Arrays/avg_damage_'+str(ID)+'.npy')
    #np.save('y_true_out',soln)
