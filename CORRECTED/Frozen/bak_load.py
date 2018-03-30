import argparse
import numpy as np
import tensorflow as tf
from reader_frozen import plot_prediction, read_data

def load_graph(frozen_graph_filename):

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
    parser.add_argument("--frozen_model_filename", default="../Model/frozen_model_1.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    # Display operators defined in graph
    #for op in graph.get_operations():
        #print(op.name)

    # Define input and output nodes
    x = graph.get_tensor_by_name('prefix/Training_Data/x:0')
    y = graph.get_tensor_by_name('prefix/VAE_Net/prediction:0')

    # Define placeholders to store regularization parameters
    drop = graph.get_tensor_by_name('prefix/Regularization/drop:0')
    training = graph.get_tensor_by_name('prefix/Regularization/training:0')

    # Specify number of plots
    PLOT_COUNT = 10

    for k in range(0,PLOT_COUNT):

        x_data = np.array( read_data(k,'../Setup/Data/') )
        x_data = [ np.transpose(x_data, (1, 2, 0)) ]

        with tf.Session(graph=graph) as sess:
            y_out = sess.run(y, feed_dict={
                x: x_data,
                drop: 0.0,
                training: False
            })
            
        plot_prediction(k, y_out, Model=0, CV=1, Rm_Outliers=False, Filter=True, Plot_Error=False)
