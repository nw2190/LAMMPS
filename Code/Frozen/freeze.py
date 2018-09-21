import os, argparse
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_folder):

    # Specify checkpoint to freeze
    checkpoint_dir = model_folder + "Checkpoints/"
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # Specify filename for frozen graph
    output_graph = model_folder + "frozen_model.pb"
    
    # Specify output node to compute
    #output_node_names = "VAE_Net/prediction"
    #output_node_names = "VAE_Net_Masked/masked_prediction"
    output_node_names = "VAE_Net_Masked/masked_prediction,VAE_Net_Masked/masked_y"
    
    # Clear devices
    clear_devices = True
    
    # Restore meta_data
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    
    # Restore graph
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # Initialize TensorFlow session
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # Remove output from convert operation
        save_stdout = sys.stdout
        sys.stdout = open('tmp_log.txt', 'w')
        
        # Convert variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

        # Restore standard output stream
        sys.stdout = save_stdout
        
        # Write graph to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        #print("%d ops in the final graph." % len(output_graph_def.node))
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../Model/", type=str, help="Model folder to export")
    args = parser.parse_args()
    
    freeze_graph(args.model_dir)
