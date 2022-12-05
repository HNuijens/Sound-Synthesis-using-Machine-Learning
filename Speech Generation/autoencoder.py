#!/usr/bin/env python
# coding: utf-8

# <h1 style = "font-size:3rem;color:darkcyan"> Train VAE model</h1>


# importing libraries
import numpy as np
import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.datasets import mnist

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# In[27]:


class VAE:
    
    def __init__(self, 
                input_shape,
                conv_filters,
                conv_kernels,
                conv_strides,
                latent_space_dim):
        
        self.input_shape = input_shape 
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.alpha = 1000000
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        
        self._build()
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, 
                           loss = self._calculate_combined_loss,
                           )
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, 
                      x_train,
                      batch_size = batch_size,
                      epochs = num_epochs,
                      shuffle = True)
        
    def save(self, save_folder='.'):
        self._create_folder_if_needed(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
    
    @classmethod
    def load(cls, save_folder='.'):
        parameters_path = os.path.join(save_folder, 'parameters.pkl')
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
        # make autoencoder object
        autoencoder = VAE(*parameters)
        # load weights
        weights_path = os.path.join(save_folder, 'weights.h5')
        autoencoder.model.load_weights(weights_path)
        return autoencoder

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_representations = self.decoder.predict(latent_representations)
        return reconstructed_representations, latent_representations
        
    def _calculate_combined_loss(self,y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.alpha * reconstruction_loss + kl_loss
        return combined_loss
    
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis = [1, 2, 3])
        return reconstruction_loss
    
    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = - 0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - 
                                K.exp(self.log_variance), axis = 1)
        return kl_loss
    
    def _load_weights(self, weights_path):
        self.model.load_weights(weights_path)
    
    def _create_folder_if_needed(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
    def _save_parameters(self, folder_name):
        parameters = [
            self.input_shape, 
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(folder_name, 'parameters.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(parameters, f)
            
    def _save_weights(self, folder_name):
        save_path = os.path.join(folder_name, 'weights.h5')
        self.model.save_weights(save_path)
        
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder() 
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name = 'encoder')
        
    def _add_encoder_input(self):
        return Input(shape = self.input_shape, name = 'encoder_input')
    
    def _add_conv_layers(self, encoder_input):
        layer_graph = encoder_input
        for i in range(self._num_conv_layers):
            layer_graph = self._add_conv_layer(i, layer_graph)
        return layer_graph
    
    def _add_conv_layer(self, layer_index, layer_graph):
        # conv2D + ReLu + batch normalization
        
        current_layer = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            name = f'encoder_conv_layer_{current_layer}'
        )
        
        layer_graph = conv_layer(layer_graph)
        layer_graph = ReLU(name=f'encoder_relu_{current_layer}')(layer_graph)
        layer_graph = BatchNormalization(name=f'encoder_bn_{current_layer}')(layer_graph)
        
        return layer_graph
    
    def _add_bottleneck(self, layer_graph): 
        # save shape for decoding
        self._shape_before_bottleneck = K.int_shape(layer_graph)[1:]
        
        # flatten data and add bottleneck with Gaussian sampling
        layer_graph = Flatten()(layer_graph)
        
        # create two branches of dense layers, one for the mean vector, one for log variance vector:
        self.mu = Dense(self.latent_space_dim, name = 'mu')(layer_graph)
        self.log_variance = Dense(self.latent_space_dim, 
                                  name = 'log_variance')(layer_graph)
        
        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape = K.shape(self.mu), 
                                      mean = 0., 
                                      stddev = 1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point
            
        # merge two layers (wrapping function in graph using Lambda)
        layer_graph = Lambda(sample_point_from_normal_distribution, 
                             name = 'encoder_output')([self.mu, self.log_variance])
        
        return layer_graph
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = 'decoder')

    def _add_decoder_input(self):
        return Input(shape = self.latent_space_dim, name = 'decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        return Dense(num_neurons, name = 'decoder_dense')(decoder_input)
       
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)
    
    def _add_conv_transpose_layers(self, layer_graph):
        for i in reversed(range(1, self._num_conv_layers)): # ignore input layer
            layer_graph = self._add_conv_transpose_layer(i, layer_graph)
        return layer_graph
    
    def _add_conv_transpose_layer(self, layer_index, layer_graph):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            name = f'decoder_conv_transpose_layer_{layer_num}'
        )
        
        layer_graph = conv_transpose_layer(layer_graph)
        layer_graph = ReLU(name = f'decoder_ReLU_{layer_num}')(layer_graph)
        layer_graph = BatchNormalization(name = f'decoder_bn_{layer_num}')(layer_graph)
        
        return layer_graph
    
    def _add_decoder_output(self, layer_graph):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = 'same',
            name = f'decoder_conv_transpose_layer_{self._num_conv_layers}'
        ) 
        
        layer_graph = conv_transpose_layer(layer_graph)
        output_layer = Activation('sigmoid', name = 'output_sigmoid_layer')(layer_graph)
        return output_layer
        
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name = 'autoencoder')
            


