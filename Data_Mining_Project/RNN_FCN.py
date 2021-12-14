from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

# For RNN layer we can use LSTM or GRU

def RNN_FCN():
    x_input = Input(shape=(1, 12288))

    # Replace this line for changing between LSTM and GRU
    x = GRU(128)(x_input)
    x = Dropout(0.8)(x)
    
    # Dimension Reshuffling
    y = Permute((2, 1))(x_input)
    
    # First Conv Block
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
    
    # Second Conv Block
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
    
    #y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    #y = BatchNormalization()(y)
    #y = Activation('relu')(y)
    #y = squeeze_excite_block(y)
    
    
    # Third Conv Block
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    output = Dense(1, activation='sigmoid')(x)

    model = Model(x_input, output, name='GRU_FCN')
    model.summary()

    # add load model code here to fine-tune

    return model



def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


