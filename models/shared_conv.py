def create_shared_dcnn_network():
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate

    # Input layers
    visible_U_1 = Input(shape=(2048, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(2048, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(2048, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(2048, 38, 1), name='V_Wire_2')

    # U-wire shared layers
    shared_conv_1_U = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_U')
    shared_pooling_1_U = MaxPooling2D(pool_size=(5, 3), name='Shared_2_U')
    shared_conv_2_U = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_U')
    shared_pooling_2_U = MaxPooling2D(pool_size=(3, 1), name='Shared_4_U')
    shared_conv_3_U = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_U')
    shared_pooling_3_U = MaxPooling2D(pool_size=(3, 1), name='Shared_6_U')
    shared_conv_4_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_U')
    shared_pooling_4_U = MaxPooling2D(pool_size=(3, 1), name='Shared_8_U')
    # shared_conv_5_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_9_U')

    # V-wire shared layers
    shared_conv_1_V = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_V')
    shared_pooling_1_V = MaxPooling2D(pool_size=(5, 3), name='Shared_2_V')
    shared_conv_2_V = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_V')
    shared_pooling_2_V = MaxPooling2D(pool_size=(3, 1), name='Shared_4_V')
    shared_conv_3_V = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_V')
    shared_pooling_3_V = MaxPooling2D(pool_size=(3, 1), name='Shared_6_V')
    shared_conv_4_V = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_V')
    shared_pooling_4_V = MaxPooling2D(pool_size=(3, 1), name='Shared_8_V')
    # shared_conv_5_V = Conv2D(128, kernel_size=3, activation='relu', name='Shared_9_V')

    # U-wire feature layers
    encoded_1_U_1 = shared_conv_1_U(visible_U_1)
    encoded_1_U_2 = shared_conv_1_U(visible_U_2)
    pooled_1_U_1 = shared_pooling_1_U(encoded_1_U_1)
    pooled_1_U_2 = shared_pooling_1_U(encoded_1_U_2)

    encoded_2_U_1 = shared_conv_2_U(pooled_1_U_1)
    encoded_2_U_2 = shared_conv_2_U(pooled_1_U_2)
    pooled_2_U_1 = shared_pooling_2_U(encoded_2_U_1)
    pooled_2_U_2 = shared_pooling_2_U(encoded_2_U_2)

    encoded_3_U_1 = shared_conv_3_U(pooled_2_U_1)
    encoded_3_U_2 = shared_conv_3_U(pooled_2_U_2)
    pooled_3_U_1 = shared_pooling_3_U(encoded_3_U_1)
    pooled_3_U_2 = shared_pooling_3_U(encoded_3_U_2)

    encoded_4_U_1 = shared_conv_4_U(pooled_3_U_1)
    encoded_4_U_2 = shared_conv_4_U(pooled_3_U_2)
    pooled_4_U_1 = shared_pooling_4_U(encoded_4_U_1)
    pooled_4_U_2 = shared_pooling_4_U(encoded_4_U_2)

    # encoded_5_U_1 = shared_conv_5_U(pooled_4_U_1)
    # encoded_5_U_2 = shared_conv_5_U(pooled_4_U_2)

    # V-wire feature layers
    encoded_1_V_1 = shared_conv_1_V(visible_V_1)
    encoded_1_V_2 = shared_conv_1_V(visible_V_2)
    pooled_1_V_1 = shared_pooling_1_V(encoded_1_V_1)
    pooled_1_V_2 = shared_pooling_1_V(encoded_1_V_2)

    encoded_2_V_1 = shared_conv_2_V(pooled_1_V_1)
    encoded_2_V_2 = shared_conv_2_V(pooled_1_V_2)
    pooled_2_V_1 = shared_pooling_2_V(encoded_2_V_1)
    pooled_2_V_2 = shared_pooling_2_V(encoded_2_V_2)

    encoded_3_V_1 = shared_conv_3_V(pooled_2_V_1)
    encoded_3_V_2 = shared_conv_3_V(pooled_2_V_2)
    pooled_3_V_1 = shared_pooling_3_V(encoded_3_V_1)
    pooled_3_V_2 = shared_pooling_3_V(encoded_3_V_2)

    encoded_4_V_1 = shared_conv_4_V(pooled_3_V_1)
    encoded_4_V_2 = shared_conv_4_V(pooled_3_V_2)
    pooled_4_V_1 = shared_pooling_4_V(encoded_4_V_1)
    pooled_4_V_2 = shared_pooling_4_V(encoded_4_V_2)

    # encoded_5_V_1 = shared_conv_5_V(pooled_4_V_1)
    # encoded_5_V_2 = shared_conv_5_V(pooled_4_V_2)

    # Merge U- and V-wire of TPC 1 and TPC 2
    merge_TPC_1 = concatenate([pooled_4_U_1, pooled_4_V_1], name='TPC_1')
    merge_TPC_2 = concatenate([pooled_4_U_2, pooled_4_V_2], name='TPC_2')

    # Flatten
    flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
    flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

    # Shared Dense Layers
    shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2')
    shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2')

    # Dense Layers
    dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

    dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

    # Merge Dense Layers
    merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Flatten
    # flat_TPC_1_and_2 = Flatten(name='TPCs')(merge_TPC_1_2)

    # Output
    output_xyze = Dense(4, activation='relu', name='Output_xyze')(merge_TPC_1_2)
    #output_TPC = Dense(1, activation='sigmoid', name='Output_TPC')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])    #outputs=[output_xyze, output_TPC])
