def create_shared_dcnn_network():
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras import regularizers

    regu = regularizers.l2(0.01)

    # Input layers
    visible_U_1 = Input(shape=(2048, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(2048, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(2048, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(2048, 38, 1), name='V_Wire_2')

    # Define U-wire shared layers
    shared_conv_1_U = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_U', kernel_regularizer=regu)
    shared_pooling_1_U = MaxPooling2D(pool_size=(5, 3), name='Shared_2_U')
    shared_conv_2_U = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_U', kernel_regularizer=regu)
    shared_pooling_2_U = MaxPooling2D(pool_size=(3, 1), name='Shared_4_U')
    shared_conv_3_U = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_U', kernel_regularizer=regu)
    shared_pooling_3_U = MaxPooling2D(pool_size=(3, 1), name='Shared_6_U')
    shared_conv_4_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_U', kernel_regularizer=regu)
    shared_pooling_4_U = MaxPooling2D(pool_size=(3, 1), name='Shared_8_U')
    # shared_conv_5_U = Conv2D(128, kernel_size=3, activation='relu', name='Shared_9_U')

    # Define V-wire shared layers
    shared_conv_1_V = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_1_V', kernel_regularizer=regu)
    shared_pooling_1_V = MaxPooling2D(pool_size=(5, 3), name='Shared_2_V')
    shared_conv_2_V = Conv2D(32, kernel_size=(5, 3), activation='relu', name='Shared_3_V', kernel_regularizer=regu)
    shared_pooling_2_V = MaxPooling2D(pool_size=(3, 1), name='Shared_4_V')
    shared_conv_3_V = Conv2D(64, kernel_size=3, activation='relu', name='Shared_5_V', kernel_regularizer=regu)
    shared_pooling_3_V = MaxPooling2D(pool_size=(3, 1), name='Shared_6_V')
    shared_conv_4_V = Conv2D(128, kernel_size=3, activation='relu', name='Shared_7_V', kernel_regularizer=regu)
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

    # Define shared Dense Layers
    shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

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
    # output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    # output_xyze = Dense(2, name='Output_xyze')(merge_TPC_1_2)
    output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)
    #output_TPC = Dense(1, activation='sigmoid', name='Output_TPC')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])    #outputs=[output_xyze, output_TPC])





def create_shared_DEEPcnn_network(var_targets, inputImages, multiplicity):
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras import regularizers
    from keras.layers import RepeatVector
    from keras.layers import LSTM
    from keras.layers import SimpleRNN
    from keras.layers import ELU
    from keras.layers import GlobalAveragePooling2D

    # regu = regularizers.l2(0.01)
    var_targets = var_targets
    # elu = ELU(alpha=1.0)

    # Input layers
    visible_U_1 = Input(shape=(400, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(400, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(400, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(400, 38, 1), name='V_Wire_2')

    # Define U-wire shared layers   # 16, 32, 64, 100, 200, 300, 400, 500
    shared_conv_1_U = Conv2D(8, kernel_size=5, activation='relu', name='Shared_1_U', padding='same')
    shared_conv_2_U = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_2_U', padding='same')
    shared_pooling_1_U = MaxPooling2D(pool_size=(2, 1), name='Shared_1p_U')
    shared_conv_3_U = Conv2D(32, kernel_size=(3, 1), activation='relu', name='Shared_3_U')
    shared_conv_4_U = Conv2D(50, kernel_size=(3, 1), activation='relu', name='Shared_4_U')
    shared_pooling_2_U = MaxPooling2D(pool_size=(3, 1), name='Shared_2p_U')
    shared_conv_5_U = Conv2D(100, kernel_size=(3, 1), activation='relu', name='Shared_5_U')
    shared_conv_6_U = Conv2D(150, kernel_size=(3, 1), activation='relu', name='Shared_6_U')
    shared_pooling_3_U = MaxPooling2D(pool_size=(3, 3), name='Shared_3p_U')
    shared_conv_7_U = Conv2D(200, kernel_size=(3, 1), activation='relu', name='Shared_7_U')
    shared_conv_8_U = Conv2D(250, kernel_size=(3, 1), activation='relu', name='Shared_8_U')
    shared_pooling_4_U = MaxPooling2D(pool_size=(4, 1), name='Shared_4p_U')

    # Define V-wire shared layers
    shared_conv_1_V = Conv2D(8, kernel_size=5, activation='relu', name='Shared_1_V', padding='same')
    shared_conv_2_V = Conv2D(16, kernel_size=(5, 3), activation='relu', name='Shared_2_V', padding='same')
    shared_pooling_1_V = MaxPooling2D(pool_size=(2, 1), name='Shared_1p_V')
    shared_conv_3_V = Conv2D(32, kernel_size=(3, 1), activation='relu', name='Shared_3_V')
    shared_conv_4_V = Conv2D(50, kernel_size=(3, 1), activation='relu', name='Shared_4_V')
    shared_pooling_2_V = MaxPooling2D(pool_size=(3, 1), name='Shared_2p_V')
    shared_conv_5_V = Conv2D(100, kernel_size=(3, 1), activation='relu', name='Shared_5_V')
    shared_conv_6_V = Conv2D(150, kernel_size=(3, 1), activation='relu', name='Shared_6_V')
    shared_pooling_3_V = MaxPooling2D(pool_size=(3, 3), name='Shared_3p_V')
    shared_conv_7_V = Conv2D(200, kernel_size=(3, 1), activation='relu', name='Shared_7_V')
    shared_conv_8_V = Conv2D(250, kernel_size=(3, 1), activation='relu', name='Shared_8_V')
    shared_pooling_4_V = MaxPooling2D(pool_size=(4, 1), name='Shared_4p_V')

    # U-wire feature layers
    if inputImages == 'U' or 'UV':
        encoded_1_U_1 = shared_conv_1_U(visible_U_1)
        encoded_1_U_2 = shared_conv_1_U(visible_U_2)

        encoded_2_U_1 = shared_conv_2_U(encoded_1_U_1)
        encoded_2_U_2 = shared_conv_2_U(encoded_1_U_2)

        pooled_1_U_1 = shared_pooling_1_U(encoded_2_U_1)
        pooled_1_U_2 = shared_pooling_1_U(encoded_2_U_2)

        pooled_1_U_1 = encoded_2_U_1
        pooled_1_U_2 = encoded_2_U_2

        encoded_3_U_1 = shared_conv_3_U(pooled_1_U_1)
        encoded_3_U_2 = shared_conv_3_U(pooled_1_U_2)

        encoded_4_U_1 = shared_conv_4_U(encoded_3_U_1)
        encoded_4_U_2 = shared_conv_4_U(encoded_3_U_2)

        pooled_2_U_1 = shared_pooling_2_U(encoded_4_U_1)
        pooled_2_U_2 = shared_pooling_2_U(encoded_4_U_2)

        # pooled_2_U_1 = encoded_4_U_1
        # pooled_2_U_2 = encoded_4_U_2

        encoded_5_U_1 = shared_conv_5_U(pooled_2_U_1)
        encoded_5_U_2 = shared_conv_5_U(pooled_2_U_2)

        encoded_6_U_1 = shared_conv_6_U(encoded_5_U_1)
        encoded_6_U_2 = shared_conv_6_U(encoded_5_U_2)

        pooled_3_U_1 = shared_pooling_3_U(encoded_6_U_1)
        pooled_3_U_2 = shared_pooling_3_U(encoded_6_U_2)

        pooled_3_U_1 = encoded_6_U_1
        pooled_3_U_2 = encoded_6_U_2

        encoded_7_U_1 = shared_conv_7_U(pooled_3_U_1)
        encoded_7_U_2 = shared_conv_7_U(pooled_3_U_2)

        encoded_8_U_1 = shared_conv_8_U(encoded_7_U_1)
        encoded_8_U_2 = shared_conv_8_U(encoded_7_U_2)

        pooled_4_U_1 = shared_pooling_4_U(encoded_8_U_1)
        pooled_4_U_2 = shared_pooling_4_U(encoded_8_U_2)

        pooled_4_U_1 = encoded_8_U_1
        pooled_4_U_2 = encoded_8_U_2


    # V-wire feature layers
    if inputImages == 'V' or 'UV':
        encoded_1_V_1 = shared_conv_1_V(visible_V_1)
        encoded_1_V_2 = shared_conv_1_V(visible_V_2)

        encoded_2_V_1 = shared_conv_2_V(encoded_1_V_1)
        encoded_2_V_2 = shared_conv_2_V(encoded_1_V_2)

        pooled_1_V_1 = shared_pooling_1_V(encoded_2_V_1)
        pooled_1_V_2 = shared_pooling_1_V(encoded_2_V_2)

        pooled_1_V_1 = encoded_2_V_1
        pooled_1_V_2 = encoded_2_V_2

        encoded_3_V_1 = shared_conv_3_V(pooled_1_V_1)
        encoded_3_V_2 = shared_conv_3_V(pooled_1_V_2)

        encoded_4_V_1 = shared_conv_4_V(encoded_3_V_1)
        encoded_4_V_2 = shared_conv_4_V(encoded_3_V_2)

        pooled_2_V_1 = shared_pooling_2_V(encoded_4_V_1)
        pooled_2_V_2 = shared_pooling_2_V(encoded_4_V_2)

        # pooled_2_V_1 = encoded_4_V_1
        # pooled_2_V_2 = encoded_4_V_2

        encoded_5_V_1 = shared_conv_5_V(pooled_2_V_1)
        encoded_5_V_2 = shared_conv_5_V(pooled_2_V_2)

        encoded_6_V_1 = shared_conv_6_V(encoded_5_V_1)
        encoded_6_V_2 = shared_conv_6_V(encoded_5_V_2)

        pooled_3_V_1 = shared_pooling_3_V(encoded_6_V_1)
        pooled_3_V_2 = shared_pooling_3_V(encoded_6_V_2)

        pooled_3_V_1 = encoded_6_V_1
        pooled_3_V_2 = encoded_6_V_2

        encoded_7_V_1 = shared_conv_7_V(pooled_3_V_1)
        encoded_7_V_2 = shared_conv_7_V(pooled_3_V_2)

        encoded_8_V_1 = shared_conv_8_V(encoded_7_V_1)
        encoded_8_V_2 = shared_conv_8_V(encoded_7_V_2)

        pooled_4_V_1 = shared_pooling_4_V(encoded_8_V_1)
        pooled_4_V_2 = shared_pooling_4_V(encoded_8_V_2)

        pooled_4_V_1 = encoded_8_V_1
        pooled_4_V_2 = encoded_8_V_2

    # Merge U- and V-wire of TPC 1 and TPC 2
    if inputImages == 'UV':

        # merge_TPC_1 = concatenate([encoded_8_U_1, encoded_8_V_1], name='TPC_1')
        # merge_TPC_2 = concatenate([encoded_8_U_2, encoded_8_V_2], name='TPC_2')

        merge_TPC_1 = concatenate([pooled_4_U_1, pooled_4_V_1], name='TPC_1')
        merge_TPC_2 = concatenate([pooled_4_U_2, pooled_4_V_2], name='TPC_2')

        # Flatten
        flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
        flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

        if multiplicity == 'SS':

            # Define shared Dense Layers
            shared_dense_1 = Dense(16, activation='relu', name='Shared_1_TPC_1_and_2')
            shared_dense_2 = Dense(8, activation='relu', name='Shared_2_TPC_1_and_2')

            # Dense Layers
            dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
            dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

            dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
            dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

            # Merge Dense Layers
            merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')


        if multiplicity == 'SS+MS':


            number_timesteps = 5

            # shared_conv_classification_1_U = Conv2D(1, kernel_size=5, activation='hard_sigmoid')
            # shared_conv_classification_2_U = Conv2D(16, kernel_size=(5, 3), activation='relu')
            # # shared_pooling_classification_U = MaxPooling2D(pool_size=(2, 1))
            # shared_conv_classification_1_V = Conv2D(1, kernel_size=5, activation='hard_sigmoid')
            # shared_conv_classification_2_V = Conv2D(16, kernel_size=(5, 3), activation='relu')
            #
            # U1 = shared_conv_classification_1_U(visible_U_1)
            # U2 = shared_conv_classification_1_U(visible_U_2)
            # # U1 = shared_conv_classification_2_U(U1)
            # # U2 = shared_conv_classification_2_U(U2)
            #
            # V1 = shared_conv_classification_1_V(visible_V_1)
            # V2 = shared_conv_classification_1_V(visible_V_2)
            # # V1 = shared_conv_classification_2_V(V1)
            # # V2 = shared_conv_classification_2_V(V2)


            # concatenate_classification = concatenate([merge_TPC_1, merge_TPC_2])
            # global_average_pooled = GlobalAveragePooling2D()(concatenate_classification)
            # classification_1 = Dense(16, activation='relu', name='Classification_1')(concatenate([flat_TPC_1, flat_TPC_2]))
            shared_classification_1 = Dense(16, activation='relu', name='shared_Classification_1')
            shared_classification_2 = Dense(8, activation='relu', name='shared_Classification_2')

            # classification1_TPC1 = shared_classification_1(concatenate([Flatten()(U1), Flatten()(V1)]))
            # classification1_TPC2 = shared_classification_1(concatenate([Flatten()(U2), Flatten()(V2)]))

            global_average_TPC1 = GlobalAveragePooling2D()(merge_TPC_1)
            global_average_TPC2 = GlobalAveragePooling2D()(merge_TPC_2)
            classification1_TPC1 = shared_classification_1(global_average_TPC1)
            classification1_TPC2 = shared_classification_1(global_average_TPC2)

            classification2_TPC1 = shared_classification_2(classification1_TPC1)
            classification2_TPC2 = shared_classification_2(classification1_TPC2)

            # classification_1 = Dense(16, activation='relu', name='Classification_1')(global_average_pooled)
            # classification_2 = Dense(8, activation='relu', name='Classification_2')(classification_1)

            # output_number_cluster = Dense(number_timesteps+1, activation='softmax', name='Output_Number_Cluster')(classification_2)
            output_number_cluster = Dense(1, name='Output_Number_Cluster')(concatenate([classification2_TPC1, classification2_TPC2]))


            repeated_TPC1 = RepeatVector(number_timesteps)(flat_TPC_1)
            repeated_TPC2 = RepeatVector(number_timesteps)(flat_TPC_2)

            # Define shared Dense Layers    32      16
            shared_dense_1 = SimpleRNN(16, activation='relu', name='Shared_1_TPC_1_and_2', return_sequences=True)
            shared_dense_2 = SimpleRNN(8, activation='relu', name='Shared_2_TPC_1_and_2', return_sequences=True)


            # lstm_1_TPC1 = LSTM(32, return_sequences=True)(repeated_TPC1)
            # lstm_1_TPC2 = LSTM(32, return_sequences=True)(repeated_TPC2)
            #
            # lstm_1_TPC1 = ELU(alpha=1.0)(lstm_1_TPC1)
            # lstm_1_TPC2 = ELU(alpha=1.0)(lstm_1_TPC2)
            #
            # lstm_2_TPC1 = LSTM(16, return_sequences=True)(lstm_1_TPC1)
            # lstm_2_TPC2 = LSTM(16, return_sequences=True)(lstm_1_TPC2)
            #
            # lstm_2_TPC1 = ELU(alpha=1.0)(lstm_2_TPC1)
            # lstm_2_TPC2 = ELU(alpha=1.0)(lstm_2_TPC2)
            #
            # lstm_3_TPC1 = LSTM(8, return_sequences=True)(lstm_2_TPC1)
            # lstm_3_TPC2 = LSTM(8, return_sequences=True)(lstm_2_TPC2)
            #
            # lstm_3_TPC1 = ELU(alpha=1.0)(lstm_3_TPC1)
            # lstm_3_TPC2 = ELU(alpha=1.0)(lstm_3_TPC2)

            # Define shared Dense Layers    32      16
            # shared_dense_1 = Dense(64, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
            # shared_dense_2 = Dense(32, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)
            #
            # # Dense Layers
            # dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
            # dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

            dense_1_TPC_1 = shared_dense_1(repeated_TPC1)
            dense_1_TPC_2 = shared_dense_1(repeated_TPC2)

            #
            # # dense_1_TPC_1 = shared_dense_1(lstm_out_TPC1)
            # # dense_1_TPC_2 = shared_dense_1(lstm_out_TPC2)
            #
            #
            dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
            dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

            # Merge Dense Layers
            merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')
            # merge_TPC_1_2 = concatenate([lstm_3_TPC1, lstm_3_TPC2], name='TPC_1_and_2')


    elif inputImages == 'U':
        # Flatten
        flat_TPC_1 = Flatten(name='Flat_TPC_1')(pooled_4_U_1)
        flat_TPC_2 = Flatten(name='Flat_TPC_2')(pooled_4_U_2)

        # Define shared Dense Layers
        shared_dense_1 = Dense(16, activation='relu', name='Shared_1_TPC_1_and_2')
        shared_dense_2 = Dense(8, activation='relu', name='Shared_2_TPC_1_and_2')

        # Dense Layers
        dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
        dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

        dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
        dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

        # Merge Dense Layers
        merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    elif inputImages == 'V':
        # Flatten
        flat_TPC_1 = Flatten(name='Flat_TPC_1')(pooled_4_V_1)
        flat_TPC_2 = Flatten(name='Flat_TPC_2')(pooled_4_V_2)

        # Define shared Dense Layers
        shared_dense_1 = Dense(16, activation='relu', name='Shared_1_TPC_1_and_2')
        shared_dense_2 = Dense(8, activation='relu', name='Shared_2_TPC_1_and_2')

        # Dense Layers
        dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
        dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

        dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
        dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

        # Merge Dense Layers
        merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Output
    if multiplicity == 'SS':
        if var_targets == 'energy' or var_targets == 'time' or var_targets == 'U' or var_targets == 'V' or var_targets == 'Z':
            output_exyz = Dense(1, name='Output')(merge_TPC_1_2)
        elif var_targets == 'energy_and_UV_position':
            output_exyz = Dense(4, name='Output_xyze')(merge_TPC_1_2)
        elif var_targets == 'position':
            output_exyz = Dense(3, name='Output_xyze')(merge_TPC_1_2)
        elif var_targets == 'U':
            output_exyz = Dense(1, name='Output_xyze')(merge_TPC_1_2)
        elif var_targets == 'Z':
            output_exyz = Dense(1, name='Output_xyze')(erge_TPC_1_2)

    if multiplicity == 'SS+MS':
        if var_targets == 'energy_and_UV_position':
            output_exyz = Dense(1, name='Output_xyze')(merge_TPC_1_2)
            # output_exyz = SimpleRNN(1, return_sequences=True, name='Output_xyze')(merge_TPC_1_2)
            # output_exyz = Dense(5, name='Output_xyze')(merge_TPC_1_2)

    if inputImages == 'UV':

        if var_targets == 'U' or var_targets == 'V' or var_targets == 'Z':
            return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_exyz])
        elif var_targets == 'Z':
            return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_exyz])

        # return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_number_cluster, output_exyz])        #outputs=[output_e, output_x, output_y, output_z])
        return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_exyz])

    elif inputImages == 'U':
        return Model(inputs=[visible_U_1, visible_U_2], outputs=[output_exyz])  # outputs=[output_xyze, output_TPC])
    elif inputImages == 'V':
        return Model(inputs=[visible_V_1, visible_V_2], outputs=[output_exyz])




def create_shared_ConvLSTM_network(var_targets, inputImages, multiplicity):
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling3D
    from keras.layers.merge import concatenate
    from keras import regularizers
    from keras.layers import RepeatVector
    from keras.layers import LSTM
    from keras.layers import ELU
    from keras.layers import ConvLSTM2D
    from keras.layers import TimeDistributed
    from keras.layers import Lambda
    from keras.layers import Reshape
    import keras.backend as K

    regu = regularizers.l2(0.01)
    var_targets = var_targets
    elu = ELU(alpha=1.0)

    number_timesteps = 3


    # Input layers
    visible_U_1 = Input(shape=(700, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(700, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(700, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(700, 38, 1), name='V_Wire_2')

    repeated_U1 = Reshape((1, 700, 38, 1))(visible_U_1)
    repeated_U2 = Reshape((1, 700, 38, 1))(visible_U_2)
    repeated_V1 = Reshape((1, 700, 38, 1))(visible_V_1)
    repeated_V2 = Reshape((1, 700, 38, 1))(visible_V_2)

    print '<<<<<'
    print visible_U_1.shape
    print repeated_U1.shape

    repeated_U1 = concatenate([repeated_U1, repeated_U1, repeated_U1], axis=1)
    repeated_U2 = concatenate([repeated_U2, repeated_U2, repeated_U2], axis=1)
    repeated_V1 = concatenate([repeated_V1, repeated_V1, repeated_V1], axis=1)
    repeated_V2 = concatenate([repeated_V2, repeated_V2, repeated_V2], axis=1)
    print repeated_U1.shape


    # Define U-wire shared layers   # 16, 32, 64, 100, 200, 300, 400, 500
    shared_conv_1_U = ConvLSTM2D(10, kernel_size=5, activation='relu', name='Shared_1_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_2_U = ConvLSTM2D(20, kernel_size=(5, 3), activation='relu', name='Shared_2_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_1_U = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_1p_U')
    shared_conv_3_U = ConvLSTM2D(30, kernel_size=(3, 1), activation='relu', name='Shared_3_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_4_U = ConvLSTM2D(40, kernel_size=(3, 1), activation='relu', name='Shared_4_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_2_U = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_2p_U')
    shared_conv_5_U = ConvLSTM2D(50, kernel_size=(3, 1), activation='relu', name='Shared_5_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_6_U = ConvLSTM2D(60, kernel_size=(3, 1), activation='relu', name='Shared_6_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_3_U = MaxPooling3D(pool_size=(1, 3, 3), name='Shared_3p_U')
    shared_conv_7_U = ConvLSTM2D(70, kernel_size=(3, 1), activation='relu', name='Shared_7_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_8_U = ConvLSTM2D(80, kernel_size=(3, 1), activation='relu', name='Shared_8_U', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_4_U = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_4p_U')

    # Define V-wire shared layers
    shared_conv_1_V = ConvLSTM2D(10, kernel_size=5, activation='relu', name='Shared_1_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_2_V = ConvLSTM2D(20, kernel_size=(5, 3), activation='relu', name='Shared_2_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_1_V = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_1p_V')
    shared_conv_3_V = ConvLSTM2D(30, kernel_size=(3, 1), activation='relu', name='Shared_3_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_4_V = ConvLSTM2D(40, kernel_size=(3, 1), activation='relu', name='Shared_4_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_2_V = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_2p_V')
    shared_conv_5_V = ConvLSTM2D(50, kernel_size=(3, 1), activation='relu', name='Shared_5_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_6_V = ConvLSTM2D(60, kernel_size=(3, 1), activation='relu', name='Shared_6_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_3_V = MaxPooling3D(pool_size=(1, 3, 3), name='Shared_3p_V')
    shared_conv_7_V = ConvLSTM2D(70, kernel_size=(3, 1), activation='relu', name='Shared_7_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_conv_8_V = ConvLSTM2D(80, kernel_size=(3, 1), activation='relu', name='Shared_8_V', kernel_regularizer=regu, data_format='channels_last', return_sequences=True)
    shared_pooling_4_V = MaxPooling3D(pool_size=(1, 3, 1), name='Shared_4p_V')

    # U-wire feature layers
    if inputImages == 'U' or 'UV':
        encoded_1_U_1 = shared_conv_1_U(repeated_U1)
        encoded_1_U_2 = shared_conv_1_U(repeated_U2)

        encoded_2_U_1 = shared_conv_2_U(encoded_1_U_1)
        encoded_2_U_2 = shared_conv_2_U(encoded_1_U_2)

        # pooled_1_U_1 = shared_pooling_1_U(encoded_2_U_1)
        # pooled_1_U_2 = shared_pooling_1_U(encoded_2_U_2)

        encoded_3_U_1 = shared_conv_3_U(encoded_2_U_1)
        encoded_3_U_2 = shared_conv_3_U(encoded_2_U_2)

        # encoded_3_U_1 = shared_conv_3_U(pooled_1_U_1)
        # encoded_3_U_2 = shared_conv_3_U(pooled_1_U_2)


        encoded_4_U_1 = shared_conv_4_U(encoded_3_U_1)
        encoded_4_U_2 = shared_conv_4_U(encoded_3_U_2)

        pooled_2_U_1 = shared_pooling_2_U(encoded_4_U_1)
        pooled_2_U_2 = shared_pooling_2_U(encoded_4_U_2)

        # encoded_5_U_1 = shared_conv_5_U(pooled_2_U_1)
        # encoded_5_U_2 = shared_conv_5_U(pooled_2_U_2)
        #
        # encoded_6_U_1 = shared_conv_6_U(encoded_5_U_1)
        # encoded_6_U_2 = shared_conv_6_U(encoded_5_U_2)
        #
        # pooled_3_U_1 = shared_pooling_3_U(encoded_6_U_1)
        # pooled_3_U_2 = shared_pooling_3_U(encoded_6_U_2)
        #
        # encoded_7_U_1 = shared_conv_7_U(pooled_3_U_1)
        # encoded_7_U_2 = shared_conv_7_U(pooled_3_U_2)
        #
        # encoded_8_U_1 = shared_conv_8_U(encoded_7_U_1)
        # encoded_8_U_2 = shared_conv_8_U(encoded_7_U_2)
        #
        # pooled_4_U_1 = shared_pooling_4_U(encoded_8_U_1)
        # pooled_4_U_2 = shared_pooling_4_U(encoded_8_U_2)


    # V-wire feature layers
    if inputImages == 'V' or 'UV':
        encoded_1_V_1 = shared_conv_1_V(repeated_V1)
        encoded_1_V_2 = shared_conv_1_V(repeated_V2)

        encoded_2_V_1 = shared_conv_2_V(encoded_1_V_1)
        encoded_2_V_2 = shared_conv_2_V(encoded_1_V_2)

        # pooled_1_V_1 = shared_pooling_1_V(encoded_2_V_1)
        # pooled_1_V_2 = shared_pooling_1_V(encoded_2_V_2)
        #
        encoded_3_V_1 = shared_conv_3_V(encoded_2_V_1)
        encoded_3_V_2 = shared_conv_3_V(encoded_2_V_2)
        #
        # encoded_3_V_1 = shared_conv_3_V(pooled_1_V_1)
        # encoded_3_V_2 = shared_conv_3_V(pooled_1_V_2)

        encoded_4_V_1 = shared_conv_4_V(encoded_3_V_1)
        encoded_4_V_2 = shared_conv_4_V(encoded_3_V_2)

        pooled_2_V_1 = shared_pooling_2_V(encoded_4_V_1)
        pooled_2_V_2 = shared_pooling_2_V(encoded_4_V_2)

        # encoded_5_V_1 = shared_conv_5_V(pooled_2_V_1)
        # encoded_5_V_2 = shared_conv_5_V(pooled_2_V_2)
        #
        # encoded_6_V_1 = shared_conv_6_V(encoded_5_V_1)
        # encoded_6_V_2 = shared_conv_6_V(encoded_5_V_2)
        #
        # pooled_3_V_1 = shared_pooling_3_V(encoded_6_V_1)
        # pooled_3_V_2 = shared_pooling_3_V(encoded_6_V_2)
        #
        # encoded_7_V_1 = shared_conv_7_V(pooled_3_V_1)
        # encoded_7_V_2 = shared_conv_7_V(pooled_3_V_2)
        #
        # encoded_8_V_1 = shared_conv_8_V(encoded_7_V_1)
        # encoded_8_V_2 = shared_conv_8_V(encoded_7_V_2)
        #
        # pooled_4_V_1 = shared_pooling_4_V(encoded_8_V_1)
        # pooled_4_V_2 = shared_pooling_4_V(encoded_8_V_2)

    # Merge U- and V-wire of TPC 1 and TPC 2
    if inputImages == 'UV':

        # merge_TPC_1 = concatenate([encoded_8_U_1, encoded_8_V_1], name='TPC_1')
        # merge_TPC_2 = concatenate([encoded_8_U_2, encoded_8_V_2], name='TPC_2')

        # merge_TPC_1 = concatenate([pooled_4_U_1, pooled_4_V_1], name='TPC_1')
        # merge_TPC_2 = concatenate([pooled_4_U_2, pooled_4_V_2], name='TPC_2')

        merge_TPC_1 = concatenate([pooled_2_U_1, pooled_2_V_1], name='TPC_1')
        merge_TPC_2 = concatenate([pooled_2_U_2, pooled_2_V_2], name='TPC_2')

        # Flatten
        # flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
        # flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

        flat_TPC_1 = Reshape((number_timesteps, -1))(merge_TPC_1)
        flat_TPC_2 = Reshape((number_timesteps, -1))(merge_TPC_2)

        if multiplicity == 'SS':

            # Define shared Dense Layers
            shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
            shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

            # Dense Layers
            dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
            dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

            dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
            dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

            # Merge Dense Layers
            merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')


        if multiplicity == 'SS+MS':

            # Define shared Dense Layers

            shared_dense_1 = TimeDistributed(Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu))
            shared_dense_2 = TimeDistributed(Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu))

            # shared_dense_1 = Dense(64, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
            # shared_dense_2 = Dense(32, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

            # Dense Layers
            # dense_1_TPC_1 = Dense(64, activation='relu')(flat_TPC_1)
            # dense_1_TPC_2 = Dense(64, activation='relu')(flat_TPC_2)

            dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
            dense_1_TPC_2 = shared_dense_1(flat_TPC_2)


            dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
            dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

            # Merge Dense Layers
            merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')
            # merge_TPC_1_2 = concatenate([lstm_3_TPC1, lstm_3_TPC2], name='TPC_1_and_2')


    # elif inputImages == 'U':
    #     # Flatten
    #     flat_TPC_1 = Flatten(name='Flat_TPC_1')(pooled_4_U_1)
    #     flat_TPC_2 = Flatten(name='Flat_TPC_2')(pooled_4_U_2)
    #
    #     # Define shared Dense Layers
    #     shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    #     shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)
    #
    #     # Dense Layers
    #     dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    #     dense_1_TPC_2 = shared_dense_1(flat_TPC_2)
    #
    #     dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    #     dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)
    #
    #     # Merge Dense Layers
    #     merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')
    #
    # elif inputImages == 'V':
    #     # Flatten
    #     flat_TPC_1 = Flatten(name='Flat_TPC_1')(pooled_4_V_1)
    #     flat_TPC_2 = Flatten(name='Flat_TPC_2')(pooled_4_V_2)
    #
    #     # Define shared Dense Layers
    #     shared_dense_1 = Dense(32, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    #     shared_dense_2 = Dense(16, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)
    #
    #     # Dense Layers
    #     dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    #     dense_1_TPC_2 = shared_dense_1(flat_TPC_2)
    #
    #     dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    #     dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)
    #
    #     # Merge Dense Layers
    #     merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Output
    if multiplicity == 'SS':
        if var_targets == 'energy' or var_targets == 'time' or var_targets == 'U' or var_targets == 'V':
            output_exyz = Dense(1, name='Output_e')(merge_TPC_1_2)
        elif var_targets == 'energy_and_UV_position':
            output_exyz = Dense(4, name='Output_xyze')(merge_TPC_1_2)
        elif var_targets == 'position':
            output_exyz = Dense(3, name='Output_xyze')(merge_TPC_1_2)

    if multiplicity == 'SS+MS':
        if var_targets == 'energy_and_UV_position':
            output_exyz = Dense(4, name='Output_xyze')(merge_TPC_1_2)
            # output_exyz = Dense(20, name='Output_xyze')(merge_TPC_1_2)

    if inputImages == 'UV':
        return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_exyz])        #outputs=[output_e, output_x, output_y, output_z])
    elif inputImages == 'U':
        return Model(inputs=[visible_U_1, visible_U_2], outputs=[output_exyz])  # outputs=[output_xyze, output_TPC])
    elif inputImages == 'V':
        return Model(inputs=[visible_V_1, visible_V_2], outputs=[output_exyz])




def create_inception_network():
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras import regularizers

    regu = regularizers.l2(0.01)

    # Input layers
    visible_U_1 = Input(shape=(2048, 38, 1), name='U_Wire_1')
    visible_U_2 = Input(shape=(2048, 38, 1), name='U_Wire_2')

    visible_V_1 = Input(shape=(2048, 38, 1), name='V_Wire_1')
    visible_V_2 = Input(shape=(2048, 38, 1), name='V_Wire_2')

    # shared Layer U-wire
    conv_1_U = Conv2D(64, (7, 7), activation='relu')

    # shared Inception module U-wire
    shared_inception_1_a_U = Conv2D(64, (1, 1), padding='same', activation='relu')
    shared_inception_1_b1_U = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_1_b2_U = Conv2D(128, (3, 3), padding='same', activation='relu')
    shared_inception_1_c1_U = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_1_c2_U = Conv2D(32, (5, 5), padding='same', activation='relu')
    shared_inception_1_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_1_d2_U = Conv2D(32, (1, 1), padding='same', activation='relu')

    shared_inception_2_a_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b1_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b2_U = Conv2D(192, (3, 3), padding='same', activation='relu')
    shared_inception_2_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_2_c2_U = Conv2D(96, (5, 5), padding='same', activation='relu')
    shared_inception_2_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_2_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_3_a_U = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_3_b1_U = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_3_b2_U = Conv2D(208, (3, 3), padding='same', activation='relu')
    shared_inception_3_c1_U = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_3_c2_U = Conv2D(48, (5, 5), padding='same', activation='relu')
    shared_inception_3_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_3_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_4_a_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_4_b1_U = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_4_b2_U = Conv2D(224, (3, 3), padding='same', activation='relu')
    shared_inception_4_c1_U = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_4_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_4_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_4_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_5_a_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b1_U = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b2_U = Conv2D(256, (3, 3), padding='same', activation='relu')
    shared_inception_5_c1_U = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_5_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_5_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_5_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_6_a_U = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_6_b1_U = Conv2D(144, (1, 1), padding='same', activation='relu')
    shared_inception_6_b2_U = Conv2D(288, (3, 3), padding='same', activation='relu')
    shared_inception_6_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_6_c2_U = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_6_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_6_d2_U = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_7_a_U = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_7_b1_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_7_b2_U = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_7_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_7_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_7_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_7_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_8_a_U = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_8_b1_U = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_8_b2_U = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_8_c1_U = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_8_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_8_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_8_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_9_a_U = Conv2D(384, (1, 1), padding='same', activation='relu')
    shared_inception_9_b1_U = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_9_b2_U = Conv2D(384, (3, 3), padding='same', activation='relu')
    shared_inception_9_c1_U = Conv2D(48, (1, 1), padding='same', activation='relu')
    shared_inception_9_c2_U = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_9_d1_U = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_9_d2_U = Conv2D(128, (1, 1), padding='same', activation='relu')

    # shared Layer V-wire
    conv_1_V = Conv2D(64, (7, 7), activation='relu')

    # shared Inception module V-wire
    shared_inception_1_a_V = Conv2D(64, (1, 1), padding='same', activation='relu')
    shared_inception_1_b1_V = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_1_b2_V = Conv2D(128, (3, 3), padding='same', activation='relu')
    shared_inception_1_c1_V = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_1_c2_V = Conv2D(32, (5, 5), padding='same', activation='relu')
    shared_inception_1_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_1_d2_V = Conv2D(32, (1, 1), padding='same', activation='relu')

    shared_inception_2_a_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b1_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_2_b2_V = Conv2D(192, (3, 3), padding='same', activation='relu')
    shared_inception_2_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_2_c2_V = Conv2D(96, (5, 5), padding='same', activation='relu')
    shared_inception_2_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_2_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_3_a_V = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_3_b1_V = Conv2D(96, (1, 1), padding='same', activation='relu')
    shared_inception_3_b2_V = Conv2D(208, (3, 3), padding='same', activation='relu')
    shared_inception_3_c1_V = Conv2D(16, (1, 1), padding='same', activation='relu')
    shared_inception_3_c2_V = Conv2D(48, (5, 5), padding='same', activation='relu')
    shared_inception_3_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_3_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_4_a_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_4_b1_V = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_4_b2_V = Conv2D(224, (3, 3), padding='same', activation='relu')
    shared_inception_4_c1_V = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_4_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_4_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_4_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_5_a_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b1_V = Conv2D(128, (1, 1), padding='same', activation='relu')
    shared_inception_5_b2_V = Conv2D(256, (3, 3), padding='same', activation='relu')
    shared_inception_5_c1_V = Conv2D(24, (1, 1), padding='same', activation='relu')
    shared_inception_5_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_5_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_5_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_6_a_V = Conv2D(112, (1, 1), padding='same', activation='relu')
    shared_inception_6_b1_V = Conv2D(144, (1, 1), padding='same', activation='relu')
    shared_inception_6_b2_V = Conv2D(288, (3, 3), padding='same', activation='relu')
    shared_inception_6_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_6_c2_V = Conv2D(64, (5, 5), padding='same', activation='relu')
    shared_inception_6_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_6_d2_V = Conv2D(64, (1, 1), padding='same', activation='relu')

    shared_inception_7_a_V = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_7_b1_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_7_b2_V = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_7_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_7_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_7_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_7_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_8_a_V = Conv2D(256, (1, 1), padding='same', activation='relu')
    shared_inception_8_b1_V = Conv2D(160, (1, 1), padding='same', activation='relu')
    shared_inception_8_b2_V = Conv2D(320, (3, 3), padding='same', activation='relu')
    shared_inception_8_c1_V = Conv2D(32, (1, 1), padding='same', activation='relu')
    shared_inception_8_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_8_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_8_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    shared_inception_9_a_V = Conv2D(384, (1, 1), padding='same', activation='relu')
    shared_inception_9_b1_V = Conv2D(192, (1, 1), padding='same', activation='relu')
    shared_inception_9_b2_V = Conv2D(384, (3, 3), padding='same', activation='relu')
    shared_inception_9_c1_V = Conv2D(48, (1, 1), padding='same', activation='relu')
    shared_inception_9_c2_V = Conv2D(128, (5, 5), padding='same', activation='relu')
    shared_inception_9_d1_V = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    shared_inception_9_d2_V = Conv2D(128, (1, 1), padding='same', activation='relu')

    # Application U-1
    # U_1_conv_1 = conv_1_U(visible_U_1)
    # U_1_conv_1 = MaxPooling2D(pool_size=3)(U_1_conv_1)

    U_1_1_a = shared_inception_1_a_U(visible_U_1)
    U_1_1_b = shared_inception_1_b2_U(shared_inception_1_b1_U(visible_U_1))
    U_1_1_c = shared_inception_1_c2_U(shared_inception_1_c1_U(visible_U_1))
    U_1_1_d = shared_inception_1_d2_U(shared_inception_1_d1_U(visible_U_1))
    inception_1_U_1 = concatenate([U_1_1_a, U_1_1_b, U_1_1_c, U_1_1_d], axis=3)

    U_1_2_a = shared_inception_2_a_U(inception_1_U_1)
    U_1_2_b = shared_inception_2_b2_U(shared_inception_2_b1_U(inception_1_U_1))
    U_1_2_c = shared_inception_2_c2_U(shared_inception_2_c1_U(inception_1_U_1))
    U_1_2_d = shared_inception_2_d2_U(shared_inception_2_d1_U(inception_1_U_1))
    inception_2_U_1 = concatenate([U_1_2_a, U_1_2_b, U_1_2_c, U_1_2_d], axis=3)

    inception_2_U_1 = MaxPooling2D(pool_size=3)(inception_2_U_1)

    U_1_3_a = shared_inception_3_a_U(inception_2_U_1)
    U_1_3_b = shared_inception_3_b2_U(shared_inception_3_b1_U(inception_2_U_1))
    U_1_3_c = shared_inception_3_c2_U(shared_inception_3_c1_U(inception_2_U_1))
    U_1_3_d = shared_inception_3_d2_U(shared_inception_3_d1_U(inception_2_U_1))
    inception_3_U_1 = concatenate([U_1_3_a, U_1_3_b, U_1_3_c, U_1_3_d], axis=3)

    U_1_4_a = shared_inception_4_a_U(inception_3_U_1)
    U_1_4_b = shared_inception_4_b2_U(shared_inception_4_b1_U(inception_3_U_1))
    U_1_4_c = shared_inception_4_c2_U(shared_inception_4_c1_U(inception_3_U_1))
    U_1_4_d = shared_inception_4_d2_U(shared_inception_4_d1_U(inception_3_U_1))
    inception_4_U_1 = concatenate([U_1_4_a, U_1_4_b, U_1_4_c, U_1_4_d], axis=3)

    inception_4_U_1 = MaxPooling2D(pool_size=3)(inception_4_U_1)

    U_1_5_a = shared_inception_5_a_U(inception_4_U_1)
    U_1_5_b = shared_inception_5_b2_U(shared_inception_5_b1_U(inception_4_U_1))
    U_1_5_c = shared_inception_5_c2_U(shared_inception_5_c1_U(inception_4_U_1))
    U_1_5_d = shared_inception_5_d2_U(shared_inception_5_d1_U(inception_4_U_1))
    inception_5_U_1 = concatenate([U_1_5_a, U_1_5_b, U_1_5_c, U_1_5_d], axis=3)

    U_1_6_a = shared_inception_6_a_U(inception_5_U_1)
    U_1_6_b = shared_inception_6_b2_U(shared_inception_6_b1_U(inception_5_U_1))
    U_1_6_c = shared_inception_6_c2_U(shared_inception_6_c1_U(inception_5_U_1))
    U_1_6_d = shared_inception_6_d2_U(shared_inception_6_d1_U(inception_5_U_1))
    inception_6_U_1 = concatenate([U_1_6_a, U_1_6_b, U_1_6_c, U_1_6_d], axis=3)

    U_1_7_a = shared_inception_7_a_U(inception_6_U_1)
    U_1_7_b = shared_inception_7_b2_U(shared_inception_7_b1_U(inception_6_U_1))
    U_1_7_c = shared_inception_7_c2_U(shared_inception_7_c1_U(inception_6_U_1))
    U_1_7_d = shared_inception_7_d2_U(shared_inception_7_d1_U(inception_6_U_1))
    inception_7_U_1 = concatenate([U_1_7_a, U_1_7_b, U_1_7_c, U_1_7_d], axis=3)

    inception_7_U_1 = MaxPooling2D(pool_size=3)(inception_7_U_1)

    U_1_8_a = shared_inception_8_a_U(inception_7_U_1)
    U_1_8_b = shared_inception_8_b2_U(shared_inception_8_b1_U(inception_7_U_1))
    U_1_8_c = shared_inception_8_c2_U(shared_inception_8_c1_U(inception_7_U_1))
    U_1_8_d = shared_inception_8_d2_U(shared_inception_8_d1_U(inception_7_U_1))
    inception_8_U_1 = concatenate([U_1_8_a, U_1_8_b, U_1_8_c, U_1_8_d], axis=3)

    U_1_9_a = shared_inception_9_a_U(inception_8_U_1)
    U_1_9_b = shared_inception_9_b2_U(shared_inception_9_b1_U(inception_8_U_1))
    U_1_9_c = shared_inception_9_c2_U(shared_inception_9_c1_U(inception_8_U_1))
    U_1_9_d = shared_inception_9_d2_U(shared_inception_9_d1_U(inception_8_U_1))
    inception_9_U_1 = concatenate([U_1_9_a, U_1_9_b, U_1_9_c, U_1_9_d], axis=3)

    inception_9_U_1 = MaxPooling2D(pool_size=(7, 1))(inception_9_U_1)


    # Application U-2
    # U_2_conv_1 = conv_1_U(visible_U_2)
    # U_2_conv_1 = MaxPooling2D(pool_size=3)(U_2_conv_1)

    U_2_1_a = shared_inception_1_a_U(visible_U_2)
    U_2_1_b = shared_inception_1_b2_U(shared_inception_1_b1_U(visible_U_2))
    U_2_1_c = shared_inception_1_c2_U(shared_inception_1_c1_U(visible_U_2))
    U_2_1_d = shared_inception_1_d2_U(shared_inception_1_d1_U(visible_U_2))
    inception_1_U_2 = concatenate([U_2_1_a, U_2_1_b, U_2_1_c, U_2_1_d], axis=3)

    U_2_2_a = shared_inception_2_a_U(inception_1_U_2)
    U_2_2_b = shared_inception_2_b2_U(shared_inception_2_b1_U(inception_1_U_2))
    U_2_2_c = shared_inception_2_c2_U(shared_inception_2_c1_U(inception_1_U_2))
    U_2_2_d = shared_inception_2_d2_U(shared_inception_2_d1_U(inception_1_U_2))
    inception_2_U_2 = concatenate([U_2_2_a, U_2_2_b, U_2_2_c, U_2_2_d], axis=3)

    inception_2_U_2 = MaxPooling2D(pool_size=3)(inception_2_U_2)

    U_2_3_a = shared_inception_3_a_U(inception_2_U_2)
    U_2_3_b = shared_inception_3_b2_U(shared_inception_3_b1_U(inception_2_U_2))
    U_2_3_c = shared_inception_3_c2_U(shared_inception_3_c1_U(inception_2_U_2))
    U_2_3_d = shared_inception_3_d2_U(shared_inception_3_d1_U(inception_2_U_2))
    inception_3_U_2 = concatenate([U_2_3_a, U_2_3_b, U_2_3_c, U_2_3_d], axis=3)

    U_2_4_a = shared_inception_4_a_U(inception_3_U_2)
    U_2_4_b = shared_inception_4_b2_U(shared_inception_4_b1_U(inception_3_U_2))
    U_2_4_c = shared_inception_4_c2_U(shared_inception_4_c1_U(inception_3_U_2))
    U_2_4_d = shared_inception_4_d2_U(shared_inception_4_d1_U(inception_3_U_2))
    inception_4_U_2 = concatenate([U_2_4_a, U_2_4_b, U_2_4_c, U_2_4_d], axis=3)

    inception_4_U_2 = MaxPooling2D(pool_size=3)(inception_4_U_2)

    U_2_5_a = shared_inception_5_a_U(inception_4_U_2)
    U_2_5_b = shared_inception_5_b2_U(shared_inception_5_b1_U(inception_4_U_2))
    U_2_5_c = shared_inception_5_c2_U(shared_inception_5_c1_U(inception_4_U_2))
    U_2_5_d = shared_inception_5_d2_U(shared_inception_5_d1_U(inception_4_U_2))
    inception_5_U_2 = concatenate([U_2_5_a, U_2_5_b, U_2_5_c, U_2_5_d], axis=3)

    U_2_6_a = shared_inception_6_a_U(inception_5_U_2)
    U_2_6_b = shared_inception_6_b2_U(shared_inception_6_b1_U(inception_5_U_2))
    U_2_6_c = shared_inception_6_c2_U(shared_inception_6_c1_U(inception_5_U_2))
    U_2_6_d = shared_inception_6_d2_U(shared_inception_6_d1_U(inception_5_U_2))
    inception_6_U_2 = concatenate([U_2_6_a, U_2_6_b, U_2_6_c, U_2_6_d], axis=3)

    U_2_7_a = shared_inception_7_a_U(inception_6_U_2)
    U_2_7_b = shared_inception_7_b2_U(shared_inception_7_b1_U(inception_6_U_2))
    U_2_7_c = shared_inception_7_c2_U(shared_inception_7_c1_U(inception_6_U_2))
    U_2_7_d = shared_inception_7_d2_U(shared_inception_7_d1_U(inception_6_U_2))
    inception_7_U_2 = concatenate([U_2_7_a, U_2_7_b, U_2_7_c, U_2_7_d], axis=3)

    inception_7_U_2 = MaxPooling2D(pool_size=3)(inception_7_U_2)

    U_2_8_a = shared_inception_8_a_U(inception_7_U_2)
    U_2_8_b = shared_inception_8_b2_U(shared_inception_8_b1_U(inception_7_U_2))
    U_2_8_c = shared_inception_8_c2_U(shared_inception_8_c1_U(inception_7_U_2))
    U_2_8_d = shared_inception_8_d2_U(shared_inception_8_d1_U(inception_7_U_2))
    inception_8_U_2 = concatenate([U_2_8_a, U_2_8_b, U_2_8_c, U_2_8_d], axis=3)

    U_2_9_a = shared_inception_9_a_U(inception_8_U_2)
    U_2_9_b = shared_inception_9_b2_U(shared_inception_9_b1_U(inception_8_U_2))
    U_2_9_c = shared_inception_9_c2_U(shared_inception_9_c1_U(inception_8_U_2))
    U_2_9_d = shared_inception_9_d2_U(shared_inception_9_d1_U(inception_8_U_2))
    inception_9_U_2 = concatenate([U_2_9_a, U_2_9_b, U_2_9_c, U_2_9_d], axis=3)

    inception_9_U_2 = MaxPooling2D(pool_size=(7, 1))(inception_9_U_2)



    # Application V-1
    # V_1_conv_1 = conv_1_V(visible_V_1)
    # V_1_conv_1 = MaxPooling2D(pool_size=3)(V_1_conv_1)

    V_1_1_a = shared_inception_1_a_V(visible_V_1)
    V_1_1_b = shared_inception_1_b2_V(shared_inception_1_b1_V(visible_V_1))
    V_1_1_c = shared_inception_1_c2_V(shared_inception_1_c1_V(visible_V_1))
    V_1_1_d = shared_inception_1_d2_V(shared_inception_1_d1_V(visible_V_1))
    inception_1_V_1 = concatenate([V_1_1_a, V_1_1_b, V_1_1_c, V_1_1_d], axis=3)

    V_1_2_a = shared_inception_2_a_V(inception_1_V_1)
    V_1_2_b = shared_inception_2_b2_V(shared_inception_2_b1_V(inception_1_V_1))
    V_1_2_c = shared_inception_2_c2_V(shared_inception_2_c1_V(inception_1_V_1))
    V_1_2_d = shared_inception_2_d2_V(shared_inception_2_d1_V(inception_1_V_1))
    inception_2_V_1 = concatenate([V_1_2_a, V_1_2_b, V_1_2_c, V_1_2_d], axis=3)

    inception_2_V_1 = MaxPooling2D(pool_size=3)(inception_2_V_1)

    V_1_3_a = shared_inception_3_a_V(inception_2_V_1)
    V_1_3_b = shared_inception_3_b2_V(shared_inception_3_b1_V(inception_2_V_1))
    V_1_3_c = shared_inception_3_c2_V(shared_inception_3_c1_V(inception_2_V_1))
    V_1_3_d = shared_inception_3_d2_V(shared_inception_3_d1_V(inception_2_V_1))
    inception_3_V_1 = concatenate([V_1_3_a, V_1_3_b, V_1_3_c, V_1_3_d], axis=3)

    V_1_4_a = shared_inception_4_a_V(inception_3_V_1)
    V_1_4_b = shared_inception_4_b2_V(shared_inception_4_b1_V(inception_3_V_1))
    V_1_4_c = shared_inception_4_c2_V(shared_inception_4_c1_V(inception_3_V_1))
    V_1_4_d = shared_inception_4_d2_V(shared_inception_4_d1_V(inception_3_V_1))
    inception_4_V_1 = concatenate([V_1_4_a, V_1_4_b, V_1_4_c, V_1_4_d], axis=3)

    inception_4_V_1 = MaxPooling2D(pool_size=3)(inception_4_V_1)

    V_1_5_a = shared_inception_5_a_V(inception_4_V_1)
    V_1_5_b = shared_inception_5_b2_V(shared_inception_5_b1_V(inception_4_V_1))
    V_1_5_c = shared_inception_5_c2_V(shared_inception_5_c1_V(inception_4_V_1))
    V_1_5_d = shared_inception_5_d2_V(shared_inception_5_d1_V(inception_4_V_1))
    inception_5_V_1 = concatenate([V_1_5_a, V_1_5_b, V_1_5_c, V_1_5_d], axis=3)

    V_1_6_a = shared_inception_6_a_V(inception_5_V_1)
    V_1_6_b = shared_inception_6_b2_V(shared_inception_6_b1_V(inception_5_V_1))
    V_1_6_c = shared_inception_6_c2_V(shared_inception_6_c1_V(inception_5_V_1))
    V_1_6_d = shared_inception_6_d2_V(shared_inception_6_d1_V(inception_5_V_1))
    inception_6_V_1 = concatenate([V_1_6_a, V_1_6_b, V_1_6_c, V_1_6_d], axis=3)

    V_1_7_a = shared_inception_7_a_V(inception_6_V_1)
    V_1_7_b = shared_inception_7_b2_V(shared_inception_7_b1_V(inception_6_V_1))
    V_1_7_c = shared_inception_7_c2_V(shared_inception_7_c1_V(inception_6_V_1))
    V_1_7_d = shared_inception_7_d2_V(shared_inception_7_d1_V(inception_6_V_1))
    inception_7_V_1 = concatenate([V_1_7_a, V_1_7_b, V_1_7_c, V_1_7_d], axis=3)

    inception_7_V_1 = MaxPooling2D(pool_size=3)(inception_7_V_1)

    V_1_8_a = shared_inception_8_a_V(inception_7_V_1)
    V_1_8_b = shared_inception_8_b2_V(shared_inception_8_b1_V(inception_7_V_1))
    V_1_8_c = shared_inception_8_c2_V(shared_inception_8_c1_V(inception_7_V_1))
    V_1_8_d = shared_inception_8_d2_V(shared_inception_8_d1_V(inception_7_V_1))
    inception_8_V_1 = concatenate([V_1_8_a, V_1_8_b, V_1_8_c, V_1_8_d], axis=3)

    V_1_9_a = shared_inception_9_a_V(inception_8_V_1)
    V_1_9_b = shared_inception_9_b2_V(shared_inception_9_b1_V(inception_8_V_1))
    V_1_9_c = shared_inception_9_c2_V(shared_inception_9_c1_V(inception_8_V_1))
    V_1_9_d = shared_inception_9_d2_V(shared_inception_9_d1_V(inception_8_V_1))
    inception_9_V_1 = concatenate([V_1_9_a, V_1_9_b, V_1_9_c, V_1_9_d], axis=3)

    inception_9_V_1 = MaxPooling2D(pool_size=(7, 1))(inception_9_V_1)


    # Application V-2
    # V_2_conv_1 = conv_1_V(visible_V_2)
    # V_2_conv_1 = MaxPooling2D(pool_size=3)(V_2_conv_1)

    V_2_1_a = shared_inception_1_a_V(visible_V_2)
    V_2_1_b = shared_inception_1_b2_V(shared_inception_1_b1_V(visible_V_2))
    V_2_1_c = shared_inception_1_c2_V(shared_inception_1_c1_V(visible_V_2))
    V_2_1_d = shared_inception_1_d2_V(shared_inception_1_d1_V(visible_V_2))
    inception_1_V_2 = concatenate([V_2_1_a, V_2_1_b, V_2_1_c, V_2_1_d], axis=3)

    V_2_2_a = shared_inception_2_a_V(inception_1_V_2)
    V_2_2_b = shared_inception_2_b2_V(shared_inception_2_b1_V(inception_1_V_2))
    V_2_2_c = shared_inception_2_c2_V(shared_inception_2_c1_V(inception_1_V_2))
    V_2_2_d = shared_inception_2_d2_V(shared_inception_2_d1_V(inception_1_V_2))
    inception_2_V_2 = concatenate([V_2_2_a, V_2_2_b, V_2_2_c, V_2_2_d], axis=3)

    inception_2_V_2 = MaxPooling2D(pool_size=3)(inception_2_V_2)

    V_2_3_a = shared_inception_3_a_V(inception_2_V_2)
    V_2_3_b = shared_inception_3_b2_V(shared_inception_3_b1_V(inception_2_V_2))
    V_2_3_c = shared_inception_3_c2_V(shared_inception_3_c1_V(inception_2_V_2))
    V_2_3_d = shared_inception_3_d2_V(shared_inception_3_d1_V(inception_2_V_2))
    inception_3_V_2 = concatenate([V_2_3_a, V_2_3_b, V_2_3_c, V_2_3_d], axis=3)

    V_2_4_a = shared_inception_4_a_V(inception_3_V_2)
    V_2_4_b = shared_inception_4_b2_V(shared_inception_4_b1_V(inception_3_V_2))
    V_2_4_c = shared_inception_4_c2_V(shared_inception_4_c1_V(inception_3_V_2))
    V_2_4_d = shared_inception_4_d2_V(shared_inception_4_d1_V(inception_3_V_2))
    inception_4_V_2 = concatenate([V_2_4_a, V_2_4_b, V_2_4_c, V_2_4_d], axis=3)

    inception_4_V_2 = MaxPooling2D(pool_size=3)(inception_4_V_2)

    V_2_5_a = shared_inception_5_a_V(inception_4_V_2)
    V_2_5_b = shared_inception_5_b2_V(shared_inception_5_b1_V(inception_4_V_2))
    V_2_5_c = shared_inception_5_c2_V(shared_inception_5_c1_V(inception_4_V_2))
    V_2_5_d = shared_inception_5_d2_V(shared_inception_5_d1_V(inception_4_V_2))
    inception_5_V_2 = concatenate([V_2_5_a, V_2_5_b, V_2_5_c, V_2_5_d], axis=3)

    V_2_6_a = shared_inception_6_a_V(inception_5_V_2)
    V_2_6_b = shared_inception_6_b2_V(shared_inception_6_b1_V(inception_5_V_2))
    V_2_6_c = shared_inception_6_c2_V(shared_inception_6_c1_V(inception_5_V_2))
    V_2_6_d = shared_inception_6_d2_V(shared_inception_6_d1_V(inception_5_V_2))
    inception_6_V_2 = concatenate([V_2_6_a, V_2_6_b, V_2_6_c, V_2_6_d], axis=3)

    V_2_7_a = shared_inception_7_a_V(inception_6_V_2)
    V_2_7_b = shared_inception_7_b2_V(shared_inception_7_b1_V(inception_6_V_2))
    V_2_7_c = shared_inception_7_c2_V(shared_inception_7_c1_V(inception_6_V_2))
    V_2_7_d = shared_inception_7_d2_V(shared_inception_7_d1_V(inception_6_V_2))
    inception_7_V_2 = concatenate([V_2_7_a, V_2_7_b, V_2_7_c, V_2_7_d], axis=3)

    inception_7_V_2 = MaxPooling2D(pool_size=3)(inception_7_V_2)

    V_2_8_a = shared_inception_8_a_V(inception_7_V_2)
    V_2_8_b = shared_inception_8_b2_V(shared_inception_8_b1_V(inception_7_V_2))
    V_2_8_c = shared_inception_8_c2_V(shared_inception_8_c1_V(inception_7_V_2))
    V_2_8_d = shared_inception_8_d2_V(shared_inception_8_d1_V(inception_7_V_2))
    inception_8_V_2 = concatenate([V_2_8_a, V_2_8_b, V_2_8_c, V_2_8_d], axis=3)

    V_2_9_a = shared_inception_9_a_V(inception_8_V_2)
    V_2_9_b = shared_inception_9_b2_V(shared_inception_9_b1_V(inception_8_V_2))
    V_2_9_c = shared_inception_9_c2_V(shared_inception_9_c1_V(inception_8_V_2))
    V_2_9_d = shared_inception_9_d2_V(shared_inception_9_d1_V(inception_8_V_2))
    inception_9_V_2 = concatenate([V_2_9_a, V_2_9_b, V_2_9_c, V_2_9_d], axis=3)

    inception_9_V_2 = MaxPooling2D(pool_size=(7, 1))(inception_9_V_2)


    # Merge U- and V-wire of TPC 1 and TPC 2
    merge_TPC_1 = concatenate([inception_9_U_1, inception_9_V_1], name='TPC_1')
    merge_TPC_2 = concatenate([inception_9_U_2, inception_9_V_2], name='TPC_2')

    # Flatten
    flat_TPC_1 = Flatten(name='Flat_TPC_1')(merge_TPC_1)
    flat_TPC_2 = Flatten(name='Flat_TPC_2')(merge_TPC_2)

    # Define shared Dense Layers
    shared_dense_1 = Dense(16, activation='relu', name='Shared_1_TPC_1_and_2', kernel_regularizer=regu)
    shared_dense_2 = Dense(8, activation='relu', name='Shared_2_TPC_1_and_2', kernel_regularizer=regu)

    # Dense Layers
    dense_1_TPC_1 = shared_dense_1(flat_TPC_1)
    dense_1_TPC_2 = shared_dense_1(flat_TPC_2)

    dense_2_TPC_1 = shared_dense_2(dense_1_TPC_1)
    dense_2_TPC_2 = shared_dense_2(dense_1_TPC_2)

    # Merge Dense Layers
    merge_TPC_1_2 = concatenate([dense_2_TPC_1, dense_2_TPC_2], name='TPC_1_and_2')

    # Output
    output_xyze = Dense(4, name='Output_xyze')(merge_TPC_1_2)

    return Model(inputs=[visible_U_1, visible_V_1, visible_U_2, visible_V_2], outputs=[output_xyze])



