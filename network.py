from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Masking


def building_ner(num_lstm_layer, num_hidden_node, dropout, time_step, vector_length, output_lenght):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_step, vector_length)))
    for i in range(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                     recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout), merge_mode='concat'))
    model.add(TimeDistributed(Dense(output_lenght)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
