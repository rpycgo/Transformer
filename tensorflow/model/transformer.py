from ..layers import Encoder, Decoder
from ..utils import is_padded, mask_future_data

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model


class Transformer:
    def __init__(self,
        vocab_size, 
        max_len, 
        num_encoders=6, 
        num_decoders=6, 
        d_model=512, 
        h=8, 
        d_ff=2048, 
        dropout_rate=0.1, 
        name='transformer'
        ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.name = name

        self.encoder = Encoder(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            num_encoders=self.num_encoders,
            d_model=self.d_model,
            h=self.h,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )()
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            num_decoders=self.num_decoders,
            d_model=self.d_model,
            h=self.h,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )()
    
    def __call__(self):
        encoder_input = Input(shape=(None,), name="encoder_input")
        decoder_input = Input(shape=(None,), name="decoder_input")

        encoder_masking = Lambda(
            is_padded,
            output_shape=(1, 1, None),
            name='encoder_masking'
            )(encoder_input)
        decoder_masking_for_leftward = Lambda(
            mask_future_data,
            output_shape=(1, None, None),
            name='decoder_masking_for_leftward'
            )(decoder_input)
        decoder_masking = Lambda(
            is_padded, 
            output_shape=(1, 1, None),
            name='decoder_masking'
            )(encoder_input)


        encoder_output = self.encoder(inputs=[encoder_input, encoder_masking])
        decoder_output = self.decoder(
            inputs=[decoder_input, encoder_output, decoder_masking_for_leftward, decoder_masking]
            )
        output = Dense(units=self.vocab_size, name='outputs')(decoder_output)

        model = Model(
            inputs=[encoder_input, decoder_input],
            outputs=output,
            name=self.name
            )

        return model
