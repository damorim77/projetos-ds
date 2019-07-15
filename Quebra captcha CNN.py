#!/usr/bin/env python
# coding: utf-8

# # Modelo para quebra de Captchas simples
# 
# Modelo de rede neural convolucional (CNN) em Keras que consegue realizar a resolução de captchas do tipo abaixo:
# 
# ![Captcha](https://i.screenshot.net/yoperf2)
# 
# O modelo foi treinado com base em **1.177 captchas** resolvidos manualmente, que compuseram um total de **7.062 letras/digitos** (média de 100 exemplares por letra/digito)
# 
# Como as posições das letras são fixas, é possível quebrar a imagem em 6 letras, e realizar a previsão de cada uma individualmente:
# 
# ![Captcha separado](https://i.screenshot.net/y1vopho)
# 
# ----
# [Link](http://danilo-amorim.000webhost.com/files/captcha_app/base_captcha.zip) para download da base

# # PARTE 1 - TREINAMENTO DO MODELO
# 
# ## Importações e definições de caminhos

# In[140]:


from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing import image
from datetime import datetime
from pprint import pprint
import numpy as np
import os

LETRA_DIM = ( 29 , 34 )
CUTPOINT = {
	"top"  : 7 ,
	"left" : [ 10 , 36 , 66 , 96 , 126 , 152 ]
}

ROOT = "C:/projetos/captcha_dev/mei/dataset/"

train_path = ROOT + "train"
valid_path = ROOT + "validation"
test_path  = ROOT + "test"
log_path  = ROOT + "logs"
prediction_path = ROOT + "prediction"
log_file = log_path + "/cnn_training.log"
model_weights_file  = ROOT + "cnn_weights.h5"
model_structure_file  = ROOT + "cnn_model.json"
model_index_file  = ROOT + "cnn_class_index.json"

captcha_file = "C:/projetos/captcha_app/mei/base_captchas/1aours.png"
letra_template = prediction_path + "/captcha_{ix}.png"


# ## Construção do modelo, ou carregamento de modelo existente

# In[126]:


def build_model():
    cnn = Sequential()

    cnn.add(Conv2D(32, (3,3), input_shape = (29, 34, 1), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size = (2,2)))

    cnn.add(Conv2D(32, (3,3), input_shape = (29, 34, 1), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size = (2,2)))

    cnn.add(Flatten())

    cnn.add(Dense(units = 256, activation = 'relu'))
    cnn.add(Dropout(0.4))
    cnn.add(Dense(units = 62, activation = 'relu'))
    cnn.add(Dropout(0.4))

    cnn.add(Dense(units = 62, activation = 'softmax'))

    cnn.compile( optimizer = 'adam'
               , loss = 'categorical_crossentropy'
               , metrics = ['accuracy'])
    
    return cnn

def load_model():
    model = model_from_json(open(model_structure_file).read())
    model.load_weights(model_weights_file)
    model.compile(  optimizer = 'adam'
                  , loss = 'categorical_crossentropy'
                  , metrics = ['accuracy'] )
    return model

def save_model(model):    
    json_string = model.to_json()
    open(model_structure_file, 'w').write(json_string)
    model.save_weights(model_weights_file, overwrite=True)

try:
    print("Carregando modelo")
#     cnn = build_model()
    cnn = load_model()
    print("Modelo carregado")
except Exception as ex:
    print("Erro ao carregar : \n" + str(ex) +  "\nConstruindo modelo")
    cnn = build_model()
    print("Modelo construído")
   


# ## Leitura das imagens separadas nos diretórios de treino, validação e teste
# 
# Utilizamos a classe ImageDataGenerator do Keras para facilitar a leitura dos diretórios

# In[23]:


train_gen, test_gen, valid_gen = ImageDataGenerator(), ImageDataGenerator(), ImageDataGenerator()

base_train = train_gen.flow_from_directory( train_path
                                          , target_size = (29, 34)
                                          , batch_size = 32
                                          , color_mode = 'grayscale'
                                          , class_mode = 'categorical'
                                          , shuffle = False
                                          , seed = 10 )

base_valid = valid_gen.flow_from_directory( valid_path
                                          , target_size = (29, 34)
                                          , batch_size = 32
                                          , color_mode = 'grayscale'
                                          , class_mode = 'categorical'
                                          , shuffle = False
                                          , seed = 10 )

base_test  = test_gen.flow_from_directory(  test_path
                                          , target_size = (29,34)
                                          , batch_size = 1
                                          , color_mode = 'grayscale'
                                          , class_mode = None
                                          , shuffle = False
                                          , seed = 10 )


# ## Definição de callbacks e fit do modelo
# 
# Os callbacks realizam as seguintes funções:
# * EarlyStopping: para o treinamento caso não haja melhora na função de perda por N época
# * ReduceLROnPlateau: reduz o learning rate do modelo quando a função de perda estiver próxima da estagnação
# * ModelCheckpoint: salva progresso  do treinamento
# * Tensorboard: permite acompanhamento do treinamento pela ferramenta Tensorboard
# * CSVLogger: grava log do treinamento em csv
# 
# 

# In[97]:


steps_train = base_train.n // base_train.batch_size
steps_valid = base_valid.n // base_valid.batch_size

callbacks = [
    EarlyStopping( monitor = 'loss'
                 , min_delta = 1e-3
                 , patience = 20
                 , verbose = 1 )
    ,
    ReduceLROnPlateau( monitor = 'loss'
                     , factor = 0.2
                     , patience = 5
                     , verbose = 1 )
    ,
    ModelCheckpoint( filepath = model_weights_path
                   , monitor = 'loss'
                   , save_best_only = True
                   , verbose = 1 )
    ,
    TensorBoard( log_dir="{path}/{time}" \
                    .format( path = log_path
                           , time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") ) )
    ,
    CSVLogger(log_file)
]

cnn.fit_generator( generator = base_train
                 , steps_per_epoch = steps_train 
                 , validation_data = base_valid
                 , validation_steps = steps_valid
                 , epochs = 100
                 , callbacks = callbacks )

save_model(cnn)


# ## Performance do modelo

# In[98]:


cnn.evaluate_generator( generator = base_valid
                      , steps = steps_valid )


# ## Realiza previsões na base de teste

# In[99]:



steps_test = base_test.n // base_test.batch_size

base_test.reset()

pred = cnn.predict_generator( base_test
                            , steps = steps_test 
                            , verbose = 1)

predicted_class_indices = np.argmax(pred,axis=1)

labels = (base_train.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

res = list( map (lambda x: (x[0][0] , x[1][-5] ) , list( zip( predictions , base_test.filenames ) )) )

acertos = list(filter(lambda x : x[0].lower() == x[1].lower() , res ))

print("Acc: " +  str(len(acertos)/len(res)) + " |  Acertos: " + str(len(acertos)) + " de " + str(len(res))  )
pprint( res  )


# # PARTE 2 : Previsão em uma nova imagem
# 
# ## Leitura da imagem e quebra em letras

# In[137]:


captcha_img = Image.open(captcha_file)

for i in range(6):			  
    captcha_img.crop(( 
			CUTPOINT["left"][i] , 
			CUTPOINT["top"] , 
			CUTPOINT["left"][i] + LETRA_DIM[0] , 
			CUTPOINT["top"] + LETRA_DIM[1] )
		) \
		.save(letra_template.format(ix=str(i)))

base_test  = test_gen     .flow_from_directory(  prediction_path
                          , target_size = (29,34)
                          , batch_size = 6
                          , color_mode = 'grayscale'
                          , class_mode = None
                          , shuffle = False
                          , seed = 10 )

base_test.reset()

pred = cnn.predict_generator( base_test
                            , steps = 1 
                            , verbose = 1)

predicted_class_indices = np.argmax(pred,axis=1)

labels = (base_train.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [ labels[k][0] for k in predicted_class_indices ]

predictions = "".join( predictions )

print ( predictions )

# Apaga imagens das letras
for i in range(6):
    os.unlink(letra_template.format(ix=str(i)))

