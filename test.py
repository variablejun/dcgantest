import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
starttime = time.time()

train_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory('../_data/cropedmbti/cropinfjM',target_size=(150,150),batch_size=128
,shuffle=False,class_mode='categorical') 


'''

'''

print(xy_train.shape)

'''
np.save('./_npy/x_train.npy',arr=xy_train[0][0])
np.save('./_npy/y_train.npy',arr=xy_train[0][1])
np.save('./_npy/x_test.npy',arr=xy_test[0][0])
np.save('./_npy/y_test.npy',arr=xy_test[0][1])
'''
end = time.time()- starttime
print("걸린시간", end)
