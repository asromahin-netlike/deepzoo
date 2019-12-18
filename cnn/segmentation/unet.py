from keras.layers import Conv2D,MaxPooling2D,concatenate,UpSampling2D,Input,Dropout
from keras.models import Model

def simple_unet(input_size = (32,32,1),alpha=1,pool_stop=None,classes=1,activation='sigmoid'):
    if(alpha<1/64):
        raise ValueError('first layer kernels smaller than 1')
    pool_stop_list=[2,4,8]
    if(not pool_stop in pool_stop_list and pool_stop!=None):
        raise ValueError('pool_stop not available, available pool_stop:',pool_stop_list)
    inputs = Input(input_size)
    conv1 = Conv2D(int(64*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(int(64*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(int(128*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(int(128*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(int(256*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(int(256*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(int(512*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(int(512*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(1024*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(int(1024*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(int(512*alpha), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(int(512*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(int(512*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    if(pool_stop==2):
        last_conv=Conv2D(classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        return Model(input = inputs, output = last_conv)
        
    up7 = Conv2D(int(256*alpha), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(int(256*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(int(256*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    if(pool_stop==4):
        last_conv=Conv2D(classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        return Model(input = inputs, output = last_conv)
        
    up8 = Conv2D(int(128*alpha), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(int(128*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(int(128*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    if(pool_stop==8):
        last_conv=Conv2D(classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        return Model(input = inputs, output = last_conv)
        
    up9 = Conv2D(int(64*alpha), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(int(64*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(int(64*alpha), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    if(pool_stop==16):
        last_conv=Conv2D(classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        return Model(input = inputs, output = last_conv)
        
        
    conv9 = Conv2D(classes*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(classes, 1, activation = activation)(conv9)

    model = Model(input = inputs, output = conv10)

    return model
    
def pool_block(inputs,kernels=1,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',pool_size=(2,2)):
    conv = Conv2D(kernels, kernel_size, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(inputs)
    conv = Conv2D(kernels, kernel_size, activation = activation, padding = padding, kernel_initializer = kernel_initializer)(conv)
    pool = MaxPooling2D(pool_size=pool_size)(conv)
    return pool
    
def upsample_block(inputs,kernels=1,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',upsample_size=(2,2)):
    out = Conv2D(kernels ,2, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(UpSampling2D(size = upsample_size)(inputs[0]))
    if(len(inputs)>1):
        out = concatenate([out,*inputs[1:]], axis = 3)
    out = Conv2D(kernels, kernel_size, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(out)
    out = Conv2D(kernels, kernel_size, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(out)
    return out
    
def connect_upsample_block(inputs,kernels=1,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',upsample_size=(2,2)):
    out = Conv2D(upsample_size[0]*upsample_size[1]*kernels ,2, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(inputs[0])
    out=Reshape(target_shape=(int(inputs.shape[0])*upsample_size[0],int(inputs.shape[1])*upsample_size[1],kernels))(out)
    if(len(inputs)>1):
        out = concatenate([out,*inputs[1:]], axis = 3)
    out = Conv2D(kernels, kernel_size, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(out)
    out = Conv2D(kernels, kernel_size, activation =activation, padding = padding, kernel_initializer = kernel_initializer)(out)
    return out
    
def simple_depth_unet(input_size = (32,32,1),alpha=1,pool_stop=None,classes=1,activation='sigmoid',depth=4):
    inputs=Input(input_size)
    poolings=[]
    conv_poolings=[]
    for i in range(len(depth)):
        if(i==0):
            out=pool_block(inputs,int(64*2**i*alpha))
        else:
            out=pool_block(out,int(64*2**i**alpha))
        poolings.append(out)
        conv_poolings.append(out.input)
    for j in range(len(depth)):
        step=len(depth)-1-j
        out=upsample_block([out,poolings[step],int(64*step*alpha))
    out = Conv2D(classes*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(out)
    out = Conv2D(classes, 1, activation = activation)(out)
    model=Model(inputs=[inputs],outputs=[out])
    return model
    
def connected_unet_depth(input_size = (32,32,1),alpha=1,pool_stop=None,classes=1,activation='sigmoid',depth=4):
    inputs=Input(input_size)
    poolings=[]
    conv_poolings=[]
    for i in range(len(depth)):
        if(i==0):
            out=pool_block(inputs,int(64*2**i*alpha))
        else:
            out=pool_block(out,int(64*2**i**alpha))
        poolings.append(out)
        conv_poolings.append(out.input)
    for j in range(len(depth)):
        step=len(depth)-1-j
        out=connect_upsample_block([out,poolings[step],int(64*step*alpha))
    out = Conv2D(classes*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(out)
    out = Conv2D(classes, 1, activation = activation)(out)
    model=Model(inputs=[inputs],outputs=[out])
    return model