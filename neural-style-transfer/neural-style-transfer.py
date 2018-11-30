import imutils
import time
import cv2
import imageio
import random


mood=random.randint(0,1)


def nst(mood):
    num=random.randint(0,2)
    #print(num)


    image = cv2.imread('images\Saurav.jpg',1)
#print(image)
#print("i")
    if(mood==0):
        if(num==0):
            net = cv2.dnn.readNetFromTorch('models/model_collection/udnie.t7')
        elif(num==1):
            net = cv2.dnn.readNetFromTorch('models/model_collection/the_scream.t7')
        elif(num==2):
            net = cv2.dnn.readNetFromTorch('models/model_collection/starry_night.t7')
    else:
        if(num==0):
            net = cv2.dnn.readNetFromTorch('models/model_collection/la_muse.t7')
        elif(num==1):
            net = cv2.dnn.readNetFromTorch('models/model_collection/composition_vii.t7')
        elif(num==2):
            net = cv2.dnn.readNetFromTorch('models/model_collection/the_wave.t7')
    

# load the input image, resize it to have a width of 600 pixels, and
# then grab the image dimensions
#image = cv2.imread(args["image"])

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

# construct a blob from the image, set the input, and then perform a
# forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

# reshape the output tensor, add back in the mean subtraction, and
# then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    print(output.shape)
# show information on how long inference took
    print("[INFO] neural style transfer took {:.4f} seconds".format(
        end - start))
# show the images
    cv2.imshow("Input", image)
    cv2.imshow("Output", output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    imageio.imwrite('astronaut-gray.png', output[:, :, :])
    cv2.waitKey(0)
    
#nst(mood)
