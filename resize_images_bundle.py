from PIL import Image
import os, sys

#defining the path to check for images
path = "/afs/crc.nd.edu/user/s/swallen2/msh/msh_tanzania_data/catagorize/"
dirs = os.listdir(path)

#path for resized pictures
#path2 = "/home/swallen2/Tensorflow_caffenet/TestingImages/ImageResized/"

#defining the resize function to resize images in folder
def resize():
    #looping over all the pictures in specified folder
    for pic in dirs:
        if os.path.isfile(path+pic):
            try:
                #Load an image from the hard drive
                original = Image.open(path+pic)

                #Find initial size
                #print ("The size of the Image is: ")
                #print(original.format, original.size, original.mode)

                #Resizing the image to be 227 pixels width (keeping ratio).
                #basewidth = 227
                #wpercent = (basewidth / float(original.size[0]))
                #hsize = int((float(original.size[1]) * float(wpercent)))
                #im2 = original.resize((basewidth, hsize), Image.ANTIALIAS)
                
                #for square images
                size = (227, 227)
                im2 = original.resize((size), Image.ANTIALIAS)
                
                #giving ability to generate new image name/folder to save
                f, e = os.path.splitext(path+pic)

                #print f
                im2.save(f + ".png")

                #Check sizing
                #print ("The size of the resized Image is: ")
                #print(im2.format, im2.size, im2.mode)

            except:
                print ("Unable to load image:" + pic)

resize()
