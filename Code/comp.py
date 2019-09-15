from PIL import Image
import os
path = "C:/Users/Sweg/Desktop/addanno/"
path2= "C:/Users/Sweg/Desktop/reduced2/"
#basewidth = 250
x=0
#img = img.resize((480, 640))
#img.save("C:/Users/Sweg/Desktop/save/kek.png",  "PNG")


for i in os.listdir(path):
    if i != "meow":
        if i != "meow2":
            newpath=path+"/"+i
            img = Image.open(newpath)
            #wpercent = (basewidth / float(img.size[0]))
            #hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((900, 900))#, Image.ANTIALIAS)
            original=path2+i
            meow=original.replace(".jpg", "")
            img.save(meow+"yaas"+".png", "PNG")
            x=x+1
            print(x)



