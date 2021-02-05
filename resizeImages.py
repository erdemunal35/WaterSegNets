import os, sys
from PIL import Image

# Download the main water dataset from https://www.kaggle.com/gvclsu/water-segmentation-dataset
# change 'size' variable to your own choice of resolution
size = 256, 256

dirName = "dataset_reshaped_" + str(size[0])+"-"+str(size[1])

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
    images_dir = "dataset/JPEGImages/"
    annotations_dir = "dataset/Annotations/"

    folders = ["aberlour", "auldgirth", "bewdley", "boston_harbor2_small_rois", 
    "buffalo0_small", "canal0", "cockermouth", "dublin", "evesham-lock", "gulf_crest", "holiday_inn_clip0", "holmrook", "keswick_greta", "mexico_beach_clip0", 
    "stream0", "stream1", "stream3_small", "worcester", "galway-city"]

    #stream2 is not included because of 'bad dataset'
    jpg_folders = ["aberlour", "auldgirth", "bewdley", "boston_harbor2_small_rois", "cockermouth", "dublin", 
    "evesham-lock", "galway-city", "holmrook", "keswick_greta", "worcester"]

    png_folders = ["buffalo0_small", "canal0", "gulf_crest", "holiday_inn_clip0", "mexico_beach_clip0", "stream0", "stream1", "stream3_small"] 

    os.mkdir(dirName+"/train/")
    os.mkdir(dirName+"/trainannot/")
    os.mkdir(dirName+"/val/")
    os.mkdir(dirName+"/valannot/")
    os.mkdir(dirName+"/test/")
    os.mkdir(dirName+"/testannot/")

    x_train_dir = os.path.join(dirName, 'train')
    y_train_dir = os.path.join(dirName, 'trainannot')

    x_valid_dir = os.path.join(dirName, 'val')
    y_valid_dir = os.path.join(dirName, 'valannot')

    x_test_dir = os.path.join(dirName, 'test')
    y_test_dir = os.path.join(dirName, 'testannot')

    #train - #trainannot
    trainIds = os.listdir(images_dir + "ADE20K")
    trainAnnotIds = os.listdir(annotations_dir + "ADE20K")
    trainIds2 = os.listdir(images_dir + "river_segs")
    trainAnnotIds2 = os.listdir(annotations_dir + "river_segs")

    trainImages = [os.path.join(images_dir + "ADE20K", image_id) for image_id in trainIds]
    trainAnnotImages = [os.path.join(annotations_dir + "ADE20K", annot_id) for annot_id in trainAnnotIds]
    trainImages2 = [os.path.join(images_dir + "river_segs", image_id) for image_id in trainIds2]
    trainAnnotImages2 = [os.path.join(annotations_dir + "river_segs", annot_id) for annot_id in trainAnnotIds2]

    trainID = 0
    trainMaskID = 0
    for image in trainImages:
        try:
            im = Image.open(image)
            width, height = im.size
            if(width>=256 and height>=256):
                im = im.resize(size, Image.ANTIALIAS)
                im.save(x_train_dir + "/train"+str(trainID)+".png", "PNG")
                trainID += 1
        except IOError:
            print("Error1 occured")

    for image in trainAnnotImages:
        try:
            im = Image.open(image)
            width, height = im.size
            if(width>=256 and height>=256):
                im = im.resize(size, Image.ANTIALIAS)
                im.save(y_train_dir + "/train"+str(trainMaskID)+".png", "PNG")
                trainMaskID += 1
        except IOError:
            print("Error1 occured")

    print("ade20k finished")

    for image in trainImages2:
        try:
            im = Image.open(image)
            width, height = im.size
            if(width>=256 and height>=256):
                im = im.resize(size, Image.ANTIALIAS)
                im.save(x_train_dir + "/train"+str(trainID)+".png", "PNG")
                trainID += 1
        except IOError:
            print("Error2 occured")

    for image in trainAnnotImages2:
        try:
            im = Image.open(image)
            width, height = im.size
            if(width>=256 and height>=256):
                im = im.resize(size, Image.ANTIALIAS)
                im.save(y_train_dir + "/train"+str(trainMaskID)+".png", "PNG")
                trainMaskID += 1
        except IOError:
            print("Error1 occured")

    print("river_segs finished")

    #val - valannot
    valImageID = 0
    valMaskID = 0
    for folder in folders:
        valIds = os.listdir(images_dir + folder)
        valAnnotIds = os.listdir(annotations_dir + folder)
        valImages = [os.path.join(images_dir + folder, image_id) for image_id in valIds]
        valAnnotImages = [os.path.join(annotations_dir + folder, annot_id) for annot_id in valAnnotIds]   

        for image in valImages:
            try:
                im = Image.open(image)
                width, height = im.size
                if(width>=256 and height>=256):
                    im = im.resize(size, Image.ANTIALIAS)
                    im.save(x_valid_dir + "/val"+str(valImageID)+".png", "PNG")
                    valImageID += 1
            except IOError:
                print("Error1 occured")

        for image in valAnnotImages:
            try:
                im = Image.open(image)
                width, height = im.size
                if(width>=256 and height>=256):
                    im = im.resize(size, Image.ANTIALIAS)
                    im.save(y_valid_dir + "/val"+str(valMaskID)+".png", "PNG")
                    valMaskID += 1
            except IOError:
                print("Error1 occured")
        print(folder, " finished.")

    # #test - testannot
    # testImageID = 0
    # testMaskID = 0
    # for folder in jpg_folders:
    #     testIds = os.listdir(images_dir + folder)
    #     testAnnotIds = os.listdir(annotations_dir + folder)
    #     testImages = [os.path.join(images_dir + folder, image_id) for image_id in testIds]
    #     testAnnotImages = [os.path.join(annotations_dir + folder, annot_id) for annot_id in testAnnotIds]   

    #     for image in testImages:
    #         try:
    #             im = Image.open(image)
    #             width, height = im.size
    #             if(width>=256 and height>=256):
    #                 im = im.resize(size, Image.ANTIALIAS)
    #                 im.save(x_test_dir + "/test"+str(testImageID)+".png", "PNG")
    #                 testImageID += 1
    #         except IOError:
    #             print("Error1 occured")

    #     for image in testAnnotImages:
    #         try:
    #             im = Image.open(image)
    #             width, height = im.size
    #             if(width>=256 and height>=256):
    #                 im = im.resize(size, Image.ANTIALIAS)
    #                 im.save(y_test_dir + "/test"+str(testMaskID)+".png", "PNG")
    #                 testMaskID += 1
    #         except IOError:
    #             print("Error1 occured")
    #     print(folder, " finished.")
else:    
    print("Directory " , dirName ,  " already exists")




