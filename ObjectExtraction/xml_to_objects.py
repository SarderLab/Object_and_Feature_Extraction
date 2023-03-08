import numpy as np
import os
import cv2
import openslide
import lxml.etree as ET
import sys
import argparse
import warnings
import time
import multiprocessing
import pandas as pd
from glob import glob
from imageio import imwrite
from subprocess import call
from xml_to_mask_6 import xml_to_mask
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
from matplotlib import path
from joblib import Parallel, delayed
from itertools import chain

cwd = os.getcwd() + '/'
# Regions with fewer pixels than size_thresh are not included
size_thresh = 800
# Amount to pad each image dimension around glomerular boundary in extracted images
pad_width=400
#Image extension
imBoxExt='.jpeg'
#Image extension
chop_box_size=500

#Which device to use
gpu_id='0'

#Directory of tensorflow model and model checkpoint number
model_dir='/blue/pinaki.sarder/nlucarelli/ObjectExtraction/model'
nuc_ckpt=351744
overlap_percent=0
# Where the deeplab folder is located
deeplab_dir=cwd+'/Deeplab-v2--ResNet-101--Tensorflow-master'


WSIs = []
XMLs = []
# Class ID
annot_ID=2
classes=['Cortical_Interstitium','Glomeruli','Sclerotic_glomeruli','Abnormal_tubule','Vessel']


def return_region(wsi_mask, wsiID, fileID, yStart, xStart, outdirT, region_size): # perform cutting in parallel
    uniqID=fileID +'_'+ str(yStart) +'_'+ str(xStart)

    slide=getWsi(wsiID)
    Im=np.array(slide.read_region((xStart,yStart),0,(region_size,region_size)))
    Im=Im[:,:,:3]

    mask_annotation=wsi_mask[yStart:yStart+region_size,xStart:xStart+region_size]

    o1,o2=mask_annotation.shape
    if o1 !=region_size:
        mask_annotation=np.pad(mask_annotation,((0,region_size-o1),(0,0)),mode='constant')
    if o2 !=region_size:
        mask_annotation=np.pad(mask_annotation,((0,0),(0,region_size-o2)),mode='constant')


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.sum(mask_annotation)!=0:
            imwrite(outdirT + basename + '/'+classes[annot_ID-1] + '/Images/' +uniqID + imBoxExt,Im)
            imwrite(outdirT + basename + '/'+classes[annot_ID-1] + '/Boundary_segmentations/' +uniqID+'_mask.png',mask_annotation)



def get_annotation_bounds(xml_path, annotationID=1):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Find listed regions
    Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")

    masks = []
    extremas=[]
    # Create padded mask and identify boundary extremas for all regions
    for Region in Regions:
        Vertices = Region.findall("./Vertices/Vertex")
        x = []
        y = []

        for Vertex in Vertices:
            x.append(int(np.float32(Vertex.attrib['X'])))
            y.append(int(np.float32(Vertex.attrib['Y'])))

        x1=min(x)
        x2=max(x)
        y1=min(y)
        y2=max(y)
        points = np.stack([np.asarray(x), np.asarray(y)], axis=1)

        points[:,1] = np.int32(np.round(points[:,1] - y1 ))
        points[:,0] = np.int32(np.round(points[:,0] - x1 ))

        mask = np.zeros([(y2-y1),x2-x1], dtype=np.int8)

        # Fill mask boundary regions
        cv2.fillPoly(mask, [points], 1)
        mask=np.pad( mask,(pad_width,pad_width),'constant',constant_values=(0,0) )

        masks.append(mask)
        extremas.append([x1,x2,y1,y2])
    return masks,extremas

def get_useable_bounds(xml_path, annotationID=6):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Find listed regions
    Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")

    masks = []
    extremas=[]
    # Create padded mask and identify boundary extremas for all regions
    region_vertices=[]
    for Region in Regions:
        Vertices = Region.findall("./Vertices/Vertex")
        vert=[]

        for Vertex in Vertices:
            vert.append([int(np.float32(Vertex.attrib['X'])),int(np.float32(Vertex.attrib['Y']))])


        region_vertices.append(vert)
    return region_vertices
def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    import openslide
    slide = openslide.OpenSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Get the information input from the user
parser = argparse.ArgumentParser()
parser.add_argument('--wsi', dest='wsi', default=' ',type=str, help='Specifies the whole slide folder path.')
parser.add_argument('--output', dest='output_directory_name', default=' ',type=str, help='Directory to save output results.')
parser.add_argument('--ext', dest='wsi_ext', default='.svs',type=str, help='Directory to save output results.')
parser.add_argument('--chop_data', dest='chop_data', default='True',type=str, help='Directory to save output results.')
args = parser.parse_args()



# Make sure user gives information
if args.output_directory_name==' ':
    print('No output directory provided, using default directory at: '+'\n')
    output_directory_name='Output/'
    outDir=cwd+output_directory_name
else:
    outDir=args.output_directory_name+'/'

# check main directory exists, if not, make it
make_folder(outDir)

# Make sure user gives information
if args.wsi == ' ':
    print('\nPlease specify the whole slide folder path.\n\nUse flag:')
    print('--wsi <path>\n')
    sys.exit()
# Get list of all whole slide images
WSIs_ = []#glob(args.wsi+'/*'+args.wsi_ext)
usable_ext=args.wsi_ext.split(',')
for ext in usable_ext:
    WSIs_.extend(glob(args.wsi + '/*' + ext))

print(len(WSIs_))
bboxes = []
im_names = []

for WSI in WSIs_:
    xml_ = glob(WSI.split('.')[0] + '.' + WSI.split('.')[1] + '.xml')
    if xml_ != []:
        print('including: ' + WSI)
        XMLs.append(xml_[0])
        WSIs.append(WSI)

if args.chop_data=='True':
    # go though all WSI
    for idx, XML in enumerate(XMLs):

        # Generate mask region and boundary maxima from XML annotation
        masks,extremas = get_annotation_bounds(XML,annot_ID)
        basename = os.path.basename(XML)
        basename = os.path.splitext(basename)[0]
        # Create output folders
        make_folder(outDir + basename+'/'+classes[annot_ID-1]+'/Boundary_segmentations')
        make_folder(outDir + basename+'/'+classes[annot_ID-1]+ '/Images')
        # Open wholeslide image data
        print('opening: ' + WSIs[idx]+'\n')
        pas_img = openslide.OpenSlide(WSIs[idx])
        dim_x,dim_y=pas_img.dimensions
        tree = ET.parse(XML)
        root = tree.getroot()
        # Find listed regions
        layer_6_regions = root.findall("./Annotation[@Id='" + str(6) + "']/Regions/Region")
        if len(layer_6_regions)!=0:
            useable_regions=get_useable_bounds(XML)
            useable_paths=[]
            for useable_region in useable_regions:
                useable_paths.append(path.Path(useable_region))

        #For instance objects we extract separate objects directly
        if annot_ID in [2,3,4,5]:


            # For all discovered regions in XML
            for idxx, ex in enumerate(extremas):
                mask = masks[idxx]
                size=np.sum(mask)
                if size >= size_thresh:
                    # Pull image from WSI
                    #x1x2y1y2
                    c_1=ex[0]-pad_width
                    c_2=ex[2]-pad_width
                    l_1=(ex[1]+pad_width)-c_1
                    l_2=(ex[3]+pad_width)-c_2
                    if len(layer_6_regions)!=0:

                        p1=[ex[0],ex[2]]
                        p2=[ex[1],ex[2]]
                        p3=[ex[0],ex[3]]
                        p4=[ex[1],ex[3]]
                        useable=True
                        for useable_path in useable_paths:
                            if useable_path.contains_point(p1):
                                useable=False
                                break
                            elif useable_path.contains_point(p2):
                                useable=False
                                break
                            elif useable_path.contains_point(p3):
                                useable=False
                                break
                            elif useable_path.contains_point(p4):
                                useable=False
                                break
                    else:
                        useable=True

                    if useable:
                        PAS = pas_img.read_region((c_1,c_2), 0, (l_1,l_2))
                        PAS = np.array(PAS)[:,:,0:3]
                        if np.sum(PAS)==0:
                            continue
                        else:
                            # print(basename + '_' + str(idxx))
                            # Save image and mask
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                tempBox =[]
                                tempBox.append(ex[0])
                                tempBox.append(ex[1])
                                tempBox.append(ex[2])
                                tempBox.append(ex[3])
                                bboxes.append(tempBox)
                                im_names.append(basename + '_' + str(idxx))
                                imwrite(outDir + basename+ '/'+classes[annot_ID-1] + '/Images/'+ basename + '_' + str(idxx) + imBoxExt,PAS)
                                # imwrite(outDir+basename+'_'+str(idxx)+imBoxExt,PAS)
                                imwrite(outDir + basename +'/'+classes[annot_ID-1] + '/Boundary_segmentations/' + basename + '_' + str(idxx) + '_mask'+'.png', mask*255)
                                # imwrite(outDir + basename + '_' + str(idxx) + '.png',mask*255)

        #For non-instance objects we extract patches
        else:
            stepHR = int(chop_box_size*(1-overlap_percent))
            if basename=='K1300466_6_PAS_05082017_001':

                wsi_mask=xml_to_mask(XML, [0,0], [87519,44938],use_six=False)
                wsi_mask=wsi_mask[:,0:36000]
                wsi_mask_6=xml_to_mask(XML, [0,0], [87519,44938],use_six=True)
                wsi_mask_6=wsi_mask_6[:,0:36000]

            elif basename=='K1300473_4_PAS_05082017_001_003':
                wsi_mask=xml_to_mask(XML, [0,0], [128600,46112],use_six=False)
                wsi_mask=wsi_mask[:,0:60000]
                wsi_mask_6=xml_to_mask(XML, [0,0], [128600,46112],use_six=True)
                wsi_mask_6=wsi_mask_6[:,0:60000]

            else:
                wsi_mask=xml_to_mask(XML, [0,0], [dim_x,dim_y],use_six=False)
                wsi_mask_6=xml_to_mask(XML, [0,0], [dim_x,dim_y],use_six=True)

            labeled_regions,num_regions=label(wsi_mask==annot_ID)
            props=regionprops(labeled_regions)
            num_cores = multiprocessing.cpu_count()

            for prop_idx,prop in enumerate(props):
                if len(layer_6_regions)!=0:
                    #x1x2y1y2
                    p1=[prop.bbox[1],prop.bbox[0]]
                    p2=[prop.bbox[1],prop.bbox[2]]
                    p3=[prop.bbox[3],prop.bbox[0]]
                    p4=[prop.bbox[3],prop.bbox[2]]
                    useable=False
                    for useable_path in useable_paths:
                        if useable_path.contains_point(p1):
                            useable=True
                            break
                        elif useable_path.contains_point(p2):
                            useable=True
                            break
                        elif useable_path.contains_point(p3):
                            useable=True
                            break
                        elif useable_path.contains_point(p4):
                            useable=True
                            break
                else:
                    useable=True


                if useable:
                    w=int(prop.bbox[3]-prop.bbox[1])
                    h=int(prop.bbox[2]-prop.bbox[0])
                    if (w*h) >= size_thresh:

                        if (w*h)>(chop_box_size*chop_box_size):

                            index_yHR=np.arange(prop.bbox[0],prop.bbox[2],stepHR)
                            index_xHR=np.arange(prop.bbox[1],prop.bbox[3],stepHR)
                            index_yHR[-1]=prop.bbox[2]-stepHR
                            index_xHR[-1]=prop.bbox[3]-stepHR

                            for j in index_yHR:
                                for i in index_xHR:
                                    sys.stdout.write('   <'+str(prop_idx)+':'+str(j)+':'+str(i)+ '>   ')
                                    sys.stdout.flush()
                                    restart_line()
                                    return_region(wsi_mask=wsi_mask, wsiID=WSIs[idx],
                                        fileID=basename+'_'+str(prop_idx), yStart=j, xStart=i, outdirT=outDir,
                                        region_size=chop_box_size)
                            #Parallel(n_jobs=num_cores)(delayed(return_region)(wsi_mask=wsi_mask, wsiID=WSIs[idx],
                            #    fileID=basename+'_'+str(idx), yStart=j, xStart=i, outdirT=outDir,
                            #    region_size=chop_box_size) for idxx,i in enumerate(index_xHR) for idxy,j in enumerate(index_yHR))
                        else:

                            xStart=prop.bbox[1]
                            yStart=prop.bbox[0]
                            sys.stdout.write('   <'+str(prop_idx)+':'+str(yStart)+':'+str(xStart)+ '>   ')
                            sys.stdout.flush()
                            restart_line()
                            uniqID=basename+'_'+str(prop_idx) +'_'+ str(yStart)+'_' + str(xStart)


                            Im=np.array(pas_img.read_region((xStart,yStart),0,(w,h)))
                            Im=Im[:,:,:3]

                            mask_annotation=wsi_mask[yStart:yStart+h,xStart:xStart+w]

                            o1,o2=mask_annotation.shape
                            if o1 !=h:
                                mask_annotation=np.pad(mask_annotation,((0,h-o1),(0,0)),mode='constant')
                            if o2 !=w:
                                mask_annotation=np.pad(mask_annotation,((0,0),(0,w-o2)),mode='constant')


                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                imwrite(outDir + basename + '/'+classes[annot_ID-1] + '/Images/' +uniqID + imBoxExt,Im)
                                imwrite(outDir + basename + '/'+classes[annot_ID-1] + '/Boundary_segmentations/' +uniqID+'_mask'+'.png',mask_annotation)


        pas_img.close()

# Begin nuclear prediction
# Get list of all cases generated by the xml extraction

# im_names = np.array(im_names)
# im_names = np.expand_dims(im_names,-1)
# bboxes = np.array(bboxes)
# im_names = pd.DataFrame(im_names)
# bboxes = pd.DataFrame(bboxes)
# combo = pd.concat([im_names,bboxes],axis=1)
# combo.columns = ['Im_name','x1','x2','y1','y2']
# combo.to_csv('FFPE_tubule_bboxes.csv')
# exit()

input_folder_list=glob(outDir+'/*'+os.path.sep)

outDir_split=outDir.split('/')
popidx=[]
for i,s in enumerate(outDir_split):
    if s=='':
        popidx.append(i)
for p in reversed(popidx):
    outDir_split.pop(p)
outDir_abbrev=outDir_split[-1]
outDir_prev='/'+'/'.join(outDir_split[:-1])
txt_loc=deeplab_dir+'/dataset/test.txt'
f=open(txt_loc,'w')
f=open(txt_loc,'a')

for case_folder in input_folder_list:

    # Get case images
    print(case_folder)
    #continue
    #if case_folder !=input_folder_list[-1]:
    #    continue
    case_folder=case_folder.split('/')[-2]
    images_for_prediction=glob(outDir+'/'+case_folder+'/'+classes[annot_ID-1]+'/Images/*.jpeg')
    # Make output folder
    make_folder(outDir+'/'+case_folder+'/'+classes[annot_ID-1]+'/Nuclear_segmentations/prediction')
    # Where to save test image names for DeepLab prediction
    already_predicted = glob(outDir+'/'+case_folder+'/'+classes[annot_ID-1]+'/Nuclear_segmentations/prediction/*.png')


    glomIDs = []
    for glom in already_predicted:
        currID = glom.split('.')[0].split('/')[-1].split('_mask')[0]
        glomIDs.append(currID)

    # Write image names to text
    for image in images_for_prediction:
        im_splits=image.split('/')
        file_ID=im_splits[-1]

        if file_ID.split('.jpeg')[0] in glomIDs:
            continue
        #folder='/'+im_splits[-2]+'/'
        folder='/'.join(im_splits[-4:-1])
        f.write('/'+outDir_abbrev+'/'+folder+'/'+file_ID+'\n')
f.close()
f=open(txt_loc,'r')
num_steps=len(f.readlines())
print('Number of images for prediction is: '+str(num_steps))
# Call deeplab for prediction
call(['python3', deeplab_dir+'/main.py',
    '--option', 'predict',
    '--test_data_list', txt_loc,
    '--out_dir', '/Nuclear_segmentations/',
    '--test_step', str(nuc_ckpt),
    '--test_num_steps', str(num_steps),
    '--modeldir', model_dir,
    '--data_dir', outDir_prev,
    '--gpu', gpu_id])
