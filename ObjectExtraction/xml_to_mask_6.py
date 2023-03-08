import numpy as np
import sys
import lxml.etree as ET
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,binary_erosion
from skimage.morphology import diamond
import time

def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num = annotation_num + 1

    return annotation_num + 1


"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size

"""

def xml_to_mask(xml_path, location, size, downsample_factor=1, verbose=0,use_six=True):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    IDs,annots = regions_in_mask(root=root, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # fill regions and create mask
    mask = Regions_to_mask(Regions=annots, IDs=IDs,bounds=bounds,use_six=use_six, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def restart_line(): # for printing labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def regions_in_mask(root, bounds, verbose=1):
    # find regions to save

    annots=[]
    annots_IDs=[]
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']

        regs=[]
        reg_IDs=[]
        for Region in Annotation.findall("./*/Region"): # iterate on all region
            reg_IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
            verts_x=[]
            verts_y=[]
            for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                # get points

                x_point = np.int32(np.float64(Vertex.attrib['X']))
                y_point = np.int32(np.float64(Vertex.attrib['Y']))
                # test if points are in bounds
                verts_x.append(x_point)
                verts_y.append(y_point)

                #IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                #break
            regs.append(np.swapaxes(np.array((verts_x,verts_y)),0,1))
        annots_IDs.append(reg_IDs)
        annots.append(regs)

    return annots_IDs, annots

def Regions_to_mask(Regions, IDs,bounds,use_six, verbose=1):

    mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) )), int(np.round((bounds['x_max'] - bounds['x_min']) )) ], dtype=np.int8)
    #mask_temp = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) )), int(np.round((bounds['x_max'] - bounds['x_min']) )) ], dtype=np.int8)

    if verbose !=0:
        print('\nMAKING MASK:')

    for annot_idx,annot in enumerate(Regions):
        annot_id=IDs[annot_idx]
        for reg_idx,reg in enumerate(annot):
            reg_id=int(annot_id[reg_idx]['annotationID'])


            x1 = min(reg[:,0])
            x2 = max(reg[:,0])

            y1 = min(reg[:,1])
            y2 = max(reg[:,1])



            reg_pass=[0,0,0,0]
            if reg_id==4:

                #for v in range(0,len(reg[1])):
                reg[:,0]=reg[:,0]-x1
                reg[:,1]=reg[:,1]-y1

                mask_temp=np.zeros((y2-y1,x2-x1))
                cv2.fillPoly(mask_temp,[reg], reg_id)


                tub_prev=mask[y1:y2,x1:x2]
                overlap=np.logical_and(tub_prev==reg_id,binary_dilation(mask_temp==reg_id,diamond(2)))
                tub_prev[mask_temp==reg_id]=reg_id
                if np.sum(overlap)>0:


                    tub_prev[overlap]=1
                    '''
                    plt.subplot(131),plt.imshow(mask_temp)
                    plt.subplot(132),plt.imshow(tub_prev)
                    plt.subplot(133),plt.imshow(overlap)
                    plt.show()
                    '''



                mask[y1:y2,x1:x2]=tub_prev



            elif reg_id==6:
                if use_six:
                    cv2.fillPoly(mask, [reg], reg_id)
                else:
                    continue
            else:
                cv2.fillPoly(mask, [reg], reg_id)



    return mask
