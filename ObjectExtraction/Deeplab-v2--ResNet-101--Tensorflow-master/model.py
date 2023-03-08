from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image

from network import *
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label
from skimage.morphology import binary_dilation,diamond,watershed,disk,reconstruction,area_opening
from matplotlib import pyplot as plt
from imageio import imwrite
from skimage.feature import peak_local_max
import cv2
"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	def predict(self):
		normal_color = "\033[0;37;40m"
		self.predict_setup()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		# load checkpoint
		checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.test_step)
		self.load(self.loader, checkpointfile)

		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# img_name_list
		image_list, _ = read_labeled_image_list('', self.conf.test_data_list)

		# Predict!
		for step in range(self.conf.test_num_steps):
			#preds,preds2,majority_vote,centers,imb= self.sess.run([self.pred,self.pred_2,self.pred_3,self.centers,self.image_batch])
			majority_vote= self.sess.run([self.pred_3])[0]
			majority_vote[0,:,:]=area_opening(majority_vote[0,:,:],area_threshold=5)

			img_splits=image_list[step].split('/')
			popidx=[]
			for i,s in enumerate(img_splits):
			    if s=='':
			        popidx.append(i)
			for p in reversed(popidx):
			    img_splits.pop(p)

			img_name = img_splits[-1].split('.')[0]
			img_folder='/'.join(img_splits[0:3])+self.conf.out_dir
			print(img_folder)
			img_folder='/blue/pinaki.sarder/nlucarelli/ObjectExtraction/'+img_folder
			# img_folder=os.getcwd()+'/'+img_folder

			if not os.path.exists(img_folder):
				os.makedirs(img_folder)
				os.makedirs(img_folder + '/prediction')
				if self.conf.visual:
					os.makedirs(img_folder + '/visual_prediction')

			im = Image.fromarray(majority_vote[0,:,:], mode='L')
			filename = img_name+'_mask.png'
			im.save(img_folder + 'prediction/' + filename)


			'''
			im = Image.fromarray(preds2[0,:,:,:], mode='RGB')
			filename = '/%s_amask.png' % (img_name)
			im.save(self.conf.out_dir + '/prediction' + filename)
			'''
			if self.conf.visual:
				msk = decode_labels(preds, num_classes=self.conf.num_classes)
				im = Image.fromarray(msk[0], mode='RGB')
				filename = '/%s_mask_visual.png' % (img_name)
				im.save(self.conf.out_dir + '/visual_prediction' + filename)
			'''
			if step>=0:
				preds_3d=np.stack(((preds2[0,:,:,0]*10).astype('uint8'),(centers*150).astype('uint8'),np.zeros(centers.shape).astype('uint8')),axis=2)


				plt.subplot(221),plt.imshow(centers)
				plt.subplot(222),plt.imshow(preds2[0,:,:,0])
				plt.subplot(223),plt.imshow(majority_vote[0,:,:]*50)
				#plt.subplot(223),plt.imshow(majority_vote)
				plt.subplot(224),plt.imshow(imb[0,:,:,:])
				plt.show()
			'''
			if step % 1 == 0:
				print(self.conf.print_color + 'step {:d}'.format(step) + ':' + filename + normal_color)

		print(self.conf.print_color + 'The output files has been saved to {}'.format(self.conf.out_dir) + normal_color)

		# finish
		self.coord.request_stop()
		self.coord.join(threads)

	def predict_setup(self):
		# Create queue coordinator.
		self.coord = tf.train.Coordinator()

		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.test_data_list,
				None,
				False, # no data-aug
				False, # no data-aug
				self.conf.ignore_label,
				IMG_MEAN,
				self.coord,
				False)#no data aug
			image, label = reader.image, reader.label # [h, w, 3 or 1]
		# Add one batch dimension [1, h, w, 3 or 1]
		image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
		self.image_batch=tf.py_func(inv_preprocess, [image_batch, 1, IMG_MEAN], tf.uint8)
		# Create network
		if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50, or deeplab")
			sys.exit(-1)
		elif self.conf.encoder_name == 'deeplab':
			net = Deeplab_v2(image_batch, self.conf.num_classes, False)
		else:
			net = ResNet_segmentation(image_batch, self.conf.num_classes, False, self.conf.encoder_name)

		# Predictions.
		triple_out=net.outputs
		raw_output = triple_out[0] # [batch_size, h, w, 21]
		center_output = triple_out[2]
		angle_output = triple_out[1]


		raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])

		raw_output = tf.argmax(raw_output, axis=3)

		self.pred = tf.cast(tf.expand_dims(raw_output, dim=3), tf.uint8)

		raw_output_up_c = tf.image.resize_bilinear(center_output, tf.shape(image_batch)[1:3,])
		raw_output_up_a = tf.image.resize_bilinear(angle_output, tf.shape(image_batch)[1:3,])


		self.pred_2 = raw_output_up_a
		majority_map,centers=tf.py_func(self.get_majority_votes, [raw_output_up_a,raw_output,raw_output_up_c], [tf.uint8,tf.int32])
		self.centers=centers
		self.pred_3=majority_map
		# Create directory

		# Loader for loading the checkpoint
		self.loader = tf.train.Saver(var_list=tf.global_variables())


	def save(self, saver, step):
		'''
		Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, saver, filename):
		'''
		Load trained weights.
		'''
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))

	def compute_IoU_per_class(self, confusion_matrix):
		mIoU = 0
		for i in range(self.conf.num_classes):
			# IoU = true_positive / (true_positive + false_positive + false_negative)
			TP = confusion_matrix[i,i]
			FP = np.sum(confusion_matrix[:, i]) - TP
			FN = np.sum(confusion_matrix[i]) - TP
			IoU = TP / (TP + FP + FN)
			print ('class %d: %.3f' % (i, IoU))
			mIoU += IoU / self.conf.num_classes
		print ('mIoU: %.3f' % mIoU)

	def angle_transform(self,images,thresh):
		o1,o2,o3=np.shape(images)
		angles_out=np.zeros((self.conf.batch_size,o2,o3,1)).astype('float64')
		for imidx in range(self.conf.batch_size):
			if np.sum(images[imidx,:,:])==0:
				pass
			else:

				for classidx in range(0,self.conf.num_classes):
					pk_sv=np.zeros((o2,o3))
					mask=np.zeros((o2,o3))

					if classidx==0:
						continue
					elif classidx==1:
						continue
					else:
						im=images[imidx,:,:]

						mask[im==classidx]=1
						if np.sum(mask)==0:
							continue
						else:
							dt=(distance_transform_edt(mask))
							'''
							[d1,d2]=np.shape(mask)

							[rr,cc]=np.meshgrid(range(d2),range(d1));

							r=cc-dt[1][0]
							c=rr-dt[1][1]
							r=np.divide(r,dt[0])
							c=np.divide(c,dt[0])
							r[mask==0]=0
							c[mask==0]=0
							r=(r-np.min(r))/(np.max(r)-np.min(r))
							c=(c-np.min(c))/(np.max(c)-np.min(c))
							r[mask==0]=0
							c[mask==0]=0
							angles_out[imidx,:,:,0]+=c
							angles_out[imidx,:,:,1]+=r
							'''
							angles_out[imidx,:,:,0]+=dt
							'''
							lab,num_regs=label(mask)
							if num_regs==1:
								dts=np.copy(dt)
								dts[np.where(lab != 1)]=0
								pk_sv[np.where(dts>thresh*np.max(dts))]=1
							else:
								for lr in range(1,num_regs+1):

									dts=np.copy(dt)
									dts[np.where(lab != lr)]=0
									pk_sv[np.where(dts>thresh*np.max(dts))]=1

							angles_out[imidx,:,:,1]+=pk_sv
							'''

		return np.nan_to_num(angles_out)


	def get_majority_votes(self,dt_map,sem_map,center_map):
		o1,o2,o3=np.shape(sem_map)
		pk_sv=np.zeros((o2,o3))
		pred_out=np.zeros((o1,o2,o3)).astype('uint8')


		sem_mask=sem_map[0,:,:]
		s=np.nonzero(sem_mask==1)
		dt_mask=dt_map[0,:,:,0]>0.1

		#markers = label(center_map[0,:,:,0]>0.8)[0]
		block_size=int(np.floor((2*o2)/16))

		if block_size % 2 == 0:
			block_size+=1
		markers = label(cv2.adaptiveThreshold((center_map[0,:,:,0]*1.5).astype('uint8'),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,-0.001))[0]
		markers=area_opening(markers,area_threshold=5)
		lab = watershed(-1*dt_map[0,:,:,0],markers,mask=dt_mask,watershed_line=True)

		#lab[binary_dilation(lab==0,disk(2))]=0

		num_regs=np.max(lab)

		for lr in range(1,num_regs+1):
			z=np.nonzero(lab==lr)
			ob_sum=np.sum(lab==lr)
			sem_preds=sem_mask[z[0],z[1]]
			sem_accum=0
			class_vote=0
			for classid in np.unique(sem_preds):
				if classid==0:
					continue
				else:
					accum=(np.sum(sem_preds==classid))
					if accum>sem_accum:
						sem_accum=accum
						class_vote=classid
			if sem_accum>=(0.25*ob_sum):
				pred_out[0,z[0],z[1]]=class_vote

		sp_nucs=reconstruction(pred_out[0,:,:],dt_mask)>0
		small_nucs=dt_mask
		small_nucs[sp_nucs]=0

		pred_out[0,small_nucs]=1

		#return (center_map[0,:,:,0]>0.1).astype('uint8')
		return pred_out,markers
