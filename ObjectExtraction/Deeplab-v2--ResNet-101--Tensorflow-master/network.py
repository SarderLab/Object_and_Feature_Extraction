import tensorflow as tf
import numpy as np
import six

class ResNet_segmentation(object):
	"""
	Original ResNet-101 ('resnet_v1_101.ckpt')
	Original ResNet-50 ('resnet_v1_50.ckpt')
	"""
	def __init__(self, inputs, num_classes, phase, encoder_name):
		if encoder_name not in ['res101', 'res50']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50")
			sys.exit(-1)
		self.encoder_name = encoder_name
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3
		self.phase = phase # train (True) or test (False), for BN layers in the decoder
		self.build_network()

	def build_network(self):
		self.encoding = self.build_encoder()
		self.outputs = self.build_panoptic_decoder(self.encoding)

	def build_encoder(self):
		print("-----------build encoder: %s-----------" % self.encoder_name)
		scope_name = 'resnet_v1_101' if self.encoder_name == 'res101' else 'resnet_v1_50'
		with tf.variable_scope(scope_name) as scope:
			outputs = self._start_block('conv1')
			print("after start block:", outputs.shape)
			with tf.variable_scope('block1') as scope:
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_1',	identity_connection=False)
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
				print("after block1:", outputs.shape)

				self.sem_decoder_recovery_2=self._conv2d(outputs,1,32,1,'sem_dec_rec_2')
				self.ins_decoder_recovery_2=self._conv2d(outputs,1,16,1,'ins_dec_rec_2')

			with tf.variable_scope('block2') as scope:
				outputs = self._bottleneck_resblock(outputs, 512, 'unit_1',	half_size=True, identity_connection=False)
				for i in six.moves.range(2, 5):
					outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)

				self.sem_decoder_recovery_1=self._conv2d(outputs,1,64,1,'sem_dec_rec_1')
				self.ins_decoder_recovery_1=self._conv2d(outputs,1,32,1,'ins_dec_rec_1')

				print("after block2:", outputs.shape)
			with tf.variable_scope('block3') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1',	identity_connection=False)
				num_layers_block3 = 23 if self.encoder_name == 'res101' else 6
				for i in six.moves.range(2, num_layers_block3+1):
					outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
				print("after block3:", outputs.shape)
			with tf.variable_scope('block4') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
				print("after block4:", outputs.shape)
				return outputs

	def build_panoptic_decoder(self, encoding):
		print("-----------building panoptic decoder-----------")
		with tf.variable_scope('decoder') as scope:
			print("-----------building semantic decoder-----------")

			outputs_s = self._ASPPV3(encoding,256, [6, 12, 18, 24],'semanticASPPV3')
			print("after ASPPV3:", outputs_s.shape)

			outputs_s=self._conv2d(outputs_s,1,256,1,'sempw1')
			outputs_s=self._batch_norm(outputs_s, name='sempw1_BN', is_training=self.phase, activation_fn=tf.nn.relu)


			c=[]
			c.append(outputs_s)
			c.append(self.sem_decoder_recovery_1)
			outputs_s=self._concat_channels(c,name='semantic_recov_1')
			print("after decoder recovery 1:", outputs_s.shape)
			outputs_s=self._depthwise_separable_conv2d(outputs_s,5,256,1,'semdw1',1)
			outputs_s=self._batch_norm(outputs_s, name='sempw2_BN', is_training=self.phase, activation_fn=tf.nn.relu)



			#d_up=outputs_s.shape

			d_up2=tf.shape(self.sem_decoder_recovery_2)

			outputs_s=tf.image.resize_images(outputs_s,(d_up2[1],d_up2[2]))

			c=[]
			c.append(outputs_s)
			c.append(self.sem_decoder_recovery_2)
			outputs_s=self._concat_channels(c,name='semantic_recov_2')

			print("after decoder recovery 2:", outputs_s.shape)
			outputs_s=self._depthwise_separable_conv2d(outputs_s,5,256,1,'semdw2',1)
			outputs_s=self._batch_norm(outputs_s, name='sempw3_BN', is_training=self.phase, activation_fn=tf.nn.relu)

			outputs_s=self._depthwise_separable_conv2d(outputs_s,5,256,1,'semdw3',1)
			outputs_s=self._batch_norm(outputs_s, name='sempw4_BN', is_training=self.phase, activation_fn=tf.nn.relu)
			print("Semantic feature maps:", outputs_s.shape)
			outputs_s=self._conv2d(outputs_s,1,self.num_classes,1,'pwclassout')
			print("after final three convolutions:", outputs_s.shape)

			print("-----------building instance decoder-----------")
			outputs_i = self._ASPPV3(encoding,256, [6, 12, 18, 24],'instanceASPPV3')
			print("after ASPPV3:", outputs_i.shape)
			outputs_i=self._conv2d(outputs_i,1,256,1,'inspw1')
			outputs_i=self._batch_norm(outputs_i, name='inspw1_BN', is_training=self.phase, activation_fn=tf.nn.relu)
			c=[]
			c.append(outputs_i)
			c.append(self.ins_decoder_recovery_1)
			outputs_i=self._concat_channels(c,name='instance_recov_1')
			print("after decoder recovery 1:", outputs_i.shape)
			outputs_i=self._depthwise_separable_conv2d(outputs_i,5,128,1,'insdw1',1)
			outputs_i=self._batch_norm(outputs_i, name='inspw2_BN', is_training=self.phase, activation_fn=tf.nn.relu)

			#d_up=outputs_i.shape
			d_up2=tf.shape(self.ins_decoder_recovery_2)

			outputs_i=tf.image.resize_images(outputs_i,[d_up2[1],d_up2[2]])
			c=[]
			c.append(outputs_i)
			c.append(self.ins_decoder_recovery_2)
			outputs_i=self._concat_channels(c,name='instance_recov_2')
			print("after decoder recovery 2:", outputs_i.shape)
			outputs_i=self._depthwise_separable_conv2d(outputs_i,5,128,1,'insdw2',1)
			outputs_i=self._batch_norm(outputs_i, name='inspw3_BN', is_training=self.phase, activation_fn=tf.nn.relu)

			#Center prediction head
			outputs_ic=self._depthwise_separable_conv2d(outputs_i,5,32,1,'inscdw',1)
			outputs_ic=self._batch_norm(outputs_ic, name='inspw4_c_BN', is_training=self.phase, activation_fn=tf.nn.relu)
			print("instance center feature map:", outputs_ic.shape)

			outputs_ic=self._conv2d(outputs_ic,1,1,1,'inscenterout')
			print("instance center output:", outputs_ic.shape)

			#Angular prediction head
			outputs_ia=self._depthwise_separable_conv2d(outputs_i,5,32,1,'insangledw',1)
			outputs_ia=self._batch_norm(outputs_ia, name='inspw4_a_BN', is_training=self.phase, activation_fn=tf.nn.relu)
			print("instance angle feature map:", outputs_ia.shape)
			outputs_ia=self._conv2d(outputs_ia,1,1,1,'insangleout')
			print("instance angle out:", outputs_ia.shape)

			outputs=[outputs_s,outputs_ia,outputs_ic]
			return outputs

	# blocks
	def _start_block(self, name):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name=name)
		outputs = self._batch_norm(outputs, name=name, is_training=self.phase, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=self.phase, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=self.phase, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=self.phase, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=self.phase, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=self.phase, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=self.phase, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=self.phase, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=self.phase, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations,identifier):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i+1), biased=True))
		return self._add(o, name='aspp_'+identifier+'/add')

	def _ASPPV3(self, x, num_o, dilations,identifier):
		pwconv=self._conv2d(x,1,num_o,1,identifier+'/'+'ASPPpwconv')
		pwconv=self._batch_norm(pwconv, name=identifier+'/'+'ASPPpwconvBN', is_training=self.phase, activation_fn=tf.nn.relu)
		imgpool=tf.reduce_mean(x, axis=[3],keepdims=True)

		o = []
		o.append(pwconv)
		o.append(imgpool)
		for i, d in enumerate(dilations):
			c=self._dilated_conv2d(x, 3, num_o, d, name=identifier+'/conv%d' % (i+1), biased=True)
			c=self._batch_norm(c,name=identifier+'/conv%d_BN' % (i+1),is_training=self.phase,activation_fn=tf.nn.relu)
			o.append(c)
		return self._concat_channels(o, name='asppV3_'+identifier+'/concat')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o
	def _depthwise_separable_conv2d(self, x, kernel_size, num_o, stride, name,dilations, biased=False):
		"""
		transpose conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w_d = tf.get_variable('weights_depthwise', shape=[kernel_size, kernel_size, num_x, 1])
			w_p = tf.get_variable('weights_pointwise', shape=[1, 1, num_x*1, num_o])
			s = [1, stride, stride, 1]
			d= [dilations, dilations]
			o = tf.nn.separable_conv2d(x, w_d,w_p , s, padding='SAME',rate=d,name='sepconv2d')

			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o


	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _concat_channels(self, x_l, name):
		return tf.concat(x_l,self.channel_axis,name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=True):
		# For a small batch size, it is better to keep
		# the statistics of the BN layers (running means and variances) frozen,
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
		with tf.variable_scope(name+'/BatchNorm') as scope:
			o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				is_training=is_training,
				trainable=trainable,
				scope=scope)
			return o
