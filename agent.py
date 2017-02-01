import cPickle as pickle
import numpy as np
import os
class Agent(object):
	def __init__(self, inputs, hidden_neurons, outputs, batch_size, learning_rate, reward_decay, decay_rate, model_file, update_method):

		# Number of inputs
		self.inputs = inputs;

		# Number of hidden neurons
		self.hidden_neurons = hidden_neurons
		
		# Number of outputs
		self.outputs = outputs

		# How many episodes until model updates > 0
		self.batch_size = batch_size

		# How much to update model every batch (0,1)
		self.learning_rate = learning_rate

		# How much should the future rewards be decayed (0,1)
		self.reward_decay = reward_decay

		self.decay_rate = decay_rate

		# Our model, containing weights
		self.model = None



		# Counter of current episode number
		self._episode_nbr = 0

		# Past observation vectors
		self._xs = []
		# Past hidden state vectors
		self._hs = []
		# Past prediction vectors
		self._dlogps = [] 
		# Past rewards
		self._drs = []

		# which update method to use [adagrad, rmsprop]
		self._update_method = update_method

		self.model_folder = "models"
		self.recording_folder = "recordings"

		# Name of file where to save our model
		self.model_file = self.model_folder + "/" + model_file


		# Name of file where to save model recordings
		self.recording_file = self.recording_folder + "/" + model_file


		self.recordings = []



	def load_model( self ):
		if os.path.isfile(self.model_file):
			if( self.model == None ):
				self.model = pickle.load(open(self.model_file, 'rb'))
				print "Model loaded from "+self.model_file

				self.recordings = pickle.load(open(self.recording_file, 'rb'))
			else:
				print "Model already initialized."
				exit()
		else:
			self.new_model()


	def new_model(self):
		if( self.model == None):
			self.model = {}
			self.model['W1'] = np.random.randn(self.inputs, self.hidden_neurons) / np.sqrt(self.inputs) # "Xavier" initialization
			self.model['W2'] = np.random.randn(self.hidden_neurons, self.outputs) / np.sqrt(self.hidden_neurons)


			self.recordings = []
			self.save_model()
			self.save_recordings()
		else:
			print "Model already initialized."
			exit()


	def init(self):
		self.init_gradbuffer()
		self.init_rmscache()
		self.init_adagrad_mem()


	def init_gradbuffer(self):
		self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # update buffers that add up gradients over a batch

	def init_rmscache(self):
		self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # rmsprop memory

	def init_adagrad_mem(self):
		self.adagrad_mem = { k : np.zeros_like(v) for k,v in self.model.iteritems() } 




	def _sigmoid(self,x): 
		return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

	def _softmax(self, x):
		probs = np.exp(x - np.max(x, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		return probs


	def discount_rewards(self, r):
		""" take 1D float array of rewards and compute discounted reward based on reward_decay"""
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(xrange(0, r.size)):
			running_add = running_add * self.reward_decay + r[t]
			discounted_r[t] = running_add
		return discounted_r


	def policy_forward(self, x):
		""" Calculate probabilites of actions based on observation """
		if(len(x.shape)==1):
			x = x[np.newaxis,...]

		h = np.dot(x, self.model['W1'])
		h[h<0] = 0 # ReLU nonlinearity
		logp = np.dot(h, self.model['W2'])
		p = self._softmax(logp)
		return p, h # return probability of taking actions, and hidden state



	def policy_backward(self, eph, epdlogp, epx):
		""" backward pass. (eph is array of intermediate hidden states) """
		dW2 = np.dot(eph.T, epdlogp)

		dh = np.dot(epdlogp, self.model['W2'].T)


		dh[eph <= 0] = 0 # backpro prelu
		dW1 = np.dot(epx.T, dh)
		return {'W1':dW1, 'W2':dW2}


	def decide_action(self, observation):
		probabilites, h = self.policy_forward(observation)
		rand = np.random.uniform(0, np.sum(probabilites))


		aprob_cum = np.cumsum(probabilites)
		a = np.where( rand < aprob_cum)[0][0]



		# record various intermediates (needed later for backprop)
		self._xs.append(observation) # observation
		self._hs.append(h) # hidden state

		dlogsoftmax = probabilites.copy()
		dlogsoftmax[0,a] -= 1
		self._dlogps.append(dlogsoftmax)

		return a


	def register_reward(self, reward):
		self._drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)



	def episode_restarted(self):
		self._xs, self._hs, self._dlogps, self._drs = [],[],[],[] # reset array memory

	def episode_finished(self):
		self._episode_nbr += 1

		epx = 		np.vstack(self._xs)
		eph = 		np.vstack(self._hs)
		epdlogp = 	np.vstack(self._dlogps)
		epr = 		np.vstack(self._drs)

		reward_sum = np.sum(self._drs)
		self.recordings.append(reward_sum)
		self._xs, self._hs, self._dlogps, self._drs = [],[],[],[] # reset array memory


		# compute the discounted reward backwards through time
		discounted_epr = self.discount_rewards(epr)
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		discounted_epr -= np.mean(discounted_epr)
		discounted_epr /= np.std(discounted_epr)


		epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

		grad = self.policy_backward(eph, epdlogp, epx)
		for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch


		if self._episode_nbr % self.batch_size == 0:
			if self._update_method == "adagrad":
				self.update_model_adagrad()
			elif self._update_method == "rmscache":
				self.update_model()

		if self._episode_nbr % 50 == 0:
			self.save_model()
			self.save_recordings()
			self.print_status(reward_sum)

	def update_model(self):
		for k,v in self.model.iteritems():
			g = self.grad_buffer[k] # gradient
			self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
			self.model[k] -= self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
			self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer


	def update_model_adagrad(self):
		# perform parameter update with Adagrad
		for param, dparam, mem in zip(	 
										[self.model['W1'], self.model['W2']],
										[self.grad_buffer['W1'], self.grad_buffer['W2']], 
										[self.adagrad_mem['W1'], self.adagrad_mem['W2']]):
			mem += dparam * dparam
			param += -0.01 * dparam / np.sqrt(mem + 1e-8) # adagrad update

	def save_model(self):
		pickle.dump(self.model, open(self.model_file, 'wb'))

	def save_recordings(self):
		pickle.dump(self.recordings, open(self.recording_file, 'wb'))

	def print_status(self, reward):
		print "Episode {} finished with reward {}".format(len(self.recordings), reward)