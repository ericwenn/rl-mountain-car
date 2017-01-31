import cPickle as pickle
import numpy as np
class Agent(object):
	def __init__(self, inputs, hidden_neurons, outputs, batch_size, learning_rate, reward_decay, decay_rate, model_file):

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

		# Name of file where to save our model
		self.model_file = model_file

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

		# First hidden state will be all zeros since there is no data yet
		self._prev_hidden_state = np.zeros(self.hidden_neurons)

		self.debug = False

	def load_model( self ):
		if( self.model == None ):
			self.model = pickle.load(open(self.model_file, 'rb'))
		else:
			print "Model already initialized."
			exit()


	def new_model(self):
		if( self.model == None):
			self.model = {}
			# Xavier initialisation
			self.model['U'] = np.random.randn(self.inputs, self.hidden_neurons) / np.sqrt(self.inputs) # Input to hidden
			self.model['W'] = np.random.randn(self.hidden_neurons, self.hidden_neurons) / np.sqrt(self.hidden_neurons) # Hidden to hidden
			self.model['V'] = np.random.randn(self.hidden_neurons, self.outputs) / np.sqrt(self.hidden_neurons) # Hidden to output
		else:
			print "Model already initialized."
			exit()



	def init_gradbuffer(self):
		self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # update buffers that add up gradients over a batch

	def init_rmscache(self):
		self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # rmsprop memory




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

		if( self.debug ):
			print "X: {}".format(x)
	

		# Calculate new hidden state based on observation and previous hidden state
		h = np.tanh(np.dot(x, self.model['U']) + np.dot(self._prev_hidden_state, self.model['W']))

		# Save hidden state to be used in next step
		self._prev_hidden_state = h;

		h[h<0] = 0 # ReLU nonlinearity
		
		logp = np.dot(h, self.model['V'])
		p = self._softmax(logp)
		return p, h # return probability of taking actions, and hidden state



	def policy_backward(self, eph, epdlogp, epx):
		""" backward pass. 
		(eph is array of intermediate hidden states)
		(epdlogp is array of modulated reward gradient) """

		debug = False
		if( debug ):
			print eph.shape
			print epdlogp.shape
			print epx.shape

		dV, dW, dU = np.zeros_like(self.model['V']), np.zeros_like(self.model['W']), np.zeros_like(self.model['U'])
		

		dhnext = np.zeros_like(eph[0])
		if( len(dhnext.shape) == 1):
			dhnext = dhnext[np.newaxis, ...]

		if( debug ):
			print "dhnext shape: {}".format(dhnext.shape)
		for t in reversed(xrange(len(eph))):
			hidden = eph[t]
			prob = epdlogp[t]
			obser = epx[t]
			prev_hidden = eph[t-1]


			if( len(hidden.shape) == 1):
				hidden = hidden[np.newaxis, ...]

			if( len(prob.shape) == 1):
				prob = prob[np.newaxis, ...]

			if( len(obser.shape) == 1):
				obser = obser[np.newaxis, ...]
			
			if( len(prev_hidden.shape) == 1):
				prev_hidden = prev_hidden[np.newaxis, ...]


			dV += np.dot( hidden.T, prob)
			if( debug ):
				print "dV shape: {}".format(dV.shape)
				print "dV: {}".format(dV)


			dh = np.dot(prob, self.model['V'].T) + dhnext
			if( debug ):
				print "dh shape: {}".format( dh.shape )
				print "dh: {}".format(dh)

			dhraw = (1 - hidden * hidden) * dh
			if( debug ):
				print "dhraw shape: {}".format(dhraw.shape)
				print "dhraw: {}".format(dhraw)

			dW += np.dot(dhraw, prev_hidden.T)
			if( debug ):
				print "dW shape: {}".format(dW.shape)
				print "dW: {}".format(dW)

			dU += np.dot(obser.T, dhraw)
			if( debug ):
				print "dU shape: {}".format(dU.shape)
				print "dU: {}".format(dU)

			dhnext = np.dot(dhraw, self.model['W'].T)

		for dparam in [dV, dU, dW]:
		    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		
		dModel = {'U':dU, 'W':dW, 'V':dV}
		return dModel


	def decide_action(self, observation):
		probabilites, h = self.policy_forward(observation)
		rand = np.random.uniform(0, np.sum(probabilites))


		aprob_cum = np.cumsum(probabilites)
		a = np.where( rand < aprob_cum)[0][0]

		if( self.debug ):
			print "PROB: {}".format(probabilites)
			print "H: {}".format(h)
			print "A: {}".format(a)
		


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

		self._xs, self._hs, self._dlogps, self._drs = [],[],[],[] # reset array memory


		# compute the discounted reward backwards through time
		discounted_epr = self.discount_rewards(epr)
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		discounted_epr -= np.mean(discounted_epr)
		discounted_epr /= np.std(discounted_epr)


		epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

		print "Backpropagating ..."
		grad = self.policy_backward(eph, epdlogp, epx)
		print "Backpropagation done"
		for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch


		if self._episode_nbr % self.batch_size == 0:
			for k,v in self.model.iteritems():
				g = self.grad_buffer[k] # gradient
				self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
				self.model[k] -= self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
				self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
			print self.model

		if self._episode_nbr % 100 == 0:
			self.save_model()

	def save_model(self):
		pickle.dump(self.model, open(self.model_file, 'wb'))
