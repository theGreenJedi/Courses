from nltk.probability import ConditionalFreqDist
from rediscollections import RedisHashMap, encode_key

class RedisHashFreqDist(RedisHashMap):
	'''
	>>> from redis import Redis
	>>> r = Redis()
	>>> rhfd = RedisHashFreqDist(r, 'test')
	>>> rhfd.items()
	[]
	>>> rhfd.values()
	[]
	>>> len(rhfd)
	0
	>>> rhfd['foo']
	0
	>>> rhfd.inc('foo')
	>>> rhfd['foo']
	1
	>>> rhfd.items()
	[('foo', 1)]
	>>> rhfd.values()
	[1]
	>>> len(rhfd)
	1
	>>> rhfd.clear()
	'''
	def inc(self, sample, count=1):
		self._r.hincrby(self._name, sample, count)
	
	def N(self):
		return int(sum(self.values()))
	
	def __getitem__(self, key):
		return int(RedisHashMap.__getitem__(self, key) or 0)
	
	def values(self):
		return [int(v) for v in RedisHashMap.values(self)]
	
	def items(self):
		return [(k, int(v)) for (k, v) in RedisHashMap.items(self)]

class RedisConditionalHashFreqDist(ConditionalFreqDist):
	'''
	>>> from redis import Redis
	>>> r = Redis()
	>>> rchfd = RedisConditionalHashFreqDist(r, 'condhash')
	>>> rchfd.N()
	0
	>>> rchfd.conditions()
	[]
	>>> rchfd['cond1'].inc('foo')
	>>> rchfd.N()
	1
	>>> rchfd['cond1']['foo']
	1
	>>> rchfd.conditions()
	['cond1']
	>>> rchfd.clear()
	'''
	def __init__(self, r, name, cond_samples=None):
		self._r = r
		self._name = name
		ConditionalFreqDist.__init__(self, cond_samples)
		# initialize self._fdists for all matching keys
		for key in self._r.keys(encode_key('%s:*' % name)):
			condition = key.split(':')[1]
			self[condition] # calls self.__getitem__(condition)
	
	def __contains__(self, condition):
		return encode_key(condition) in self._fdists
	
	def __getitem__(self, condition):
		if condition not in self._fdists:
			key = '%s:%s' % (self._name, condition)
			self._fdists[condition] = RedisHashFreqDist(self._r, key)
		
		return self._fdists[condition]
	
	def clear(self):
		for fdist in self._fdists.values():
			fdist.clear()

if __name__ == '__main__':
	import doctest
	doctest.testmod()