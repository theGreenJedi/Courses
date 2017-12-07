# -*- coding: utf-8 -*-
import chardet

def detect(s):
	'''
	>>> detect('ascii')
	{'confidence': 1.0, 'encoding': 'ascii'}
	>>> detect(u'abcdé')
	{'confidence': 0.75249999999999995, 'encoding': 'utf-8'}
	>>> detect('\222\222\223\225')
	{'confidence': 0.5, 'encoding': 'windows-1252'}
	'''
	try:
		return chardet.detect(s)
	except UnicodeDecodeError:
		return chardet.detect(s.encode('utf-8'))

def convert(s):
	'''
	>>> convert('ascii')
	u'ascii'
	>>> convert(u'abcdé')
	u'abcd\\xc3\\xa9'
	>>> convert('\222\222\223\225')
	u'\u2019\u2019\u201c\u2022'
	'''
	encoding = detect(s)['encoding']
	
	if encoding == 'utf-8':
		return unicode(s)
	else:
		return unicode(s, encoding)

if __name__ == '__main__':
	import doctest
	doctest.testmod()