import re

from time import clock as now

def test(f, *args, **kargs):
    start = now()
    f(*args, **kargs)
    print "The function %s lasted: %f" %(f.__name__, now() - start)

def non_catastrophic(n):
    print "Testing with %d characters" %n
    pat = re.compile('(x+)+(b+)+c')
    text = 'x' * n
    text += 'b' * n
    text += 'c'
    pat.search(text)

for n in range(12, 18):
    test(non_catastrophic, n)




