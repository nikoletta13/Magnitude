# create_qpmX
import numpy as np
import sympy as sp
# import cloudpickle as cp
import sys
import storing as st

X = sys.argv[1]

#================================================================
# introduce variables and functions
#================================================================
n=2

mu = (n+1)/2

R = sp.symbols('R', real=True, positive=True)
xi = sp.symbols('xi1:{}'.format(n+1),real=True)
x = sp.symbols('x1:{}'.format(n+1),real=True)

def D(num,expr,var):
	return (-1j)**(num)*sp.diff(expr,*var)
def invD(num,expr,var):
	return (1j)**(num)*sp.diff(expr,*var)	
def d(num, expr, var):
	return sp.diff(expr,*var)

phi = sp.Function('phi')(x[0])		
h0=1 #graph

hp = sp.Function('h_+',nonzero=True)(*x,*xi[0:-1],R)
hm = sp.Function('h_-',nonzero=True)(*x,*xi[0:-1],R)

#================================================================


#================================================================
# Load parts and combine
#================================================================

with open('length{}.txt'.format(X),'r') as inf:
	l = inf.read()
ll = int(l)



qpn = 0
qmn = 0
for i in range(1,ll):
	qpn += st.cpopen('qp',X,i)
	# with open('qp{}part{}'.format(X,i),'rb') as inf:
	# 	part = cp.load(inf)
	# qpn += part
	qmn += st.cpopen('qm',X,i)
	# with open('qm{}part{}'.format(X,i),'rb') as inf:
	# 	part = cp.load(inf)
	# qmn += part	 
#================================================================


#================================================================
# Save qpm2
#================================================================

st.cpstore(qpn,'qp',X)
# with open('qp{}'.format(X),'wb') as inf:
# 	cp.dump(qpn,inf)


st.cpstore(qmn,'qm',X)
# with open('qm{}'.format(X),'wb') as inf:
# 	cp.dump(qmn,inf)
#================================================================
