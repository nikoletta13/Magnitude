# pfqXalt
import numpy as np
import sympy as sp
from itertools import product
from datetime import datetime
import sys
# import cloudpickle as cp
import storing as st
import partfrac2 as pf


X = sys.argv[1] #pf_qX
th = sys.argv[2] # deal with this part


#================================================================
# introduce variables and functions
#================================================================
n=2

mu = (n+1)/2

R = sp.symbols('R', real=True, positive=True)
xi = sp.symbols('xi1:{}'.format(n+1),real=True)
x = sp.symbols('x1:{}'.format(n+1),real=True)

hp = sp.Function('h_+',nonzero=True)(*x,*xi[0:-1],R)
hm = sp.Function('h_-',nonzero=True)(*x,*xi[0:-1],R)
def tfunc(b):
	return tcoeff(b[0],b[1])


# only return if an = 0 , 1
def tcoeff(ap,an):
	if an==1:
		return (-1)**(ap+an) * ((-2)/(sp.factorial(ap)))* sp.diff(phi, (x[0], ap))
	if an==0:
		bdiff = 1
		s_ap_bp = 0
		while bdiff<ap:
			adiff = ap-bdiff
			s_ap_bp+= (-1)**(ap)*(1/sp.factorial(adiff))*(1/sp.factorial(bdiff))*(sp.diff(phi,(x[0],adiff)))*(sp.diff(phi,(x[0],bdiff)))
			bdiff+=1	
		return s_ap_bp
	else:
		return 0
def omega(n):
	return (sp.pi**(n/2)/sp.gamma(sp.S(n+2)/2))		
h0=1 #graph	
def D(num,expr,var):
	return (-1j)**(num)*sp.diff(expr,*var)

phi = sp.Function('phi')(x[0])	
	
#================================================================

#================================================================
# load part
#================================================================
this = st.cpopenpart('q',X,th)
# with open('q{}part{}'.format(X,th),'rb') as inf:
# 	this = cp.load(inf)

#================================================================

#================================================================
# Partial fraction decomposition
#================================================================
#

#================================================================
# Do pf
#================================================================
pp = 0
mm = 0
count = 0
qp = 0
qm = 0
qp0 = sp.factorial(n)*omega(n)*( h0)**(-mu)*(xi[-1]-hp)**(-mu)
qm0 = (xi[-1]-hm)**(-mu)
for ii in range(0,len(this)):
	this[ii] = sp.factor(sp.nsimplify(this[ii]))
pfed = pf.pfthese(this,X)
print(len(pfed)-len(this))
# for item in this:
# 	count+=1
# 	item = sp.factor(sp.nsimplify(item))
# 	# print('doing {}/{}'.format(count,len(this)))
# 	pp, mm = pfdecomp(item,int(X)+3)
for ii in range(0,len(this)):
	qp += sp.factor(sp.nsimplify(qp0*pfed[0,ii])) 
	qm += sp.factor(sp.nsimplify(qm0*pfed[1,ii]))
#================================================================

#================================================================
# Save part
#================================================================
st.cpstorepart(qp,'qp',X,th)
# with open('qp{}part{}'.format(X,th),'wb') as inf:
# 	cp.dump(qp,inf)
st.cpstorepart(qm,'qm',X,th)
# with open('qm{}part{}'.format(X,th),'wb') as inf:
# 	cp.dump(qm,inf)	
#================================================================


