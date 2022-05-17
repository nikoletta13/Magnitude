# create_wterms
import numpy as np
import sympy as sp
from itertools import product
from datetime import datetime
import sys
# import cloudpickle as cp

import storing as st
startTime = datetime.now()

K = sys.argv[1] #calculate up to K-1
K= int(K)

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

# #this is the loc
# phi = 1 - x[0]**2/2
phi = sp.Function('phi')(x[0])	

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
#================================================================


#================================================================
# Load previous q
#================================================================
qq = sp.zeros(rows=2,cols=K +1)

qp[0,0] = sp.factorial(n)*omega(n)*( h0)**(-mu)*(xi[1]-hp)**(-mu)
qm[1,0] = (xi[1]-hm)**(-mu)

for KK in range(1,K+1):
	qq[0,KK] = st.cpopen('qp',KK)
	# with open('qp{}'.format(KK),'rb') as inf:
	# 	qq[0,KK] = cp.load(inf)
	qq[1,KK] = st.cpopen('qm',KK)
	# with open('qm1','rb') as inf:
	# 	qq[1,KK] = cp.load(inf)
#================================================================


#================================================================
# Calculate w terms
#================================================================


# first col is w_+,j , second is w_-,j since calculation is analogous
# input j and sign, output w_{sign,j}
def w(j,s): # s is 0 or 1, 0 is + and 1 is -
	w = sp.zeros(rows = 2, cols = j+1)
	w[0,0] = 1/(sp.factorial(n)*omega(n))*( h0)**(mu)*(xi[-1]-hp)**(mu)
	w[1,0] = (xi[-1]-hm)**(mu)
	for J in range(1,j+1): #this is the index of a in the result
		print(J)
		print(datetime.now() -  startTime)
		w[s,J]=0
		for l in range(0,J): #this is the index of a inside the sum on RHS
			for k in range(0,j+1): # this is the index of q
				mod_a = J-l-k
				if mod_a==0:  #then only possibilities are l+k = J 
					term = (qq[s,k]*w[s,l]).doit()
					w[s,J]+= sp.nsimplify(term)	
				elif mod_a >0:
					arr1 = 	[item for item in range(0,mod_a+1)]
					arr2 = list(product(arr1,repeat = n))
					arr3 = []
					for item in arr2:
						if sum(item)==mod_a:
							arr3.append(item) 
					for item in arr3: 
						xider=[]
						xder=[]
						for number in range(0,n):
							xider+= [xi[number]]*item[number]
							xder +=[x[number]]*item[number]
						xires = sp.diff(qq[s,k],*xider)
						# xres = sp.diff(w[s,l],*xder)	
						term = (1/(sp.factorial(mod_a)))*(xires)*D(mod_a,w[s,l],xder)
						w[s,J]+= sp.nsimplify(term)		
		w[s,J] *= (- w[s,0])	
	return w[s,-1]
#================================================================


#================================================================
# Store w terms
#================================================================
for KK in range(1,K+1):
	wpt = w(KK,0)
	st.cpstore(wpt,'wp',KK)
	# with open('wp{}'.format(KK),'wb') as inf:
	# 	cp.dump(wpt, inf)
	wmt = w(KK,1)
	st.cpstore(wmt,'wm',KK)
	# with open('wm{}'.format(KK),'wb') as inf:
	# 	cp.dump(wmt, inf)			
#================================================================