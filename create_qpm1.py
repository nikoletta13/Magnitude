#create_qpm1
import numpy as np
import sympy as sp
from itertools import product
from datetime import datetime
import sys
# import cloudpickle as cp
import partfrac2 as pf
import storing as st
startTime = datetime.now()


#================================================================
# Glossarry
#================================================================
#
# Code name		|Paper name 
# 	n       	|	n  
#	mu      	|	\mu
#	x			|	x
#	xi      	|	\xi
#	R       	|	R
#	I(j)		|	I_j
#	gamma 		|	\gamma
#	g		|	g
#	rk		|	rk	
#	C(expr,gamma)	|	C^{\gamma}expr
#	c(l,n)		|	c_{l,n}
#	omega		|	\omega
#	q(j)		|	q_j(x,\xi,R)
#	a(j) 		|	a_j(x,\xi,R)
#	h0		|	h_0
#	hp 		|	h_+
#	hm 		| 	h_-
#	qd(j)		|	q^\partial_j(x,\xi,R)
#	whqd(j)[0,-1]	|	q^\partial_{+,j}(x,\xi,R)
#	whqd(j)[1,-1]	|	q^\partial_{-,j}(x,\xi,R)
#	w(j)		|	w_{+,j}(x,\xi,R)
#	w(j)		|	w_{-,j}(x,\xi,R)
#================================================================


#================================================================
# introduce variables
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
#================================================================


#================================================================
# Define I_j, C^\gamma, rk(\gamma), c(l,n) , \omega_n
#================================================================
def vec(j):
	a = [item for item in range(3,j+2+1)]
	a.append(0)
	return(a)

# input j, output set I_j
def I(j):
	i=[]
	ii = []
	iii=[]
	iiii=[]
	ct = 0
	#create all j-tuples from vectors
	allcombs = list(product(vec(j),repeat = j))
	#check if they satisfy the sum condition
	for gam in allcombs:
		k = np.count_nonzero(gam)
		if sum(gam)==j+2*k:
			i.append(gam)
	#remove zeros 		
	for gam in i:
		ii.append(list(filter(lambda num: num !=0, gam)))
	#remove duplicates	
	L=[]
	for gam in ii:
		if gam not in L:
			L.append(gam)		
	return L

# Metric
g = sp.Function('g')(*x,xi[0])

# rk
def rk(gamma):
	return len(gamma)


# these are the t functions/coefficients, subscript is the separation of the
# tangential and normal variables i.e. (d\xi',d\xi_n)
def tfunc(b):
	return tcoeff(b[0],b[1])



# only return if an = 0 , 1 even though impossibility by eligible_t
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

# only consider graph setting i.e. if d\xi_n>1 t=0 so do not take it into account 
def sumvec(j):
	arr1 = [item for item in range(0,j+1)]
	arr2 = list(product(arr1,repeat=n))
	arr3=[]
	for item in arr2:
		if sum(item)==j:
			arr3.append(item)
	return arr3	

def eligible_t(j):
	el =[]
	for item in sumvec(j):
		if item[-1]<=1:
			el.append(item)
	return el		

	
# Define C
def C(expr,gamma):
	gampart = 1
	length = len(gamma)
	final = []
	for num in gamma:
		arrder=[]
		allf=[]
		arr3 = eligible_t(num)
		# create differentiation vec
		for item in arr3:
			new= []
			for number in range(0,n): # differentiate wrt
				new.append([xi[number]]*item[number])
			arrder.append(new)
		#remove list in list make one list
		arr4 = []	
		for nn in arrder:
			nnn=[]	
			for tou in range(0,len(nn)):
				nnn +=nn[tou]	
			arr4.append(nnn) 
		t = len(arr4)
		# vector of t terms to match corresponding diff
		tvec = [] 
		for item in arr3:
			tvec.append(tfunc(item)) 		
		for j in range(0,t): 
			# allf.append(tvec[j]*(sp.diff(expr,*arr4[j]))) change to invD
			allf.append(tvec[j]*(invD(len(arr4[j]),expr,arr4[j])))
			print(tvec[j], *arr4[j], gamma)
		# redefine expr at end of loop so that new tdxi can act on whatever created	
		expr = sp.nsimplify(sum(allf))	 
	return expr

# define volume of unit ball
def omega(k):
	return (sp.pi**(k/2)/sp.gamma(sp.S(k+2)/2))		

# Define coefficients c(l,n) for n=2 #this could just well be c_l but being consisten
def c(l,n):
	if 2*l-n-1<0:
		return (-1)**l*(sp.factorial(n-2*l))*omega(n-2*l)*omega(2*l)
	else:
		return omega(2*l)/((sp.factorial(2*l-n))*omega(2*l-n))
	
	
#================================================================


#================================================================
# Boundary terms qd in the expansion of Q^\partial
#================================================================
h0=1 #graph

hp = sp.Function('h_+',nonzero=True)(*x,*xi[0:-1],R)
hm = sp.Function('h_-',nonzero=True)(*x,*xi[0:-1],R)

gd = sp.Function('g_d')(*x,xi[0])


# input j, output qd_j
def qd(j):
	tot = 0
	if j==0:
		tot= sp.factorial(n)*omega(n)*( h0)**(-mu)*(xi[-1]-hp)**(-mu)*(xi[-1]-hm)**(-mu)
	else:
		for gamma in I(j):
			expr = ((h0)**(-mu+rk(gamma))*(xi[-1]-hp)**(-mu+ rk(gamma))
				*(xi[-1]-hm)**(-mu+rk(gamma)))
			prod = c(rk(gamma),n) *C(expr,gamma)  
			tot +=prod
	return tot.doit()


#================================================================
# Partial fraction decomposition
#================================================================
# These '*split' functions help split a term into components containing 
# xi_n or not because the solver used in the function pfdecomp cannot handle 
# multivariate derivatives. Also makes the calculation faster as these are very quick



#================================================================


#================================================================
# Wiener-Hopf near the boundary -- Q^\partial_\pm,j = whqd[0,j] resp whqd[1,j]
#================================================================

# want to do this in array just like a, because need previous terms to calculate next
# return as a matrix and use it in the future, instead of calculating again
# input j, output whqd_0,\pm up to whqd_j,\pm
def whqd(j):
	wh = sp.zeros(rows = 2, cols = j+1) #first row (0) +, second(1) -
	wh[0,0] = sp.factorial(n)*omega(n)*( h0)**(-mu)*(xi[-1]-hp)**(-mu)
	wh[1,0] = (xi[-1]-hm)**(-mu)
	for J in range(1,j+1): #controls col
		tots = []
		for k in range(0,J):
			for l in range(0,J):
				mod_a = J-k-l
				if mod_a ==0: 
					print('doing a=0 at {}'.format(J)) # this is only to keep track
					term = -wh[0,k]*wh[1,l]
					term /= qd(0)
					term = sp.expand(sp.nsimplify(term))
					tlist = term.as_ordered_terms()
					tots += list(tlist)
				elif mod_a >0:
					arr3 = sumvec(mod_a)
					for item in arr3: 
						xider=[]
						xder=[]
						for number in range(0,n):
							xider += [xi[number]]*item[number]
							xder +=[x[number]]*item[number]
						print('doing derivative {},{} at {}'. format(xider,xder,J))		
						xires = sp.diff(wh[0,k],*xider)
						# xres = sp.diff(wh[1,l],*xder)
						term = -(1/(sp.factorial(mod_a)))*(xires)*D(mod_a,wh[1,l],xder)
						term /= qd(0)
						term = sp.expand(sp.nsimplify(term))
						tlist = term.as_ordered_terms()
						tots += list(tlist)
		print('doing lhs1')				
		term = sp.expand(sp.nsimplify(qd(J))/(qd(0))) 
		tlist = term.as_ordered_terms()
		tots += list(tlist)
		pp = 0
		mm = 0
		count = 0
		print('tots = {}'.format(len(tots)))
		pfdlist = pf.pfthese(tots, J)
		for ii in range(0,len(tots)):
			wh[0,J] += pfdlist[0,ii]
			wh[1,J] += pfdlist[1,ii]
		wh[0,J] *= wh[0,0]
		wh[1,J] *= wh[1,0]
		# for item in tots:
		# 	count+=1
		# 	item = sp.factor(sp.nsimplify(item))
		# 	print('doing {}/{}'.format(count,len(tots)))
		# 	pp, mm = pf.pfdecomp(item,J)
		# 	wh[0,J] += pp
		# 	wh[1,J] += mm
		# wh[0,J] *= wh[0,0] 
		# wh[1,J] *= wh[1,0]
	return wh

qq = whqd(3)

# st.cpstore(qq[0,1],'qp','1')
# st.cpstore(qq[1,1],'qm','1')



# st.cpstore(qq[0,2],'qp','2')
# st.cpstore(qq[1,2],'qm','2')


st.cpstore(qq[0,3],'qp','3')
st.cpstore(qq[1,3],'qm','3')



# st.cpstore(qq[0,4],'qp','4')
# st.cpstore(qq[1,4],'qm','4')




# with open('qp1','wb') as inf:
# 	cp.dump(qq[0,-1],inf)



# with open('qm1','wb') as inf:
# 	cp.dump(qq[1,-1],inf)

