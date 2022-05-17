#create_qpm1
import numpy as np
import sympy as sp
from itertools import product
from datetime import datetime
import sys

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
			# print(tvec[j], *arr4[j], gamma)
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
def mulsplit(g):
	# print('mulsplit')
	thisone = list(g.as_coeff_Mul())
	xi2list = []
	noxi2list = []
	for item in thisone:
		if item.has(xi[1]):
			xi2list.append(item)
	for item in thisone:
		if not item.has(xi[1]):
			noxi2list.append(item)		
	return sp.prod(xi2list), sp.prod(noxi2list)		

def DOsplit(g):
	# print('DOsplit')
	thisone = list(g.as_coeff_mul(xi[-1]))
	xi2list = sp.prod(thisone[1])
	noxi2list = thisone[0]
	return xi2list, noxi2list

def addsplit(g):
	# print('addsplit')
	thisone = g.as_coeff_Add()
	this = list(thisone)

vault = {}
# partial fraction decomposition, takes in a function and its order.
# from calculations we know the highest power in the denominator has to be j+1
# input expression f and degree of denominator, 
# output partial fraction decomposition of f
def pfdecomp(f,j):
	hpterm = 0
	hmterm = 0
	if f==0:
		return 0,0
	# performing some simplifications here	
	f1 = sp.ratsimp(f)
	f = sp.nsimplify(f)
	f = sp.factor(f)
	do, dont = mulsplit(f)
	f = sp.expand(do)
	gg = f.as_ordered_terms()
	# have sufficiently simplified the term so can pf
	for item in gg:				
		num = sp.numer(item)
		den = sp.denom(item)
		den = sp.factor(den)
		DO, DONT = DOsplit(num)
		denDO, denDONT = DOsplit(den)
		f = DO/denDO # last step to remove terms not involving xi_n
		powh = j+7 #from the theorem we know that this is the highest power, lowest being 1
		# create coefficients to match
		A = sp.symbols('A1:{}'.format(powh+1)) 
		B = sp.symbols('B1:{}'.format(powh+1))
		Acoeff = list(A)	
		Bcoeff = list(B)
		E = sp.symbols('E1:{}'.format(7))
		Ecoeff = list(E)
		fin = 0
		for p in range(1,powh+1):
			fin += Acoeff[p-1]*(xi[-1]-hp)**(-p)
		for m in range(1,powh+1):
			fin += Bcoeff[m-1]*(xi[-1]-hm)**(-m)
		for e in range(0,len(Ecoeff)):
			fin += Ecoeff[e]*xi[-1]**e 	
		if f in vault:
			solset = vault[f]
			# print('exists')
		else:	
			# print('new')
			# print(f)
			# print('solving for pf coefficients A and B and E...')
			solset = sp.solve_undetermined_coeffs(sp.Eq(sp.nsimplify(fin), f),
				[*Acoeff,*Bcoeff, *Ecoeff],xi[-1],list=True,rational=True)
			if not isinstance(solset,dict): # make sure we have a solution
				print(f)
				sys.exit('Solution set is not a dictionary - check terms + orders')
			# print('done solving')
			vault[f] = solset	
			# print(vault)
		#do the substitutions
		for itemm in solset:
			fin = fin.subs(itemm, solset[itemm]).doit()	
		for JJ in range(1,powh+1):
			hmterm += DONT*dont*(fin.coeff((xi[-1]-hm),-JJ) * (xi[-1]-hm)**(-JJ))/denDONT	
		for JJ in range(1,powh+1): # make choice to put positive powers in h_+
			hpterm += DONT*dont*(fin.coeff((xi[-1]-hp),-JJ) * (xi[-1]-hp)**(-JJ))/denDONT
			hpterm += DONT*dont*(fin.coeff(xi[-1],JJ-1) * xi[-1]**(JJ-1))/denDONT	
	return hpterm, hmterm
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
					# print('doing a=0 at {}'.format(J)) # this is only to keep track
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
						# print('doing derivative {},{} at {}'. format(xider,xder,J))		
						xires = sp.diff(wh[0,k],*xider)
						# xres = sp.diff(wh[1,l],*xder)
						term = -(1/(sp.factorial(mod_a)))*(xires)*D(mod_a,wh[1,l],xder)
						term /= qd(0)
						term = sp.expand(sp.nsimplify(term))
						tlist = term.as_ordered_terms()
						tots += list(tlist)
		# print('doing lhs1')				
		term = sp.expand(sp.nsimplify(qd(J))/(qd(0))) 
		tlist = term.as_ordered_terms()
		tots += list(tlist)
		pp = 0
		mm = 0
		count = 0
		vault_tots = {}
		for ts in tots:
			breakts = ts.as_coeff_mul()
			coeff_ts = breakts[0]
			terms_ts = breakts[-1]
			term_ts = 1
			for termts in terms_ts:
				term_ts *= termts
			# print(ts, breakts,coeff_ts,term_ts)
			# input('hhhhhhhhh')
			if term_ts in vault_tots:
				vault_tots[term_ts] += coeff_ts	
			else:
				vault_tots[term_ts] = coeff_ts		
		new_tots = []
		for ex in vault_tots:
			new_tots.append(sp.nsimplify(vault_tots[ex]*ex))
		print(len(tots) - len(new_tots))
		# print(sp.latex(vault_tots))
		# input('second')
		new_tots = tots
		for item in new_tots:
			count+=1
			item = sp.factor(sp.nsimplify(item))
			# print('doing {}/{}'.format(count,len(new_tots)))
			pp, mm = pfdecomp(item,J)
			wh[0,J] += pp
			wh[1,J] += mm
		wh[0,J] *= wh[0,0] 
		wh[1,J] *= wh[1,0]
	return wh



qq = whqd(2)


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
# Calculate boundary integrand for c_j
#================================================================

# input k, output boundary integrand of c_k
def bdyc(k):
	tot = 0
	for gn in range(1,k+1):
		for j in range(0,k):
			for l in range(0,k):
				mod_b = k - gn - j - l
				if mod_b == 0 and gn>1:
					tot += ((-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(gn))*w(1,j)
						*(sp.diff(w(0,l),(xi[1],gn),(x[1],gn-1))))
				elif mod_b == 0	and gn==1:
					tot += (-1)**(mod_b+1)*(1j)**(mod_b+gn)*w(1,j)*(sp.diff(w(0,l),xi[1]))		
				elif mod_b>0 and gn>1:
					for bp in range(0,mod_b+1):
						bn = mod_b-bp
						tot+=( (-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(gn + bn) * sp.factorial(bp))
							*sp.diff(w(1,j),(x[0],bp),(x[1],bn))
							*(sp.diff(w(0,l),(xi[1],bn+gn),(xi[0],bp), (x[1],gn-1))))
				elif mod_b>0 and gn==1:
					for bp in range(0,mod_b+1):
						bn = mod_b-bp
						tot+=( (-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(1 + bn) * sp.factorial(bp))
							*sp.diff(w(1,j),(x[0],bp),(x[1],bn))
							*(sp.diff(w(0,l),(xi[1],bn+1),(xi[0],bp))))
	return tot	
#================================================================


gy = 1 + (sp.Abs(sp.diff(phi,x[0])))**2
b = sp.diff(phi,x[0])
H = sp.symbols('H',real=True)
A = sp.symbols('A',real=True)
G = sp.symbols('G',real=True)



bdyymb = bdyc(1)

print('calculated boundary term 1 ')
print(datetime.now() - startTime)

bdy = bdyymb.xreplace({hp:b*xi[0] + 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2), hm:b*xi[0] - 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2)}).doit()


bdy = bdy.xreplace({xi[1]:0})
bdy = bdy.subs(xi[0],0)


bdy = bdy.subs(sp.diff(phi,x[0],x[0],x[0]),G)
bdy = bdy.subs(sp.diff(phi,x[0],x[0]),H)
bdy = bdy.subs(sp.diff(phi,x[0]),A)

bdy = bdy.subs(A,0)

with open('bdy1.txt','w') as inf:
	inf.write(sp.latex(bdy))



#================================================================


bdyymb = bdyc(2)

print('calculated boundary term 2 ')
print(datetime.now() - startTime)

bdy = bdyymb.xreplace({hp:b*xi[0] + 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2), hm:b*xi[0] - 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2)}).doit()


bdy = bdy.xreplace({xi[1]:0})
bdy = bdy.subs(xi[0],0)


bdy = bdy.subs(sp.diff(phi,x[0],x[0],x[0]),G)
bdy = bdy.subs(sp.diff(phi,x[0],x[0]),H)
bdy = bdy.subs(sp.diff(phi,x[0]),A)

bdy = bdy.subs(A,0)

with open('bdy2.txt','w') as inf:
	inf.write(sp.latex(bdy))


#================================================================


bdyymb = bdyc(3)

print('calculated boundary term 3 ')
print(datetime.now() - startTime)

bdy = bdyymb.xreplace({hp:b*xi[0] + 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2), hm:b*xi[0] - 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2)}).doit()


bdy = bdy.xreplace({xi[1]:0})
bdy = bdy.subs(xi[0],0)


bdy = bdy.subs(sp.diff(phi,x[0],x[0],x[0]),G)
bdy = bdy.subs(sp.diff(phi,x[0],x[0]),H)
bdy = bdy.subs(sp.diff(phi,x[0]),A)

bdy = bdy.subs(A,0)

with open('bdy3.txt','w') as inf:
	inf.write(sp.latex(bdy))


#================================================================


