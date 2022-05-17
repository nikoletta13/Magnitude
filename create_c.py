#create_c
import numpy as np
import sympy as sp
from itertools import product
from datetime import datetime
import sys
import cloudpickle as cp

startTime = datetime.now()

K = sys.argv[1]
inp = sys.argv[2]

ck = int(K) + 1

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

#this is the loc
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
def d(num, expr, var):
	return sp.diff(expr,*var)



#================================================================


#================================================================
# Load w terms
#================================================================
w = sp.zeros(rows=2, cols=ck)
w[0,0] = 1/(sp.factorial(n)*omega(n))*( h0)**(mu)*(xi[-1]-hp)**(mu)
w[1,0] = (xi[-1]-hm)**(mu)
for ww in range(1, ck):
	with open('wp{}'.format(ww),'rb') as inf:
		w[0,ww] = cp.load(inf)
	with open('wm{}'.format(ww),'rb') as inf:
		w[1,ww] = cp.load(inf)

print('opened all')
print(datetime.now() - startTime)
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
					tot += ((-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(gn))*w[1,j]
						*(sp.diff(w[0,l],(xi[1],gn),(x[1],gn-1))))
				elif mod_b == 0	and gn==1:
					tot += (-1)**(mod_b+1)*(1j)**(mod_b+gn)*w[1,j]*(sp.diff(w[0,l],xi[1]))		
				elif mod_b>0 and gn>1:
					for bp in range(0,mod_b+1):
						bn = mod_b-bp
						tot+=( (-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(gn + bn) * sp.factorial(bp))
							*sp.diff(w[1,j],(x[0],bp),(x[1],bn))
							*(sp.diff(w[0,l],(xi[1],bn+gn),(xi[0],bp), (x[1],gn-1))))
				elif mod_b>0 and gn==1:
					for bp in range(0,mod_b+1):
						bn = mod_b-bp
						tot+=( (-1)**(mod_b+1)*(1j)**(mod_b+gn)/(sp.factorial(1 + bn) * sp.factorial(bp))
							*sp.diff(w[1,j],(x[0],bp),(x[1],bn))
							*(sp.diff(w[0,l],(xi[1],bn+1),(xi[0],bp))))
	return tot						
#================================================================

bdysymb = bdyc(ck)

gy = 1 + (sp.Abs(sp.diff(phi,x[0])))**2
b = sp.diff(phi,x[0])
H = sp.symbols('H',real=True)
A = sp.symbols('A',real=True)
G = sp.symbols('G',real=True)


print('calculated boundary term')
print(datetime.now() - startTime)

bdy = bdyymb.xreplace({hp:b*xi[0] + 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2), hm:b*xi[0] - 1j*sp.sqrt(xi[0]**2*gy+1+(b*xi[0])**2)}).doit()


bdy = bdy.xreplace({xi[1]:0})
bdy = bdy.subs(xi[0],0)

inphi = sp.sympify(inp)

if inphi != 0:
	bdy = bdy.xreplace({phi:inphi})	
else:
	bdy = bdy.subs(sp.diff(phi,x[0],x[0],x[0]),G)
	bdy = bdy.subs(sp.diff(phi,x[0],x[0]),H)
	bdy = bdy.subs(sp.diff(phi,x[0]),A)





# print('subb1')
# print(datetime.now() - startTime)

# print(bdy)

# bdy = sp.simplify(bdy)


print('subb2')
print(datetime.now() - startTime)

# bdy = bdy.subs(A,0)
# with open('c4howboutnoo','wb') as inf:
# 	cp.dump(bdy,inf)

# print('written')
# print(datetime.now() - startTime)
bdy = sp.expand(bdy)
bdy = sp.nsimplify(bdy)
bdy = sp.simplify(bdy)
with open('bdy.txt','w') as inf:
	inf.write(bdy)
print(sp.latex(bdy))
# print(sp.latex(sp.simplify(sp.simplify(bdy))))
# with open('c2symb','wb') as inf:
# 	cp.dump(c2symb,inf)

print(datetime.now() - startTime)

# with open('c3symb','wb') as inf:
# 	cp.dump(c3symb,inf)


# dd = {R:1, xi[-1]:0}

# c2sub = sp.nsimplify(c2symb.subs(dd))
# c3sub = sp.nsimplify(c3symb.subs(dd))

# with open('c2subs','w') as inf:
# 	inf.write(sp.latex(c2sub))

# with open('c3subs','w') as inf:
# 	inf.write(sp.latex(c3sub))

