#partfrac2
import numpy as np
import sympy as sp
import sys

n=2
mu=(n+1)/2

R = sp.symbols('R', real=True, positive=True)
xi = sp.symbols('xi1:{}'.format(n+1),real=True)
x = sp.symbols('x1:{}'.format(n+1),real=True)

hp = sp.Function('h_+',nonzero=True)(*x,*xi[0:-1],R)
hm = sp.Function('h_-',nonzero=True)(*x,*xi[0:-1],R)

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

def pfthese(totlist,h):
	h = int(h)
	pfed = sp.zeros(rows=2,cols = len(totlist))
	for ir in range(0,len(totlist)):
		pfed[0,ir], pfed[1,ir] = pfdecomp(totlist[ir],h+1)
		print('done {}/{}'.format(ir,len(totlist)))
	return pfed	







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
		if denDO == 1:
			hpterm += dont*DONT
		else:	
			mpart, ppa = list(denDO.as_coeff_mul(hp))
			if list(ppa) != []:
				ppart = list(ppa)[0]
				degp = sp.degree(ppart,xi[-1])
			else:
				ppart = 1
				degp = 0	
			if mpart != []:	
				degm = sp.degree(mpart,xi[-1])
			else:
				degm = 0	
			degnum =sp.degree(DO, xi[-1])
			A = sp.symbols('A1:{}'.format(degp+1)) 
			B = sp.symbols('B1:{}'.format(degm+1))
			Acoeff = list(A)	
			Bcoeff = list(B)
			degdiff = degnum - degp - degm 
			Ecoeff = []
			if degdiff >= 0:
				E = sp.symbols('E1:{}'.format(degdiff + 2))
				Ecoeff = list(E)	
			fins = 0
			# construct simplified straight away
			for p in range(1,degp+1):
				fins += Acoeff[p-1]*(xi[-1]-hp)**(degp-p)*(xi[-1]-hm)**(degm)
			for m in range(1,degm+1):
				fins += Bcoeff[m-1]*(xi[-1]-hm)**(degm-m)*(xi[-1]-hp)**(degp)
			for e in range(0,len(Ecoeff)):
				fins += Ecoeff[e]*xi[-1]**e*(xi[-1]-hm)**(degm)*(xi[-1]-hp)**(degp)
			# final expression form
			fin = 0
			for p in range(1,degp+1):
				fin += Acoeff[p-1]*(xi[-1]-hp)**(-p)
			for m in range(1,degm+1):
				fin += Bcoeff[m-1]*(xi[-1]-hm)**(-m)
			for e in range(0,len(Ecoeff)):
				fin += Ecoeff[e]*xi[-1]**e 
			if f in vault:
				solset = vault[f]
				print('exists')
			else:	
				print('new')
				print(f)
				# create expression that we will solve against =0
				mul_exp = DO-fins
				# first part, solved by hp
				solset = {}

				soln = sp.solve(mul_exp.subs(xi[-1],hp), *Acoeff,*Ecoeff, dict = True, rational=True, quick=True)
				soln = soln[0]
				for key in soln:
					solset[key] = soln[key]
				soln = sp.solve(mul_exp.subs(xi[-1],hm), *Bcoeff, dict = True, rational=True, quick=True)
				soln = soln[0]
				for key in soln:
					solset[key] = soln[key]
				for varr in solset:
					fin = fin.subs(varr, solset[varr]).doit()
					mul_exp = mul_exp.subs(varr, solset[varr]).doit()
				if degp>1 or degm>1 or degdiff>0:
					solsetunsolved = {}
					allcoeffs = [*Acoeff[:-1], *Bcoeff[:-1], *Ecoeff]
					# print(allcoeffs)
					# input('leftovercoeffs')
					for coeffl in allcoeffs:
						solset[coeffl] = 0

					totunsolved = degp + degm - 2 + len(Ecoeff)
					smats = sp.zeros(rows = 1, cols = totunsolved)
					for dds in range(0,totunsolved):
						smats[dds] = mul_exp.subs((dds)*(hp-hm),xi[-1])
					eqmat = smats
					for dds in range(0,totunsolved):
						eqmat[dds] = sp.Eq(smats[dds],0)
					soln = sp.solve(eqmat, *Acoeff, *Bcoeff, *Ecoeff,dict = True, rational=True, quick=True)		
					if soln != []:
						soln = soln[0]
						for key in soln:
							solset[key] = soln[key]
				# if not isinstance(solset,dict): # make sure we have a solution
				# 	print(f)
				# 	sys.exit('Solution set is not a dictionary - check terms + orders')
				# print('done solving')
				vault[f] = solset	
				# print(vault)
			#do the substitutions
			for itemm in solset:
				fin = fin.subs(itemm, solset[itemm]).doit()	
			for JJ in range(1,degp+1):
				hmterm += DONT*dont*(fin.coeff((xi[-1]-hm),-JJ) * (xi[-1]-hm)**(-JJ))/denDONT	
			for JJ in range(1,degm+1): # make choice to put positive powers in h_+
				hpterm += DONT*dont*(fin.coeff((xi[-1]-hp),-JJ) * (xi[-1]-hp)**(-JJ))/denDONT
				hpterm += DONT*dont*(fin.coeff(xi[-1],JJ-1) * xi[-1]**(JJ-1))/denDONT	
	return hpterm, hmterm
