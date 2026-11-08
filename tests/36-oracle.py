"""Independent exact Gaussian elimination, no NumPy/optimizer import."""
from fractions import Fraction as F

def solve(A,y,lam):
    rows=[[F(v) for v in row] for row in A];target=list(map(F,y));d=len(rows[0]);H=[[sum(row[j]*row[k] for row in rows)+(F(lam) if j==k else 0) for k in range(d)] for j in range(d)];b=[sum(row[j]*t for row,t in zip(rows,target)) for j in range(d)];M=[row+[v] for row,v in zip(H,b)]
    for j in range(d):
        pivot=next(k for k in range(j,d) if M[k][j]);M[j],M[pivot]=M[pivot],M[j];div=M[j][j];M[j]=[v/div for v in M[j]]
        for k in range(d):
            if k!=j:
                mult=M[k][j];M[k]=[a-mult*b for a,b in zip(M[k],M[j])]
    return [row[-1] for row in M]
