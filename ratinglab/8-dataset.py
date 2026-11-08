"""Original licensed UCI ratings only, without consumer profiles or locations."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent

def load():
    raw=(ROOT/'data/rating_final.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='fc994ea4e872365fc63e7f0bc970b36f86edeb4e80218d6a674a3d0e51181901':raise ValueError('original ratings hash differs')
    rows=list(csv.DictReader(io.StringIO(raw.decode())))
    users=sorted({r['userID'] for r in rows});items=sorted({r['placeID'] for r in rows})
    u={v:i for i,v in enumerate(users)};v={x:i for i,x in enumerate(items)}
    result=np.array([[u[r['userID']],v[r['placeID']],int(r['rating'])] for r in rows],dtype=int)
    if len({tuple(r[:2]) for r in result})!=len(result):raise ValueError('duplicate user/item pair')
    return result,len(users),len(items),hashlib.sha256(raw).hexdigest()

def split(n):
    ids=np.random.default_rng(20261108).permutation(n);a=3*n//5;b=4*n//5
    return {'train':sorted(ids[:a].tolist()),'validation':sorted(ids[a:b].tolist()),'test':sorted(ids[b:].tolist())}
