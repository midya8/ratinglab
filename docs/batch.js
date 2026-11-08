'use strict';
const RatingBatch=(()=>{
 const dot=(a,b)=>a.reduce((s,v,j)=>s+v*b[j],0);
 function factor(H){const n=H.length,L=Array.from({length:n},()=>Array(n).fill(0));for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let v=H[i][j];for(let k=0;k<j;k++)v-=L[i][k]*L[j][k];if(i===j){if(!Number.isFinite(v)||v<=0)throw Error('System lost numerical positive definiteness');L[i][j]=Math.sqrt(v);}else L[i][j]=v/L[j][j];}return L;}
 function solve(L,b){const n=b.length,y=[],x=Array(n).fill(0);for(let i=0;i<n;i++){let v=b[i];for(let j=0;j<i;j++)v-=L[i][j]*y[j];y.push(v/L[i][i]);}for(let i=n-1;i>=0;i--){let v=y[i];for(let j=i+1;j<n;j++)v-=L[j][i]*x[j];x[i]=v/L[i][i];}return x;}
 function response(A,t,lam,rows,q,deltas=null,remove=false){
  const real=v=>typeof v==='number'&&Number.isFinite(v);
  if(!Array.isArray(A)||!A.length||!Array.isArray(A[0])||!A[0].length||A[0].length>33||A.some(r=>!Array.isArray(r)||r.length!==A[0].length||r.some(v=>!real(v)))||!Array.isArray(t)||t.length!==A.length||t.some(v=>!real(v))||!Array.isArray(q)||q.length!==A[0].length||q.some(v=>!real(v)))throw Error('Finite matching nonempty problem required');
  if(!real(lam)||lam<=0||typeof remove!=='boolean'||!Array.isArray(rows)||rows.length>64||new Set(rows).size!==rows.length||rows.some(k=>!Number.isInteger(k)||k<0||k>=A.length))throw Error('Positive penalty and distinct original rows required');
  if(remove&&deltas!==null)throw Error('Deletion cannot edit targets');
  const changes=remove?rows.map(()=>0):deltas===null?rows.map(()=>0):deltas;
  if(!Array.isArray(changes)||changes.length!==rows.length||changes.some(v=>!real(v)))throw Error('One finite delta per selected row required');
  const d=q.length,H=Array.from({length:d},(_,j)=>Array.from({length:d},(_,k)=>(j===k?lam:0)+A.reduce((s,a)=>s+a[j]*a[k],0))),rhs=q.map((_,j)=>A.reduce((s,a,k)=>s+a[j]*t[k],0)),L=factor(H),base=solve(L,rhs),B=rows.map(k=>A[k]),zs=B.map(b=>solve(L,b)),Z=q.map((_,j)=>zs.map(z=>z[j])),K=B.map(b=>zs.map(z=>dot(b,z))),schur=K.map((r,j)=>r.map((v,k)=>(j===k?1:0)-(v+K[k][j])/2)),residual=B.map((b,k)=>t[rows[k]]-dot(b,base));
  let coefficient=changes,condition=null;
  if(remove&&rows.length){const LS=factor(schur),inv=rows.map((_,j)=>solve(LS,rows.map((_,k)=>j===k?1:0))),norm=Math.max(...schur.map(r=>r.reduce((s,v)=>s+Math.abs(v),0))),invnorm=Math.max(...rows.map((_,j)=>inv.reduce((s,col)=>s+Math.abs(col[j]),0)));condition=norm*invnorm;if(condition>1e12)throw Error('Schur system condition too large');coefficient=solve(LS,residual).map(v=>-v);}
  const point=base.map((v,j)=>v+dot(Z[j],coefficient)),individual=remove?residual.map((v,k)=>-v/schur[k][k]):changes,separate=base.map((v,j)=>v+dot(Z[j],individual));
  const selected=new Set(rows),RA=A.filter((_,k)=>!remove||!selected.has(k)),rt=t.flatMap((v,k)=>remove&&selected.has(k)?[]:[v+(!remove&&selected.has(k)?changes[rows.indexOf(k)]:0)]),CH=q.map((_,j)=>q.map((_,k)=>(j===k?lam:0)+RA.reduce((s,a)=>s+a[j]*a[k],0))),crhs=q.map((_,j)=>RA.reduce((s,a,k)=>s+a[j]*rt[k],0));
  const normalResidual=Math.max(...point.map((_,j)=>Math.abs(dot(CH[j],point)-crhs[j]))),scale=1+Math.max(...crhs.map(Math.abs))+Math.max(...CH.flat().map(Math.abs))*d*Math.max(...point.map(Math.abs)),relative=normalResidual/scale;
  if(point.some(v=>!Number.isFinite(v))||![normalResidual,scale,relative].every(Number.isFinite)||relative>1e-10)throw Error('Updated normal residual too large');
  return {schema:1,rows:rows.slice(),delete:remove,deltas:remove?null:changes.slice(),design:A.map(r=>r.slice()),target:t.slice(),regularization:lam,query:q.slice(),normal:H,rhs,baseline:base,directions:Z,leverage_block:K,schur,selected_residuals:residual,coefficient,point,separate_sum_point:separate,baseline_score:dot(q,base),score:dot(q,point),score_change:dot(q,point.map((v,j)=>v-base[j])),separate_score_change:dot(q,separate.map((v,j)=>v-base[j])),interaction:dot(q,point.map((v,j)=>v-separate[j])),schur_condition:condition,maximum_normal_residual:normalResidual,scaled_normal_residual:relative,schur_eigenvalue_lower_bound:lam/(lam+A.reduce((s,a)=>s+dot(a,a),0))};
 }
 function blockResponse(data,m,side,index,sources,query,deltas=null,remove=false){
  if(!['user','item'].includes(side))throw Error('Known side required');const user=side==='user',own=user?'users':'items',opposite=user?'items':'users',obias=user?'item_bias':'user_bias',ownbias=user?'user_bias':'item_bias';
  if(!Number.isInteger(index)||index<0||index>=m[own].length||!Number.isInteger(query)||query<0||query>=m[opposite].length)throw Error('Known block and query required');const block=data.split.train.filter(j=>data.edges[j][user?0:1]===index);
  if(!Array.isArray(sources)||new Set(sources).size!==sources.length||sources.some(j=>!Number.isInteger(j)||!block.includes(j)))throw Error('Distinct training IDs within selected block required');
  const change=remove?null:deltas===null?sources.map(()=>0):deltas;
  if(!remove&&(!Array.isArray(change)||change.length!==sources.length||change.some((v,k)=>typeof v!=='number'||!Number.isFinite(v)||data.edges[sources[k]][2]+v<0||data.edges[sources[k]][2]+v>2)))throw Error('Edited ratings must stay in [0,2]');
  const A=block.map(j=>[1,...m[opposite][data.edges[j][user?1:0]]]),t=block.map(j=>{const e=data.edges[j];return e[2]-m.mean-m[obias][e[user?1:0]];}),q=[1,...m[opposite][query]],out=response(A,t,m.regularization,sources.map(j=>block.indexOf(j)),q,change,remove),offset=m.mean+m[obias][query],original=[m[ownbias][index],...m[own][index]];
  return {...out,side,index,source_rows:sources.slice(),block_source_rows:block,query_other:query,original_ratings:sources.map(j=>data.edges[j][2]),offset,saved_score:offset+dot(q,original),baseline_score:offset+out.baseline_score,score:offset+out.score,saved_to_baseline_max_change:Math.max(...original.map((v,j)=>Math.abs(v-out.baseline[j])))};
 }
 return {response,blockResponse};
})();
if(typeof module!=='undefined')module.exports=RatingBatch;
