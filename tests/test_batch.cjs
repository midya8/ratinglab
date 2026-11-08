'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),batch=require('../docs/batch.js');
const native=JSON.parse(fs.readFileSync('ratinglab/artifacts/batch-reference.json','utf8')),context={};vm.runInNewContext(fs.readFileSync('docs/data.js','utf8'),context);const data=context.RATING_DATA;
function close(a,b,path=''){if(typeof a==='number'&&typeof b==='number'){assert(Math.abs(a-b)<=2e-11*(1+Math.abs(b)),path+':'+a+' vs '+b);return;}if(Array.isArray(a)&&Array.isArray(b)){assert.equal(a.length,b.length,path);a.forEach((v,k)=>close(v,b[k],path+'/'+k));return;}assert.deepEqual(a,b,path);}
const fields=['rows','delete','deltas','design','target','regularization','query','normal','rhs','baseline','directions','leverage_block','schur','selected_residuals','coefficient','point','separate_sum_point','baseline_score','score','score_change','separate_score_change','interaction','maximum_normal_residual','scaled_normal_residual','schur_eigenvalue_lower_bound'];
test('all320 source-bound batch responses and controls match independent Cholesky JS',()=>{
 assert.equal(native.cases.length,320);
 for(const c of native.cases){const w=c.witness,m=data.candidates.find(m=>m.id===c.model_id).model,actual=batch.blockResponse(data,m,w.side,w.index,w.source_rows,w.query_other,w.deltas,w.delete);for(const k of [...fields,'side','index','source_rows','block_source_rows','query_other','original_ratings','offset','saved_score','saved_to_baseline_max_change'])close(actual[k],w[k],k);}
 for(const w of native.controls){const a=batch.response(w.design,w.target,w.regularization,w.rows,w.query,w.deltas,w.delete);fields.forEach(k=>close(a[k],w[k],k));}
});
test('independent direct augmented-design scalar normal solve and permutation/additivity controls',()=>{
 const single=require('../docs/influence.js');
 for(const c of native.cases){const w=c.witness,selected=new Set(w.rows),A=w.design.filter((_,i)=>!w.delete||!selected.has(i)),t=w.target.flatMap((v,i)=>w.delete&&selected.has(i)?[]:[v+(!w.delete&&selected.has(i)?w.deltas[w.rows.indexOf(i)]:0)]),d=w.query.length,aug=[...A,...Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>i===j?Math.sqrt(w.regularization):0))],y=[...t,...Array(d).fill(0)],H=Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>aug.reduce((s,a)=>s+a[i]*a[j],0))),rhs=Array.from({length:d},(_,i)=>aug.reduce((s,a,j)=>s+a[i]*y[j],0));close(single.solve(H,rhs),w.point,'direct augmented normal');if(!w.delete)assert(Math.abs(w.interaction)<1e-11);}
 const a=batch.response([[1,1],[1,1],[1,0]],[1,2,0],1,[0,1],[1,1],null,true);assert(Math.abs(a.interaction+.675)<1e-12);close(a.point,batch.response(a.design,a.target,1,[1,0],a.query,null,true).point);
});
test('strict batch and source identities, bounds and nonmutation',()=>{
 const A=[[1,2],[3,4]],t=[1,0],q=[1,1],snapshot=JSON.stringify([A,t,q]);batch.response(A,t,1,[0],q,[.5]);assert.equal(JSON.stringify([A,t,q]),snapshot);
 for(const rows of [[true],[0,0],[-1],[2]])assert.throws(()=>batch.response(A,t,1,rows,q));
 for(const delta of [[true],[NaN],[],[1,2]])assert.throws(()=>batch.response(A,t,1,[0],q,delta));
 assert.throws(()=>batch.response(A,t,1,[0],q,[0],true));
 const c=native.cases[0],w=c.witness,m=data.candidates.find(v=>v.id===c.model_id).model;
 assert.throws(()=>batch.blockResponse(data,m,w.side,w.index,[true],0,null,true));
 const foreign=data.split.train.find(j=>!w.block_source_rows.includes(j));assert.throws(()=>batch.blockResponse(data,m,w.side,w.index,[foreign],0,null,true));
 assert.throws(()=>batch.blockResponse(data,m,w.side,w.index,[w.block_source_rows[0]],0,[3],false));
});

test('lost numerical deletion positivity is refused, target edit remains allowed',()=>{
 assert.throws(()=>batch.response([[1],[1]],[1,2],1e-16,[0,1],[1],null,true));
 assert(Math.abs(batch.response([[1],[1]],[1,2],1e-16,[0,1],[1],[1,0],false).point[0]-2)<1e-12);
});
