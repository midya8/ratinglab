"""Strict offline JSON commands. Unknown users are explicit, not silently reindexed."""
import argparse,json,sys
from pathlib import Path
from .experiment import saved,run,audit
from .controls import controls
from .core import prediction,block_witness

def read(path):
    def constant(v):raise ValueError('nonfinite JSON')
    def pairs(values):
        out={}
        for k,v in values:
            if k in out:raise ValueError('duplicate key '+k)
            out[k]=v
        return out
    return json.loads(Path(path).read_text(),parse_constant=constant,object_pairs_hook=pairs)

def main():
    p=argparse.ArgumentParser(description='Observed-rating ALS, not top-N utility');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('batch-export');a=s.add_parser('batch');a.add_argument('--input',required=True);
    s.add_parser('report');s.add_parser('controls');a=s.add_parser('audit');a.add_argument('--report');a=s.add_parser('fit');a.add_argument('--output',required=True);a.add_argument('--budget',type=int,default=80)
    for command in ['predict','witness']:
        a=s.add_parser(command);a.add_argument('--model-id')
        if command=='predict':a.add_argument('--input',required=True)
        else:a.add_argument('--side',choices=['user','item'],default='user');a.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    try:
        if args.command=='batch-export':
            from .batch_export import export
            out=export()
        elif args.command=='batch':
            from .batch import response
            d=read(args.input)
            if not isinstance(d,dict) or not {'design','target','regularization','rows','query'}<=set(d) or set(d)-{'design','target','regularization','rows','query','deltas','delete'}:raise ValueError('batch requires design, target, regularization, rows, query; optional deltas/delete only')
            out=response(**d)
        elif args.command=='report':
            r=saved();out={k:r[k] for k in ['shape','selected_id','selected_bias_id','global_test_rmse','cold','protocol','scope']};out['metrics']=[{k:c[k] for k in ['id','train','validation','test','status']} for c in r['candidates']+r['bias_candidates']]
        elif args.command=='controls':out=controls()
        elif args.command=='audit':out=audit(read(args.report) if args.report else saved())
        elif args.command=='fit':
            r=run(max_sweeps=args.budget);Path(args.output).write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');out=dict(output=args.output,selected_id=r['selected_id'],selected_bias_id=r['selected_bias_id'])
        else:
            r=saved();id=args.model_id or r['selected_id'];c=next((c for c in r['candidates']+r['bias_candidates'] if c['id']==id),None)
            if c is None:raise ValueError('unknown model')
            if args.command=='predict':
                d=read(args.input)
                if not isinstance(d,dict) or set(d)!={'pairs'} or not isinstance(d['pairs'],list) or not d['pairs'] or any(not isinstance(row,list) or len(row)!=2 or any(type(v) is not int for v in row) for row in d['pairs']):raise ValueError('pairs must contain integer [user,item] indices')
                out=dict(model_id=id,predictions=prediction(c['model'],[x[0] for x in d['pairs']],[x[1] for x in d['pairs']]).tolist(),scope=r['scope'])
            else:
                if not 0<=args.index<r['shape'][0 if args.side=='user' else 1]:raise ValueError('index out of range')
                out=block_witness(c['model'],[r['edges'][j] for j in r['split']['train']],args.side,args.index)
        print(json.dumps(out,indent=2,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,ArithmeticError,OSError) as e:print('ratinglab: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
