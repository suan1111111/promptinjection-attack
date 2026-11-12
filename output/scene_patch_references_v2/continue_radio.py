"""Continue the saved pilot and audit five training questions with the loaded model."""
import sys, json, shutil
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pixel_attack_trials'))
import optimize_qwen_pixels as base
previous=json.loads((Path(__file__).parent/'training_eps16_launch.json').read_text(encoding='utf-8-sig'))
old=Path(previous['directory'])/'run'
directory=ROOT/'outputs'/('radio_reference_patch_eps16_100steps_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
run=directory/'run'
shutil.copytree(old,run)
arguments=previous['arguments'][2:]
arguments[arguments.index('--steps')+1]='100'
arguments[arguments.index('--run-dir')+1]=str(run)
arguments+=['--resume',str(run/'checkpoint.pt')]
(Path(__file__).parent/'training_100steps_launch.json').write_text(json.dumps(dict(directory=str(directory),arguments=arguments),indent=2),encoding='utf-8')
original_evaluate=base.evaluate_saved_images

def evaluate_and_audit(args,model,processor,run_dir,torch):
    result=original_evaluate(args,model,processor,run_dir,torch)
    from PIL import Image
    rows=[r for r in base.read_dataset_rows(args.csv) if r['split']=='train'][:5]
    records=[]
    for row in rows:
        record={'question':row['question'],'target_answer':row['target_answer']}
        for kind,filename in [('reference','constraint_reference.png'),('adversarial','adversarial_final.png')]:
            image=Image.open(run_dir/filename).convert('RGB')
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['question']}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[prompt],images=[image],return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=args.eval_max_new_tokens)
            answer=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            record[kind+'_answer']=answer
            marker=row.get('target_contains','')
            record[kind+'_marker_match']=bool(marker and marker in answer)
            del inputs,generated
        records.append(record)
        base.write_json(run_dir/'training_question_audit.json',{'records':records,'complete':len(records)==len(rows),'notice':'First five TRAINING questions; not a held-out test.'})
    base.status('Training-question audit: '+str(sum(r['adversarial_marker_match'] for r in records))+'/'+str(len(records)))
    return result

base.evaluate_saved_images=evaluate_and_audit
sys.argv=[str(ROOT/'pixel_attack_trials/optimize_qwen_pixels.py')]+arguments
print('Output: '+str(directory),flush=True)
raise SystemExit(base.main())
