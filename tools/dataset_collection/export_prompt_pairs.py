"""Export only source-provided user prompts; never synthesize missing prompts."""
import json,csv
from pathlib import Path
root=Path('data/datasets/text_prompt_injection_50')
path=root/'manifest.json';rows=json.loads(path.read_text(encoding='utf8'))
paired=[];missing=[]
for row in rows:
 prompt=row.get('user_task')
 if isinstance(prompt,str) and prompt.strip():
  source=row['source_dataset']
  row['user_prompt']=prompt
  row['user_prompt_status']='source_provided'
  row['user_prompt_source_field']='user_input_text' if source.startswith('facebook/') else 'instruction'
  prompt_file=Path(row['file']).with_suffix('.prompt.txt')
  (root/prompt_file).write_text(prompt,encoding='utf8')
  row['user_prompt_file']=prompt_file.as_posix()
  paired.append(row)
 else:
  row['user_prompt']=None
  row['user_prompt_status']='missing_from_verified_source'
  row['user_prompt_missing_reason']='Source does not supply a full per-image user prompt; attack content, objective and expected answer are not substitutes.'
  missing.append(row)
path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
with (root/'paired_samples.jsonl').open('w',encoding='utf8') as f:
 for row in paired:
  f.write(json.dumps({k:row.get(k) for k in ['file','user_prompt','user_prompt_file','user_prompt_source_field','system_prompt','source_dataset','task_id']},ensure_ascii=False)+'\n')
with (root/'missing_user_prompts.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['file','source_dataset','task_id','user_prompt_missing_reason'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(missing)
for row in paired:
 assert (root/row['user_prompt_file']).read_text(encoding='utf8')==row['user_task']
print('Complete source-provided pairs:',len(paired),'Missing:',len(missing))
