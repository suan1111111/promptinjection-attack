from fetch_candidates import get,root
import json,concurrent.futures
repos=['facebook/cyberseceval3-visual-prompt-injection','erjiaxiao/Typographic-Visual-Prompt-Injection-Dataset','Dagger7164/VisualPromptInjection']
def run(repo):
 d=get('https://huggingface.co/api/datasets/'+repo).json();(root/(repo.replace('/','__')+'.json')).write_text(json.dumps(d),encoding='utf8')
 print(repo,d.get('description','')[:1400].encode('ascii','backslashreplace').decode(),[x['rfilename'] for x in d.get('siblings',[])][:20],flush=True)
 print('card',str(d.get('cardData'))[:1400],flush=True)
with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(run,repos))
