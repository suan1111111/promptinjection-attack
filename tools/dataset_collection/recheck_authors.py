import json,re,concurrent.futures
from fetch_candidates import get,root
repos=['trailofbits/anamorpher','uw-nsl/ArtPrompt','declare-lab/red-instruct','munzzyy/injection-fixtures']
def run(repo):
 try:
  meta=get('https://api.github.com/repos/'+repo).json();branch=meta['default_branch']
  tree=get(f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1').json()
  (root/(repo.replace('/','__')+'_tree.json')).write_text(json.dumps(tree),encoding='utf8')
  readme=get(f'https://raw.githubusercontent.com/{repo}/{branch}/README.md').text
  (root/(repo.replace('/','__')+'_README.md')).write_text(readme,encoding='utf8')
  print(repo,'sha',tree.get('sha'),'license',meta.get('license'),flush=True)
  print([x['path'] for x in tree.get('tree',[]) if x['path'].lower().endswith(('.png','.jpg','.webp','.zip','.csv'))][:70],flush=True)
 except Exception as e:print(repo,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(run,repos))
