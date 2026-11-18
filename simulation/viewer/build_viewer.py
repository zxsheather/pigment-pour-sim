"""Bundle the 3D viewer, source photo and GLB into one offline HTML file."""
from pathlib import Path
import base64
import json
import subprocess

HERE=Path(__file__).resolve().parent
SIM=HERE.parent
ROOT=SIM.parent

def main():
    subprocess.run([str(HERE/'node_modules/.bin/esbuild'),str(HERE/'app.js'),'--bundle','--minify','--format=iife','--target=es2020','--outfile='+str(HERE/'app.bundle.js')],check=True)
    template=(HERE/'template.html').read_text()
    config=(SIM/'world_model/scene_config.json').read_text()
    photo='data:image/jpeg;base64,'+base64.b64encode((ROOT/'场地照片/任务场地.jpg').read_bytes()).decode()
    robot=base64.b64encode((SIM/'reconstruction/quanta_x1.glb').read_bytes()).decode()
    js=(HERE/'app.bundle.js').read_text().replace('</script','<\\/script')
    html=template.replace('__CONFIG__',config).replace('__PHOTO__',json.dumps(photo)).replace('__ROBOT__',json.dumps(robot)).replace('__APP__',js)
    output=ROOT/'场景重建.html';output.write_text(html)
    print(f'Built {output} ({output.stat().st_size/1e6:.2f} MB)')

if __name__=='__main__':main()
