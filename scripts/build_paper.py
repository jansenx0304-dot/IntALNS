"""从干净的临时编译目录生成唯一的论文 PDF。"""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    paper = ROOT/'paper'
    with tempfile.TemporaryDirectory(prefix='intalns_latex_') as directory:
        build = Path(directory)
        for source in paper.iterdir():
            if source.is_dir() and source.name in ('sections', 'figures', 'tables'):
                shutil.copytree(source, build/source.name)
            elif source.suffix in ('.tex', '.bib', '.cls', '.bst'):
                shutil.copy2(source, build/source.name)
        latex = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                 '-halt-on-error', '-file-line-error', 'main.tex']
        for command in (latex, ['bibtex', 'main'], latex, latex):
            result = subprocess.run(command, cwd=build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(result.stdout.decode('utf-8', errors='replace')[-12000:])
        log = (build/'main.log').read_text(encoding='utf-8', errors='replace')
        warnings = [line for line in log.splitlines() if 'undefined' in line.lower()
                    or 'Missing character:' in line or 'Overfull' in line]
        if warnings:
            raise RuntimeError('论文编译检查未通过：\n'+'\n'.join(warnings))
        shutil.copyfile(build/'main.pdf', paper/'main.pdf')
    print('论文编译完成：paper/main.pdf')

if __name__ == '__main__':
    main()
