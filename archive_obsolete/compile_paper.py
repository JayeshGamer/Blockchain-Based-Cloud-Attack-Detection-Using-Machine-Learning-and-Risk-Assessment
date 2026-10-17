"""
Script to compile IEEE LaTeX paper to PDF.
Checks for pdflatex or tectonic on system path and compiles paper/ieee_paper.tex.
"""

import os
import shutil
import subprocess

def compile_paper():
    paper_dir = "paper"
    tex_file = "ieee_paper.tex"

    compiler = None
    for cand in ["pdflatex", "tectonic", "xelatex"]:
        if shutil.which(cand):
            compiler = cand
            break

    if not compiler:
        print("Notice: pdflatex / tectonic is not installed in the local PATH.")
        print("To compile the paper to PDF:")
        print("1. Install MiKTeX / TeX Live, or upload paper/ folder to Overleaf.")
        print("2. Run: pdflatex ieee_paper.tex && bibtex ieee_paper && pdflatex ieee_paper.tex")
        return

    print(f"Compiling with {compiler}...")
    cwd = os.path.abspath(paper_dir)
    subprocess.run([compiler, tex_file], cwd=cwd, check=True)
    if compiler == "pdflatex":
        subprocess.run(["bibtex", "ieee_paper"], cwd=cwd, check=False)
        subprocess.run([compiler, tex_file], cwd=cwd, check=True)
        subprocess.run([compiler, tex_file], cwd=cwd, check=True)
    print("Paper compilation completed: paper/ieee_paper.pdf")

if __name__ == "__main__":
    compile_paper()
