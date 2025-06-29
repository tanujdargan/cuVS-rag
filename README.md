# cuVS-rag

Base Notebook From: https://gist.github.com/lowener/08eef6aca69cae5c2151224c801521b0

## Setup Environment

### Installing conda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

### Clone the repo
`git clone https://github.com/tanujdargan/cuVS-rag`

### Setup conda env

Install environment by doing: `conda create --name cuVStesting --file requirements.txt`
