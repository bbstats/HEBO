First install git and create conda enviroment

```bash
conda env create -f enviroment.yaml 
conda activate DGM
```
Install Absolut under antigenbinding (follow instructions on git repo)

https://github.com/csi-greifflab/Absolut

Install all pre-computed structures (~33GB)

```bash
cd Absolut
aria2c -i ../urls.txt --auto-file-renaming=false --continue=true
```

Run any baseline on said structure
