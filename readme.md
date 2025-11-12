# Moon Location
## 1. Installation
### 1.1 Create and Activate a Conda Environment
```bash
conda create -n moon_location
conda activate moon_location
```

### 1.2 Install Required Libraries
```bash
pip install -r requirements.txt
```
- Torch is an essential library for running SuperPoint and SuperGlue.
- You can download the appropriate version from the official PyTorch website (https://pytorch.org/) or refer to the installation instructions on the SuperGlue official repository (https://github.com/magicleap/SuperGluePretrainedNetwork).

### 1.3 Install ccmaster
1. Download the ContextCapture software from www.bentley.com and install it.
2. Locate the corresponding whl file in the installation path: `Bentley\ContextCapture Center\sdk\dist`
3. Install the whl file using pip (replace the filename with the actual version):
```bash
pip install ccmasterkernel-x.x.x.x-cpxx-cpxxm-win_amd64.whl
```

---

## 2. Running
### Method 1: Launch the Software Interface Directly
```bash
python main.py
```

### Method 2: Run Step-by-Step to Obtain Intermediate Results
1. Run the code in the `utils` directory in sequence. This method helps you better obtain intermediate results.
2. For detailed steps, refer to the instructions in `Threadclass.py`.