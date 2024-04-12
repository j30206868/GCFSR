#### clone
```
git clone https://github.com/j30206868/GCFSR.git
cd GCFSR
```
下載權重`https://drive.google.com/file/d/1Yzi1O5SeSFq_yrunrTJ1nkTT8jbBOaLJ/view`

#### 環境設置
```
conda create env --name GCFSR python==3.8
conda activate GCFSR
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116`
pip install -r requirements.txt  
```

#### setup安裝
```
### setup cuda path
export PATH=/home/cwz/cuda/cuda-11.6/bin:${PATH}
export LD_LIBRARY_PATH=/home/cwz/cuda/cuda-11.6/lib64:${LD_LIBRARY_PATH}

### setup install
python setup.py develop
```

#### 執行時需要設定`BASICSR_JIT`
> 參考`${GCFSR}/basicsr/ops/fused_act/fused_act.py`
```python
BASICSR_JIT = os.getenv('BASICSR_JIT')
if BASICSR_JIT == 'True':
    from torch.utils.cpp_extension import load
    module_path = os.path.dirname(__file__)
    fused_act_ext = load(
        'fused',
        sources=[
            os.path.join(module_path, 'src', 'fused_bias_act.cpp'),
            os.path.join(module_path, 'src', 'fused_bias_act_kernel.cu'),
        ],
    )
```

#### 執行
`python inference/inference_gcfsr_blind.py --model_path gcfsr_blind_512.pth --input inputs/whole_imgs`