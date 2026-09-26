FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pip==25.3 'setuptools<81'
RUN pip install --no-cache-dir numpy==1.26.4 torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl mmengine==0.10.7 mmdet==3.3.0 'opencv-python<4.12' 'yapf==0.40.1'
RUN python -c "import torch,mmcv,mmdet;from mmcv.ops import nms;print(torch.__version__,mmcv.__version__,mmdet.__version__);print(nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]),torch.tensor([.9,.8]),.5))"
WORKDIR /project
CMD ["python", "scripts/mmdet_pilot.py"]
