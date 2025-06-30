import pandas as pd
import numpy as np
from argparse import Namespace
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# 1. 경로 및 설정
full_path = r"C:\Users\andyt\Downloads\Time-Series-Library-main\Time-Series-Library-main\dataset\ETT-small\cluster0_2541_5_1.csv"
df = pd.read_csv(full_path)

# 2. 마지막 96개만 추출
df_tail_96 = df.iloc[-96:].copy()

# 3. 추론용 임시 CSV 저장
infer_input_path = r"C:\Users\andyt\Downloads\Time-Series-Library-main\Time-Series-Library-main\dataset\ETT-small\cluster0_tail96_infer.csv"
df_tail_96.to_csv(infer_input_path, index=False)

# 4. TimesNet 세팅
args = Namespace(
    is_training=0,
    task_name='long_term_forecast',  # ✅ 이거 빠져있어서 에러났던 거
    model_id='subway21_96_96',
    model='TimesNet',
    data='custom',
    features='M',
    seq_len=96,
    label_len=48,
    pred_len=96,
    e_layers=2,
    d_model=64,
    d_ff=256,
    enc_in=9,
    dec_in=9,
    c_out=1,
    des='subway_congestion',
    itr=1,
    use_gpu=True,
    gpu=0,
    gpu_type='cuda',
    use_multi_gpu=False,
    devices='0',
    root_path='./dataset/ETT-small/',
    data_path='cluster0_2024_filtered_small2.csv',
    target='train_subway21.congestion',
    embed='timeF',
    dropout=0.1,
    activation='gelu',
    factor=1,
    
    freq='h',  # ✅ 요거 추가해줘야 에러 해결됨
    loss='MSE',
    lradj='type1',
    seasonal_patterns='weekly',
    checkpoints='./checkpoints',
    p_hidden_dims=[128, 128],
    p_hidden_layers=2,
    batch_size=64,
    top_k=5,
    num_kernels=6,
    num_workers=6,
    distil=True
)
# 5. 모델 로딩 및 추론
exp = Exp_Long_Term_Forecast(args)
exp.test(setting='infer_tail96_cluster0', test=1)
