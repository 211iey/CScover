import torch
import pandas as pd
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider

# ✅ 경로 설정
input_path = './dataset/ETT-small/cluster0_2024_filtered.csv'
output_path = './dataset/ETT-small/cluster0_2024_predicted.csv'
checkpoint_path = 'checkpoint저장소/cluster0_checkpoint.pth'

# ✅ 설정값 정의
args = Namespace(
    is_training=0,
    task_name='long_term_forecast',
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
    num_workers=0,
    gpu=0,
    gpu_type='cuda',
    use_multi_gpu=False,
    devices='0',
    root_path='./dataset/ETT-small/',
    data_path='cluster0_2024_filtered.csv',
    target='train_subway21.congestion',
    freq='h',
    checkpoints='./checkpoints/',
    dropout=0.1,
    embed='timeF',
    activation='gelu',
    distil=True,
    factor=1,
    loss='MSE',
    lradj='type1',
    seasonal_patterns='weekly',
    use_amp=False,
    inverse=False,
    p_hidden_dims=[128, 128],
    p_hidden_layers=2,
    batch_size=32,
    use_dtw=False,
    extra_tag='cluster5_eval',
    top_k=5,
    num_kernels=6
)

# ✅ 모델 초기화
args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
exp = Exp_Long_Term_Forecast(args)
exp.model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
exp.model.eval()

# ✅ 데이터 로드
df = pd.read_csv(input_path)
test_data, test_loader = data_provider(args, flag='test')

# ✅ 예측 수행
# ✅ 예측 수행 (100개 정도만 예측용)
print(f">>>>>>> Running fast inference (first 100 samples) on {args.data_path}")
all_preds = []
total_preds = 0
max_preds = 100

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Predicting (fast test)"):
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

    with torch.no_grad():
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        pred = outputs.detach().cpu().numpy()  # (B, pred_len, 1)
        B, L, C = pred.shape
        all_preds.append(pred)

        total_preds += B * L
        if total_preds >= max_preds:
            break

# ✅ 예측값 정리
pred_array = np.concatenate(all_preds, axis=0).squeeze()
pred_array = pred_array[:max_preds]  # 정확히 100개로 자르기
pred_array = pred_array.reshape(-1)

# ✅ 길이 맞추기
congestion_pred = np.full(len(df), np.nan)
start_idx = len(congestion_pred) - len(pred_array)
congestion_pred[start_idx:] = pred_array

# ✅ 저장
df['pred_congestion'] = congestion_pred
test_output_path = './dataset/ETT-small/cluster0_2024_predicted_test100.csv'
df.to_csv(test_output_path, index=False)
print(f"✅ (100개 예측) 저장 완료 → {test_output_path}")
