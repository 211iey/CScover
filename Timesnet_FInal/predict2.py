import torch
import pandas as pd
from argparse import Namespace
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# 1. 체크포인트 경로
checkpoint_path = 'checkpoint저장소/cluster0_checkpoint.pth'

# 2. 설정값 정의
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
    batch_size=1,
    use_dtw=False,
    extra_tag='cluster5_eval',
    top_k=5,
    num_kernels=6
)

# 3. 디바이스 설정 및 모델 초기화
args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
exp = Exp_Long_Term_Forecast(args)
exp.model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
exp.model.eval()

# 4. 데이터 로드
from data_provider.data_factory import data_provider
test_data, test_loader = data_provider(args, flag='test')

# 5. 예측 실행 (배치 단위)
preds = []
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

    with torch.no_grad():
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        pred = outputs.detach().cpu().numpy()
        preds.append(pred)

# 6. 예측값 후처리
import numpy as np
preds = np.concatenate(preds, axis=0)  # [B, L, D]
pred_flat = preds.reshape(-1)  # [L]

# 7. 원본 CSV 불러와서 붙이기
df = pd.read_csv(args.root_path + args.data_path)
df_pred = df.copy()

# 마지막 pred_len 길이만큼 predicted_congestion 컬럼 생성
df_pred.loc[len(df_pred)-args.pred_len:, 'predicted_congestion'] = pred_flat

# 8. 저장
output_path = args.root_path + 'cluster0_2024_with_prediction.csv'
df_pred.to_csv(output_path, index=False)
print(f"✅ 예측 결과가 저장되었습니다 → {output_path}")
