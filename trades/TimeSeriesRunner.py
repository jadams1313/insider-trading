import os 
import sys
sys.path.append(os.path.abspath(r"C:\Users\User\OpenSourceRepos\Time-Series-Library"))
import argparse
import torch
import TimeSeries  # adjust import to match your file name
import os

def main():
    # ======== 1️⃣  Setup arguments ========
    parser = argparse.ArgumentParser(description='TimesNet Forecasting Example')

    parser.add_argument('--model', type=str, default='TimesNet')
    parser.add_argument('--data', type=str, default='custom')   # not M4
    parser.add_argument('--root_path', type=str, default='./')  # directory containing your bp.csv
    parser.add_argument('--data_path', type=str, default='bp.csv')  # your uploaded CSV
    parser.add_argument('--features', type=str, default='S', help='S: univariate, M: multivariate')
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='label length')
    parser.add_argument('--pred_len', type=int, default=24, help='forecast length')

    # training setup
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--loss', type=str, default='MSE')

    # device setup
    parser.add_argument('--use_gpu', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly')

    args = parser.parse_args([])  # use [] for interactive or notebook runs

    # ======== 2️⃣  Create experiment ========
    exp = TimeSeries(args)

    setting = f"{args.model}_{args.data}_seq{args.seq_len}_pred{args.pred_len}"

    print("Starting training...")
    exp.train(setting)

    print("Running test...")
    exp.test(setting, test=1)

    print("✅ Forecasting complete!")

if __name__ == '__main__':
    main()