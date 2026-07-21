from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Solar, Dataset_Pred, Dataset_Exchange_Rate, Dataset_Electricity, Dataset_Weather, Dataset_Traffic
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'exchange_rate': Dataset_Exchange_Rate,
    'electricity': Dataset_Electricity,
    'weather': Dataset_Weather,
    'traffic': Dataset_Traffic,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only
    # Determine loading mode (train, test, predict)
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    dataset_len = len(data_set)
    print(flag, dataset_len)
    
    # Check if dataset is empty
    if dataset_len == 0:
        raise ValueError(
            f"Dataset is empty for {flag} set. "
            f"This usually happens when the data length is too short compared to seq_len ({args.seq_len}) + pred_len ({args.pred_len}). "
            f"Please check:\n"
            f"1. Data file exists and has enough rows\n"
            f"2. seq_len ({args.seq_len}) + pred_len ({args.pred_len}) is not too large\n"
            f"3. Data split boundaries are correct"
        )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
