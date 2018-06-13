# Binary_GRU

### Description
```
python main.py -h
```

### Dataset

Link: https://iu.box.com/s/p9okuh3l91wwq66zi0cq1oiw01vw8l8e

File: data_xl_m.pkl 

Size: 6.8 GB

Contents: Object containing two Dictionaries, train and test. 
&nbsp;&nbsp;train Keys: ‘M’, ‘M_bin’, ‘y’
&nbsp;&nbsp;&nbsp;&nbsp;‘M’: Array of 1200 mixed signal STFTs. Each STFT has shape (n, 513), where n varies. 1200 signals were created with 120 clean signals mixed with 10 noise signals. So every 10 signals of ‘M' consists of same n since the same clean signal was used to be mixed with 10 different noises.
&nbsp;&nbsp;&nbsp;&nbsp;‘M_bin’: Array of 1200 ‘M’ signals quantized with 100% GMM sampling. Each STFT has shape (n, 2052).
&nbsp;&nbsp;&nbsp;&nbsp;‘y’: Array of 1200 target IBMs. Each with shape (n, 513).
&nbsp;&nbsp;test Keys: ‘M’, ‘M_bin’, ‘y’, ’S.
&nbsp;&nbsp;&nbsp;&nbsp;‘M’, ‘M_bin’, and ‘y' are the same as train except there are 400 signals instead of 1200 (since 4 speakers were used)
&nbsp;&nbsp;&nbsp;&nbsp;’S’: Array if 40 original signal STFTs. Each with shape (n, 513).

&nbsp;&nbsp; *Refer to https://github.iu.edu/kimsunw/Binary_GRU/blob/master/DataSets.py for hard-details*

Example Usage:
```
  with open(‘data_xl_m.pkl', 'rb') as f: 
    data = pickle.load(f) 
  train_data = data.train
  test_data = data.test
  mixed_signal_stfts = train_data[’M’]
```



### Example: Running Pretraining
```
python main.py 4e-5 1 90 150 1 -d large -n few -k data_l_f.pkl -b 4 -e
```
