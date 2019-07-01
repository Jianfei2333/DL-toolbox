  # # #
  # Modifying
  # # #

  # labels = np.zeros(data.__len__())
  
  # t = 0
  # for img, label in data:
  #   labels[t] = int(label)
  #   t += 1
  # print (min(labels))
  
  # val = None
  # # train = None
  # train_sample = None
  # val_sample = None
  # for i in range(10):
    # idx = np.arange(0,50000, 1)[labels == i]
    
    # folds
    
    # folds = np.array_split(idx, 5)
    # if val is not None:
    #   val = np.hstack((val, folds[0]))
    #   train = np.hstack((train, folds[1], folds[2], folds[3], folds[4]))
    # else:
    #   val = folds[0]
    #   train = np.hstack((folds[1], folds[2], folds[3], folds[4]))

    # small sample
    
    # sample_idx = np.random.choice(idx, 100)
    # sample_folds = np.array_split(sample_idx, 5)
    # if val_sample is not None:
    #   val_sample = np.hstack((val_sample, sample_folds[0]))
    #   train_sample = np.hstack((train_sample, sample_folds[1], sample_folds[2], sample_folds[3], sample_folds[4]))
    # else:
    #   val_sample = sample_folds[0]
    #   train_sample = np.hstack((sample_folds[1], sample_folds[2], sample_folds[3], sample_folds[4]))
  # np.save(DATAPATH+'train.npy', train)
  # np.save(DATAPATH+'val.npy', val)
  # np.save(DATAPATH+'sample_train.npy', train_sample)
  # np.save(DATAPATH+'sample_val.npy', val_sample)
