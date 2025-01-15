import splitfolders

# directory name which contains dataset
dir = 'dataset'

# split dataset into train and val directory in 7 : 3 ratio
splitfolders.ratio(dir,"splitdataset",ratio=(0.7,0.2, 0.1))
